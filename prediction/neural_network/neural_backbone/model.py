import torch.nn as nn
import torch

# ==========================================
# 4. Stronger Neural Backbone (1D ResNet-style)
# ==========================================
class ConvBlock1D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_c, out_c, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    



class DeepEEGNet(nn.Module):
    def __init__(self, in_channels=19):
        super().__init__()
        self.extractor = nn.Sequential(
            ConvBlock1D(in_channels, 32, stride=2),
            nn.Dropout(0.2),
            ConvBlock1D(32, 64, stride=2),
            nn.Dropout(0.2),
            ConvBlock1D(64, 128, stride=2),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 1) # Outputs LOGITS, not probabilities

    def forward(self, x_raw):
        h = self.extractor(x_raw).squeeze(-1)
        return self.fc(h).squeeze(-1) # Shape: [batch_size]
    
    




class MultiScaleDeepEEGNet(nn.Module):
    
    def __init__(self, in_nch=19, first_layer_ch = 32, lstm_nch=16, post_lin_weights=[16], out_nch=1):
        super().__init__()

        self.in_nch = in_nch
        self.out_nch = out_nch
        self.nlayers = 2
        self.nhid = lstm_nch

        layers= []
        layers.append(
            nn.Conv1d(
                in_channels=in_nch,
                out_channels=first_layer_ch,
                kernel_size=10,
                stride=10,
                padding=0,
                bias=True
            )
        )
        layers.append(nn.BatchNorm1d(first_layer_ch))
        layers.append(nn.ReLU(inplace=False))       

        self.time_layers_s1 = nn.Sequential(*layers)

        layers= []
        layers.append(
            nn.Conv1d(
                in_channels=in_nch,
                out_channels=first_layer_ch,
                kernel_size=5,
                stride=5,
                padding=0,
                bias=True
            )
        )
        layers.append(nn.BatchNorm1d(first_layer_ch))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.AvgPool1d(2,2))

        self.time_layers_s2 = nn.Sequential(*layers)

        layers= []
        layers.append(
            nn.Conv1d(
                in_channels=in_nch,
                out_channels=first_layer_ch,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True
            )
        )
        layers.append(nn.BatchNorm1d(first_layer_ch))
        layers.append(nn.ReLU(inplace=False))
        layers.append(nn.AvgPool1d(5,5))


        self.time_layers_s3 = nn.Sequential(*layers)


        self.lstm = nn.LSTM(
            input_size=first_layer_ch*3, #*3
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            batch_first=True,
            bidirectional=False
        )
        self.time_norm = nn.BatchNorm1d(self.nhid)


        layers = []
        for i in range(len(post_lin_weights)):

            if i == 0:
                out_channels = int(post_lin_weights[i])
                layers.append(
                    nn.Linear(in_features=self.nhid,
                              out_features=out_channels, bias=True)
                )
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(inplace=False))
                in_channels = out_channels
            else:
                out_channels = int(post_lin_weights[i])
                layers.append(
                    nn.Linear(
                        in_features=in_channels,
                        out_features=out_channels, bias=True)
                )
                layers.append(nn.BatchNorm1d(out_channels))
                layers.append(nn.ReLU(inplace=False))
                in_channels = out_channels

        layers.append(nn.Linear(in_features=in_channels,
                                out_features=self.out_nch, bias=True))

        self.post_lstm = nn.Sequential(*layers)


        self.last_s1f = None
        self.last_s2f = None
        self.last_s3f = None

    def multiscaleFE(self, x):   
        x_s1 = self.time_layers_s1(x)
        x_s2 = self.time_layers_s2(x)
        x_s3 = self.time_layers_s3(x)
        
        self.last_s1f = x_s1
        self.last_s2f = x_s2
        self.last_s3f = x_s3
        
        x = torch.cat([x_s1, x_s2, x_s3], 1)
        return x


    def forward(self, inpt):
        bb, _, ll = inpt.shape

        x = self.multiscaleFE(inpt)

        lstm_out, _ = self.lstm(x.view(bb,ll//10,-1).contiguous())
        lstm_out = self.time_norm(lstm_out[:,-1,:])
        out = self.post_lstm(lstm_out)        

        #return out
        return out 