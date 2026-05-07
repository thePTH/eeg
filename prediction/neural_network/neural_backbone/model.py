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
    
    

