import torch
import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Basic 1D convolution block: Conv1d + BatchNorm1d + ReLU."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel_size: int = 7,
        stride: int = 1,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_c,
            out_c,
            kernel_size,
            stride,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block."""
        return self.relu(self.bn(self.conv(x)))


class DeepEEGNet(nn.Module):
    """Simple 1D CNN backbone for EEG binary classification."""

    def __init__(self, in_channels: int = 19):
        super().__init__()

        self.extractor = nn.Sequential(
            ConvBlock1D(in_channels, 32, stride=2),
            nn.Dropout(0.2),
            ConvBlock1D(32, 64, stride=2),
            nn.Dropout(0.2),
            ConvBlock1D(64, 128, stride=2),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x_raw
            EEG tensor with shape ``[batch_size, n_channels, n_times]``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``[batch_size]``.
        """
        h = self.extractor(x_raw).squeeze(-1)
        return self.fc(h).squeeze(-1)


class MultiScaleDeepEEGNet(nn.Module):
    """Multi-scale 1D CNN + LSTM EEG backbone."""

    def __init__(
        self,
        in_nch: int = 19,
        first_layer_ch: int = 32,
        lstm_nch: int = 16,
        post_lin_weights: list[int] | None = None,
        out_nch: int = 1,
    ):
        super().__init__()

        if post_lin_weights is None:
            post_lin_weights = [16]

        self.in_nch = in_nch
        self.out_nch = out_nch
        self.nlayers = 2
        self.nhid = lstm_nch

        self.time_layers_s1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_nch,
                out_channels=first_layer_ch,
                kernel_size=10,
                stride=10,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm1d(first_layer_ch),
            nn.ReLU(inplace=False),
        )

        self.time_layers_s2 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_nch,
                out_channels=first_layer_ch,
                kernel_size=5,
                stride=5,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm1d(first_layer_ch),
            nn.ReLU(inplace=False),
            nn.AvgPool1d(2, 2),
        )

        self.time_layers_s3 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_nch,
                out_channels=first_layer_ch,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.BatchNorm1d(first_layer_ch),
            nn.ReLU(inplace=False),
            nn.AvgPool1d(5, 5),
        )

        self.lstm = nn.LSTM(
            input_size=first_layer_ch * 3,
            hidden_size=self.nhid,
            num_layers=self.nlayers,
            batch_first=True,
            bidirectional=False,
        )

        self.time_norm = nn.BatchNorm1d(self.nhid)

        post_layers: list[nn.Module] = []
        in_channels = self.nhid

        for out_channels in post_lin_weights:
            out_channels = int(out_channels)

            post_layers.extend(
                [
                    nn.Linear(
                        in_features=in_channels,
                        out_features=out_channels,
                        bias=True,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=False),
                ]
            )

            in_channels = out_channels

        post_layers.append(
            nn.Linear(
                in_features=in_channels,
                out_features=self.out_nch,
                bias=True,
            )
        )

        self.post_lstm = nn.Sequential(*post_layers)

        self.last_s1f = None
        self.last_s2f = None
        self.last_s3f = None

    def multiscaleFE(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate multi-scale temporal features."""
        x_s1 = self.time_layers_s1(x)
        x_s2 = self.time_layers_s2(x)
        x_s3 = self.time_layers_s3(x)

        self.last_s1f = x_s1
        self.last_s2f = x_s2
        self.last_s3f = x_s3

        return torch.cat([x_s1, x_s2, x_s3], dim=1)

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inpt
            EEG tensor with shape ``[batch_size, n_channels, n_times]``.

        Returns
        -------
        torch.Tensor
            Logits with shape ``[batch_size, out_nch]``.
        """
       
        x = self.multiscaleFE(inpt)
        x = x.permute(0, 2, 1).contiguous()

        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.time_norm(last_hidden)

        return self.post_lstm(last_hidden)