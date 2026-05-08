from typing import Any, Tuple

import torch
from torch import nn


class EEGEncoderCNN(nn.Module):
    """EEGNet-style CNN encoder for EEG trials shaped [B, C, T]."""

    def __init__(
        self,
        eeg_channels: int,
        eeg_timesteps: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        if eeg_channels <= 0 or eeg_timesteps <= 0:
            raise ValueError("eeg_channels and eeg_timesteps must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        self.eeg_channels = int(eeg_channels)
        self.eeg_timesteps = int(eeg_timesteps)
        self.output_dim = int(output_dim)

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(1, 15),
                padding=(0, 7),
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(self.eeg_channels, 1),
                groups=32,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 9),
                padding=(0, 4),
                groups=64,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(1, 5),
                stride=(1, 2),
                padding=(0, 2),
                groups=128,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU(),
            nn.Dropout2d(p=0.1),
        )
        self.feature_dim = 128
        self.pool_bins = 8
        self.pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim * self.pool_bins, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, output_dim),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.eeg_channels:
            raise ValueError(
                f"Expected EEG channels C={self.eeg_channels}, got C={int(x.shape[1])}."
            )
        if x.shape[2] != self.eeg_timesteps:
            raise ValueError(
                f"Expected EEG timesteps T={self.eeg_timesteps}, got T={int(x.shape[2])}."
            )
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.features(x)
        x = x.squeeze(2)    # [B, 128, T']
        x = self.pool(x)    # [B, 128, 8]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        return self.head(x)


def infer_eeg_shape(sample: torch.Tensor) -> Tuple[int, int]:
    """Helper for dynamic model init from one dataset sample [C, T]."""
    if sample.ndim != 2:
        raise ValueError(f"Expected single EEG sample [C, T], got {tuple(sample.shape)}")
    return int(sample.shape[0]), int(sample.shape[1])


def extract_eeg_encoder_cnn_arch_metadata(_model: EEGEncoderCNN) -> dict[str, Any]:
    """Serialize architecture-defining values for robust checkpoint reload validation."""
    return {
        "frontend": "eegnet_depthwise_separable_adaptive_pool",
        "input_shape": [
            int(_model.eeg_channels),
            int(_model.eeg_timesteps),
        ],
        "temporal_conv": {
            "in_channels": 1,
            "out_channels": 32,
            "kernel": (1, 15),
            "padding": (0, 7),
        },
        "depthwise_spatial_conv": {
            "in_channels": 32,
            "out_channels": 64,
            "kernel": (int(_model.eeg_channels), 1),
            "groups": 32,
        },
        "separable_temporal_blocks": [
            {
                "depthwise_kernel": (1, 9),
                "depthwise_stride": (1, 1),
                "depthwise_padding": (0, 4),
                "pointwise_out_channels": 128,
            },
            {
                "depthwise_kernel": (1, 5),
                "depthwise_stride": (1, 2),
                "depthwise_padding": (0, 2),
                "pointwise_out_channels": 128,
            },
        ],
        "norm": "groupnorm",
        "activation": "gelu",
        "pool": "adaptive_avg_pool_time",
        "pool_bins": int(_model.pool_bins),
        "feature_dim": int(_model.feature_dim),
        "head_input_dim": int(_model.feature_dim * _model.pool_bins),
        "head_hidden_dims": [512, 256],
        "feature_dropout2d": 0.1,
        "head_dropout": [0.2, 0.2],
    }
