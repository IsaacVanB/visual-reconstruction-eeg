from typing import Tuple

import torch
from torch import nn


class EEGEncoderCNN(nn.Module):
    """CNN encoder for EEG trials shaped [B, C, T]."""

    def __init__(
        self,
        eeg_channels: int = 17,
        eeg_timesteps: int = 100,
        output_dim: int = 512,
        temporal_filters: int = 32,
        depth_multiplier: int = 2,
        temporal_kernel1: int = 51,
        temporal_kernel3: int = 13,
        pool1: int = 2,
        pool3: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if temporal_kernel1 <= 0 or temporal_kernel3 <= 0:
            raise ValueError("temporal kernels must be positive.")
        if pool1 <= 0 or pool3 <= 0:
            raise ValueError("pool sizes must be positive.")

        block2_channels = temporal_filters * depth_multiplier

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=temporal_filters,
                kernel_size=(1, temporal_kernel1),
                padding=(0, temporal_kernel1 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool1)),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_filters,
                out_channels=block2_channels,
                kernel_size=(eeg_channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(block2_channels),
            nn.ELU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=block2_channels,
                out_channels=block2_channels,
                kernel_size=(1, temporal_kernel3),
                padding=(0, temporal_kernel3 // 2),
                bias=False,
            ),
            nn.BatchNorm2d(block2_channels),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, pool3)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, eeg_channels, eeg_timesteps)
            features = self._forward_features(dummy)
            flat_dim = int(features.flatten(start_dim=1).shape[1])

        self.head = nn.Sequential(
            nn.Linear(flat_dim, 1024),
            nn.ELU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, output_dim),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        x = x.unsqueeze(1)  # [B, 1, C, T]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.flatten(start_dim=1)
        return self.head(x)


def infer_eeg_shape(sample: torch.Tensor) -> Tuple[int, int]:
    if sample.ndim != 2:
        raise ValueError(f"Expected single EEG sample [C, T], got {tuple(sample.shape)}")
    return int(sample.shape[0]), int(sample.shape[1])
