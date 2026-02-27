from typing import Tuple

import torch
from torch import nn


class EEGEncoderCNN(nn.Module):
    """CNN encoder for EEG trials shaped [B, C, T].

    Architecture experimentation guide:
    - Safe to change:
      - `temporal_filters`, `depth_multiplier` (capacity/width).
      - `temporal_kernel1`, `temporal_kernel3` (temporal receptive field).
      - `pool1`, `pool3` (temporal downsampling).
      - hidden head width (`1024`) and dropout/activation choices.
    - Must stay consistent:
      - Input convention is [B, C, T] where C=eeg channels, T=time points.
      - Final projection dim must match training target dim (`output_dim`).
      - Grouped conv in block2 requires `out_channels` divisible by `groups`.
    """

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

        # Constraint from grouped convolution below:
        # groups=temporal_filters means out_channels must be a multiple of temporal_filters.
        block2_channels = temporal_filters * depth_multiplier

        # Block 1: temporal filtering.
        # Change kernel/pool here to alter temporal context and compression.
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
        # Block 2: depthwise spatial mixing across EEG channels.
        # kernel_size=(eeg_channels,1) collapses spatial channel dimension to 1.
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
        # Block 3: additional temporal modeling + downsampling.
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

        # Shape inference keeps head dimensions valid after architecture edits.
        with torch.no_grad():
            dummy = torch.zeros(1, eeg_channels, eeg_timesteps)
            features = self._forward_features(dummy)
            flat_dim = int(features.flatten(start_dim=1).shape[1])

        # Projection head:
        # - You can replace with deeper MLP, residual MLP, or layer norm variants.
        # - Final layer output must remain `output_dim`.
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
    """Helper for dynamic model init from one dataset sample [C, T]."""
    if sample.ndim != 2:
        raise ValueError(f"Expected single EEG sample [C, T], got {tuple(sample.shape)}")
    return int(sample.shape[0]), int(sample.shape[1])
