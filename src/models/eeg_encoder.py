from typing import Any, Tuple

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
        eeg_channels: int,
        eeg_timesteps: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        if eeg_channels <= 0 or eeg_timesteps <= 0:
            raise ValueError("eeg_channels and eeg_timesteps must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")

        # Block 1: temporal filtering.
        # Change kernel/pool here to alter temporal context and compression.
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=(1, 51),
                padding=(0, 25),
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 2)),
        )
        # Block 2: depthwise spatial mixing across EEG channels.
        # kernel_size=(eeg_channels,1) collapses spatial channel dimension to 1.
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(eeg_channels, 1),
                groups=32,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
        )
        # Block 3: additional temporal modeling + downsampling.
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(1, 13),
                padding=(0, 6),
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 5)),
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
            nn.Dropout(p=0.3),
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


def extract_eeg_encoder_cnn_arch_metadata(model: EEGEncoderCNN) -> dict[str, Any]:
    """Serialize architecture-defining values used to construct this model instance."""
    conv1 = model.block1[0]
    pool1 = model.block1[3]
    conv2 = model.block2[0]
    conv3 = model.block3[0]
    pool3 = model.block3[3]
    dropout = model.head[2]
    return {
        "temporal_filters": int(conv1.out_channels),
        "depth_multiplier": int(conv2.out_channels // conv1.out_channels),
        "temporal_kernel1": int(conv1.kernel_size[1]),
        "temporal_kernel3": int(conv3.kernel_size[1]),
        "pool1": int(pool1.kernel_size[1]),
        "pool3": int(pool3.kernel_size[1]),
        "dropout": float(dropout.p),
    }
