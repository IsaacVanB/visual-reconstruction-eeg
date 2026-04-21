from typing import Any, Tuple

import torch
from torch import nn


class EEGEncoderCNN(nn.Module):
    """1D CNN encoder for EEG trials shaped [B, C, T]."""

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

        # Requested architecture:
        # Conv1d 17->64 k7 p3 -> GN -> GELU
        # Conv1d 64->128 k5 p2 s2 -> GN -> GELU
        # Conv1d 128->256 k5 p2 s2 -> GN -> GELU
        # Conv1d 256->256 k3 p1 -> GELU
        # Global average pool over time
        # Linear 256->256 -> GELU -> Dropout(0.1) -> Linear 256->output_dim
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=17, out_channels=64, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.GELU(),
            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.GELU(),
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.GELU(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, output_dim),
        )

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != 17:
            raise ValueError(
                f"Expected EEG channels C=17 for this architecture, got C={int(x.shape[1])}."
            )
        x = self.features(x)
        x = self.global_pool(x).squeeze(-1)  # [B, 256]
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
        "frontend": "conv1d",
        "in_channels": 17,
        "channels": [64, 128, 256, 256],
        "kernels": [7, 5, 5, 3],
        "strides": [1, 2, 2, 1],
        "paddings": [3, 2, 2, 1],
        "norm": "groupnorm",
        "activation": "gelu",
        "global_pool": "adaptive_avg_pool1d_1",
        "head_hidden_dim": 256,
        "head_dropout": 0.1,
    }
