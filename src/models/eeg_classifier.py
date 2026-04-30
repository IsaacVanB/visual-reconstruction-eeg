from typing import Any

import torch
from torch import nn


class ChannelSoftmax(nn.Module):
    """Apply softmax across Conv2d feature channels."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)


class ChannelLayerNorm2d(nn.Module):
    """LayerNorm over channels for NCHW tensors, matching Keras channel-last LayerNorm."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(x.shape)}")
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class EEGClassifier20CNN(nn.Module):
    """Notebook-style 2D CNN classifier for EEG trials shaped [B, C, T]."""

    def __init__(
        self,
        eeg_channels: int,
        eeg_timesteps: int,
        num_classes: int = 20,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if eeg_channels <= 0 or eeg_timesteps <= 0:
            raise ValueError("eeg_channels and eeg_timesteps must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")

        self.eeg_channels = int(eeg_channels)
        self.eeg_timesteps = int(eeg_timesteps)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3)),
            ChannelSoftmax(),
            ChannelLayerNorm2d(16),
            nn.MaxPool2d(kernel_size=(3, 2)),
            nn.Dropout2d(p=0.25),
            nn.Conv2d(16, 40, kernel_size=(3, 3)),
            nn.ELU(),
            ChannelLayerNorm2d(40),
            nn.MaxPool2d(kernel_size=(3, 2)),
            nn.Dropout2d(p=0.30),
        )
        flat_dim = self._infer_flat_dim()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    @property
    def dense_l1_parameters(self):
        return (self.classifier[1].weight,)

    def _infer_flat_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.eeg_channels, self.eeg_timesteps)
            out = self.features(dummy)
        return int(out.flatten(start_dim=1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.eeg_channels:
            raise ValueError(
                f"Expected EEG channels C={self.eeg_channels}, got C={int(x.shape[1])}."
            )
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


def extract_eeg_classifier20_arch_metadata(model: EEGClassifier20CNN) -> dict[str, Any]:
    return {
        "frontend": "conv2d_notebook_classifier",
        "input_channels": int(model.eeg_channels),
        "input_timesteps": int(model.eeg_timesteps),
        "conv_filters": [16, 40],
        "kernels": [[3, 3], [3, 3]],
        "pooling": [[3, 2], [3, 2]],
        "activation": ["channel_softmax", "elu", "elu"],
        "norm": "channel_layernorm",
        "dropout": [0.25, 0.30],
        "hidden_dim": int(model.hidden_dim),
        "num_classes": int(model.num_classes),
    }
