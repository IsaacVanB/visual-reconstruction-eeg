from typing import Any, Optional

import torch
from torch import nn


SUPPORTED_CLASSIFIER_ARCHITECTURES = {"cnn", "eegnet"}


def resolve_classifier_architecture_name(architecture: str | None) -> str:
    value = "cnn" if architecture is None else str(architecture).strip().lower()
    aliases = {
        "cnn": "cnn",
        "eegclassifier20cnn": "cnn",
        "eegnet": "eegnet",
        "eegnetclassifier": "eegnet",
    }
    if value not in aliases:
        raise ValueError(
            "model_architecture must be one of "
            f"{sorted(SUPPORTED_CLASSIFIER_ARCHITECTURES)}, got: {architecture}"
        )
    return aliases[value]


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


class SamePadTemporalConv2d(nn.Module):
    """Conv2d over the temporal axis with explicit same padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_length: int,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if kernel_length <= 0:
            raise ValueError("kernel_length must be positive.")
        left_pad = (int(kernel_length) - 1) // 2
        right_pad = int(kernel_length) // 2
        self.pad = nn.ZeroPad2d((left_pad, right_pad, 0, 0))
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, int(kernel_length)),
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


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


class EEGNetClassifier(nn.Module):
    """Compact EEGNet-style classifier for EEG trials shaped [B, C, T]."""

    def __init__(
        self,
        eeg_channels: int,
        eeg_timesteps: int,
        num_classes: int = 20,
        f1: int = 8,
        d: int = 2,
        f2: Optional[int] = None,
        kernel_length: int = 63,
        separable_kernel_length: int = 15,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if eeg_channels <= 0 or eeg_timesteps <= 0:
            raise ValueError("eeg_channels and eeg_timesteps must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if f1 <= 0 or d <= 0:
            raise ValueError("f1 and d must be positive.")
        if f2 is not None and f2 <= 0:
            raise ValueError("f2 must be positive when provided.")
        if kernel_length <= 0 or separable_kernel_length <= 0:
            raise ValueError("EEGNet kernel lengths must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.eeg_channels = int(eeg_channels)
        self.eeg_timesteps = int(eeg_timesteps)
        self.num_classes = int(num_classes)
        self.f1 = int(f1)
        self.d = int(d)
        self.f2 = int(f2) if f2 is not None else int(f1) * int(d)
        self.kernel_length = int(kernel_length)
        self.separable_kernel_length = int(separable_kernel_length)
        self.dropout = float(dropout)

        self.block1 = nn.Sequential(
            SamePadTemporalConv2d(
                1,
                self.f1,
                kernel_length=self.kernel_length,
                bias=False,
            ),
            nn.BatchNorm2d(self.f1),
            nn.Conv2d(
                self.f1,
                self.f1 * self.d,
                kernel_size=(self.eeg_channels, 1),
                groups=self.f1,
                bias=False,
            ),
            nn.BatchNorm2d(self.f1 * self.d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=self.dropout),
        )
        self.block2 = nn.Sequential(
            SamePadTemporalConv2d(
                self.f1 * self.d,
                self.f1 * self.d,
                kernel_length=self.separable_kernel_length,
                groups=self.f1 * self.d,
                bias=False,
            ),
            nn.Conv2d(self.f1 * self.d, self.f2, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=self.dropout),
        )
        flat_dim = self._infer_flat_dim()
        self.classifier = nn.Linear(flat_dim, self.num_classes)

    @property
    def dense_l1_parameters(self):
        return (self.classifier.weight,)

    def _infer_flat_dim(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.eeg_channels, self.eeg_timesteps)
            out = self.block2(self.block1(dummy))
        return int(out.flatten(start_dim=1).shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.eeg_channels:
            raise ValueError(
                f"Expected EEG channels C={self.eeg_channels}, got C={int(x.shape[1])}."
            )
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_eeg_classifier_model(
    architecture: str,
    eeg_channels: int,
    eeg_timesteps: int,
    num_classes: int = 20,
    cnn_hidden_dim: int = 128,
    eegnet_f1: int = 8,
    eegnet_d: int = 2,
    eegnet_f2: Optional[int] = None,
    eegnet_kernel_length: int = 63,
    eegnet_separable_kernel_length: int = 15,
    eegnet_dropout: float = 0.25,
) -> nn.Module:
    resolved_architecture = resolve_classifier_architecture_name(architecture)
    if resolved_architecture == "cnn":
        return EEGClassifier20CNN(
            eeg_channels=eeg_channels,
            eeg_timesteps=eeg_timesteps,
            num_classes=num_classes,
            hidden_dim=cnn_hidden_dim,
        )
    return EEGNetClassifier(
        eeg_channels=eeg_channels,
        eeg_timesteps=eeg_timesteps,
        num_classes=num_classes,
        f1=eegnet_f1,
        d=eegnet_d,
        f2=eegnet_f2,
        kernel_length=eegnet_kernel_length,
        separable_kernel_length=eegnet_separable_kernel_length,
        dropout=eegnet_dropout,
    )


def extract_eeg_classifier20_arch_metadata(model: nn.Module) -> dict[str, Any]:
    if isinstance(model, EEGClassifier20CNN):
        return {
            "architecture": "cnn",
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
    if isinstance(model, EEGNetClassifier):
        return {
            "architecture": "eegnet",
            "frontend": "eegnet",
            "input_channels": int(model.eeg_channels),
            "input_timesteps": int(model.eeg_timesteps),
            "f1": int(model.f1),
            "d": int(model.d),
            "f2": int(model.f2),
            "kernel_length": int(model.kernel_length),
            "separable_kernel_length": int(model.separable_kernel_length),
            "dropout": float(model.dropout),
            "num_classes": int(model.num_classes),
        }
    raise TypeError(f"Unsupported classifier model type: {type(model).__name__}")
