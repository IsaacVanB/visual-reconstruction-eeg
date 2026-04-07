from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch


class Compose:
    def __init__(self, transforms: Iterable) -> None:
        self.transforms = list(transforms)

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class EEGPerSampleNormalize:
    """L2-normalize a single EEG sample shaped [channels, time]."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    def __call__(self, eeg):
        eeg_np = np.asarray(eeg, dtype=np.float32)
        norm = np.linalg.norm(eeg_np)
        if norm < self.eps:
            return eeg_np
        return eeg_np / norm


class EEGChannelZScoreNormalize:
    """Channel-wise z-score normalize EEG sample [C, T] using train-set stats."""

    def __init__(self, mean, std, eps: float = 1e-6) -> None:
        mean_np = np.asarray(mean, dtype=np.float32).reshape(-1, 1)
        std_np = np.asarray(std, dtype=np.float32).reshape(-1, 1)
        if mean_np.shape != std_np.shape:
            raise ValueError(
                f"zscore mean/std shape mismatch: {mean_np.shape} vs {std_np.shape}"
            )
        self.mean = mean_np
        self.std = np.clip(std_np, eps, None)
        self.eps = float(eps)

    def __call__(self, eeg):
        eeg_np = np.asarray(eeg, dtype=np.float32)
        if eeg_np.ndim != 2:
            raise ValueError(f"Expected EEG sample [C, T], got shape {tuple(eeg_np.shape)}")
        if eeg_np.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"EEG channels ({eeg_np.shape[0]}) do not match zscore stats "
                f"({self.mean.shape[0]})."
            )
        return (eeg_np - self.mean) / self.std


class EEGToTensor:
    def __call__(self, eeg):
        if isinstance(eeg, torch.Tensor):
            return eeg.to(dtype=torch.float32)
        return torch.from_numpy(np.asarray(eeg, dtype=np.float32))


class ResizeImage:
    def __init__(self, size: Tuple[int, int], resample=Image.BICUBIC) -> None:
        self.size = size
        self.resample = resample

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize(self.size, resample=self.resample)


class ImageToTensor:
    """Convert PIL image to float32 tensor in [0, 1], shape [C, H, W]."""

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            return image.to(dtype=torch.float32)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(image_np).permute(2, 0, 1).contiguous()


class NormalizeImage:
    def __init__(self, mean: Sequence[float], std: Sequence[float], eps: float = 1e-8) -> None:
        mean_tensor = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        std_tensor = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.mean = mean_tensor
        self.std = std_tensor.clamp_min(eps)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.mean) / self.std


def build_image_transform(
    image_size: Tuple[int, int] = (256, 256),
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Compose:
    transforms = [ResizeImage(image_size), ImageToTensor()]
    if mean is not None and std is not None:
        transforms.append(NormalizeImage(mean=mean, std=std))
    return Compose(transforms)


def build_eeg_transform(
    normalize_per_sample: bool = True,
    to_tensor: bool = True,
    normalize_mode: str | None = None,
    zscore_mean=None,
    zscore_std=None,
    zscore_eps: float = 1e-6,
) -> Compose:
    transforms = []
    if normalize_mode is None:
        normalize_mode = "l2" if normalize_per_sample else "none"
    normalize_mode = str(normalize_mode).lower()

    if normalize_mode == "l2":
        transforms.append(EEGPerSampleNormalize())
    elif normalize_mode == "zscore":
        if zscore_mean is None or zscore_std is None:
            raise ValueError("zscore normalization requires zscore_mean and zscore_std.")
        transforms.append(
            EEGChannelZScoreNormalize(
                mean=zscore_mean,
                std=zscore_std,
                eps=zscore_eps,
            )
        )
    elif normalize_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown EEG normalization mode: {normalize_mode}")

    if to_tensor:
        transforms.append(EEGToTensor())
    return Compose(transforms)

"""
EXAMPLE USAGE

from src.data import build_eeg_dataloader, build_eeg_transform, build_image_transform

eeg_tf = build_eeg_transform(normalize_per_sample=True, to_tensor=True)
img_tf = build_image_transform(
    image_size=(256, 256),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)

loader = build_eeg_dataloader(
    dataset_root="datasets",
    split="train",
    transform=eeg_tf,
    image_transform=img_tf,
    batch_size=32,
)

"""
