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
) -> Compose:
    transforms = []
    if normalize_per_sample:
        transforms.append(EEGPerSampleNormalize())
    if to_tensor:
        transforms.append(EEGToTensor())
    return Compose(transforms)
