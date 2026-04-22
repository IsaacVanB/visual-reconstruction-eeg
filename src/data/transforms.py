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


def resolve_eeg_time_window(
    times,
    pre_ms: Optional[float] = None,
    post_ms: Optional[float] = None,
) -> Optional[dict]:
    if pre_ms is None and post_ms is None:
        return None
    if pre_ms is None or post_ms is None:
        raise ValueError("Both pre_ms and post_ms must be provided when cropping EEG windows.")

    times_np = np.asarray(times, dtype=np.float32).reshape(-1)
    if times_np.ndim != 1 or times_np.size == 0:
        raise ValueError("Expected a non-empty 1D `times` array.")
    if pre_ms < 0 or post_ms < 0:
        raise ValueError("pre_ms and post_ms must be non-negative.")

    start_s = -float(pre_ms) / 1000.0
    end_s = float(post_ms) / 1000.0
    tolerance = 1e-6
    keep_indices = np.where(
        (times_np >= start_s - tolerance) & (times_np <= end_s + tolerance)
    )[0]
    if keep_indices.size == 0:
        raise ValueError(
            f"No EEG timepoints found in requested window [{start_s:.6f}, {end_s:.6f}] seconds."
        )

    start_idx = int(keep_indices[0])
    end_idx = int(keep_indices[-1])
    selected_times = times_np[start_idx : end_idx + 1]
    return {
        "pre_ms": float(pre_ms),
        "post_ms": float(post_ms),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "requested_start_s": float(start_s),
        "requested_end_s": float(end_s),
        "actual_start_s": float(selected_times[0]),
        "actual_end_s": float(selected_times[-1]),
        "num_timepoints": int(selected_times.size),
    }


def crop_eeg_time_window(eeg, start_idx: int, end_idx: int):
    eeg_np = np.asarray(eeg, dtype=np.float32)
    if eeg_np.ndim < 2:
        raise ValueError(f"Expected EEG array with at least 2 dims, got shape {tuple(eeg_np.shape)}")
    time_dim = eeg_np.shape[-1]
    if start_idx < 0 or end_idx < start_idx or end_idx >= time_dim:
        raise IndexError(
            f"Invalid EEG time window [{start_idx}, {end_idx}] for time dimension {time_dim}."
        )
    return eeg_np[..., start_idx : end_idx + 1]


class EEGTimeWindowCrop:
    """Crop EEG sample [C, T] to a configured time window."""

    def __init__(self, start_idx: int, end_idx: int) -> None:
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)

    def __call__(self, eeg):
        eeg_np = np.asarray(eeg, dtype=np.float32)
        if eeg_np.ndim != 2:
            raise ValueError(f"Expected EEG sample [C, T], got shape {tuple(eeg_np.shape)}")
        return crop_eeg_time_window(eeg_np, start_idx=self.start_idx, end_idx=self.end_idx)


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
    crop_start_idx: int | None = None,
    crop_end_idx: int | None = None,
) -> Compose:
    transforms = []
    if crop_start_idx is not None or crop_end_idx is not None:
        if crop_start_idx is None or crop_end_idx is None:
            raise ValueError("Both crop_start_idx and crop_end_idx must be set together.")
        transforms.append(EEGTimeWindowCrop(start_idx=crop_start_idx, end_idx=crop_end_idx))

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
