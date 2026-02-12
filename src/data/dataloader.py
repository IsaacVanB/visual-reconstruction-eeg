from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .datasets import EEGImageDataset


def _eeg_image_collate(batch):
    """Batch EEG/labels and stack images when they are tensors."""
    if not batch:
        raise ValueError("Empty batch received.")

    has_image_name = len(batch[0]) == 4
    eeg_items = [item[0] for item in batch]
    if isinstance(eeg_items[0], torch.Tensor):
        eeg = torch.stack(eeg_items, dim=0)
    else:
        eeg = torch.from_numpy(np.stack(eeg_items))
    images = [item[1] for item in batch]
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.long)

    if has_image_name:
        image_names = [item[3] for item in batch]
        return eeg, images, labels, image_names

    return eeg, images, labels


def build_eeg_dataloader(
    dataset_root: str,
    subject: str = "sub-1",
    split: str = "train",
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: Optional[bool] = None,
    transform=None,
    image_transform=None,
    target_transform=None,
    split_seed: int = 0,
    return_image_name: bool = False,
    mmap_mode: Optional[str] = "r",
    persistent_workers: bool = False,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"

    dataset = EEGImageDataset(
        dataset_root=dataset_root,
        subject=subject,
        split=split,
        transform=transform,
        image_transform=image_transform,
        target_transform=target_transform,
        split_seed=split_seed,
        return_image_name=return_image_name,
        mmap_mode=mmap_mode,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_eeg_image_collate,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
