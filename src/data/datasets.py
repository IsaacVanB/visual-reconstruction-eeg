import os
from typing import Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# this file is in eeg_project_26/src/data
# data is in eeg_project_26/datasets/THINGS_EEG_2 --> sub-1/preprocessed_eeg_training.npy


def _resolve_class_indices(
    class_indices: Optional[Sequence[int]], num_classes: int
) -> np.ndarray:
    if class_indices is None:
        return np.arange(num_classes, dtype=np.int64)

    resolved = np.asarray(list(class_indices), dtype=np.int64)
    if resolved.ndim != 1 or resolved.size == 0:
        raise ValueError("class_indices must be a non-empty 1D list/sequence of class ids.")
    if np.any(resolved < 0) or np.any(resolved >= num_classes):
        raise ValueError(f"class_indices must be in [0, {num_classes - 1}].")
    if np.unique(resolved).size != resolved.size:
        raise ValueError("class_indices contains duplicates.")
    return resolved


class EEGImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        subject: str = "sub-1",
        split: str = "train",
        class_indices: Optional[Sequence[int]] = None,
        transform=None,
        target_transform=None,
        mmap_mode: Optional[str] = "r",
        image_transform=None,
        return_image_name: bool = False,
        split_seed: int = 0,
    ) -> None:
        self.dataset_root = dataset_root
        self.subject = subject
        self.split = split
        self.class_indices = class_indices
        self.transform = transform
        self.target_transform = target_transform
        self.mmap_mode = mmap_mode
        self.image_transform = image_transform
        self.return_image_name = return_image_name
        self.split_seed = split_seed

        eeg_path = os.path.join(
            self.dataset_root, "THINGS_EEG_2", self.subject, "preprocessed_eeg_training.npy"
        )
        if not os.path.exists(eeg_path):
            raise FileNotFoundError(f"EEG file not found: {eeg_path}")

        self.eeg = None
        if self.mmap_mode:
            try:
                self.eeg = np.load(eeg_path, mmap_mode=self.mmap_mode)
            except ValueError:
                self.eeg = None

        if self.eeg is None:
            self.eeg = np.load(eeg_path, allow_pickle=True)
            if isinstance(self.eeg, np.ndarray) and self.eeg.dtype == object:
                if self.eeg.ndim == 0:
                    self.eeg = self.eeg.item()
                else:
                    try:
                        self.eeg = np.stack(self.eeg)
                    except ValueError as exc:
                        raise ValueError(
                            "EEG array has object dtype and could not be stacked into a dense array."
                        ) from exc

        if isinstance(self.eeg, dict):
            if "preprocessed_eeg_data" not in self.eeg:
                raise KeyError("Expected key 'preprocessed_eeg_data' in EEG dict.")
            self.eeg = self.eeg["preprocessed_eeg_data"]

        img_metadata_path = os.path.join(self.dataset_root, "THINGS_EEG_2", "image_metadata.npy")
        if not os.path.exists(img_metadata_path):
            raise FileNotFoundError(f"Image metadata not found: {img_metadata_path}")
        img_metadata = np.load(img_metadata_path, allow_pickle=True).item()
        if "train_img_files" not in img_metadata:
            raise KeyError("Expected key 'train_img_files' in image metadata.")
        self.train_img_files = img_metadata["train_img_files"]

        self.image_root = os.path.join(self.dataset_root, "images_THINGS", "object_images")
        if self.eeg.ndim != 4 or self.eeg.shape[1:] != (4, 17, 100):
            raise ValueError("Expected EEG shape (16540, 4, 17, 100); " f"got {self.eeg.shape}")

        self.num_images = self.eeg.shape[0]
        self.repetitions = self.eeg.shape[1]
        self.images_per_class = 10
        if self.num_images % self.images_per_class != 0:
            raise ValueError(
                "Number of images must be divisible by images_per_class; "
                f"got {self.num_images} and {self.images_per_class}."
            )
        self.num_classes = self.num_images // self.images_per_class
        self.class_indices = _resolve_class_indices(self.class_indices, self.num_classes)

        self._split_counts = {"train": 7, "valid": 2, "test": 1} # if changing, also change in ImageDataset (~line 289)
        if self.split not in self._split_counts:
            raise ValueError("split must be one of: 'train', 'valid', 'test'.")

        self._split_image_indices = self._build_split_image_indices()
        self._sample_index = [
            (image_idx, rep_idx)
            for image_idx in self._split_image_indices
            for rep_idx in range(self.repetitions)
        ]

    def _build_split_image_indices(self):
        split_offsets = {
            "train": (0, self._split_counts["train"]),
            "valid": (
                self._split_counts["train"],
                self._split_counts["train"] + self._split_counts["valid"],
            ),
            "test": (
                self._split_counts["train"] + self._split_counts["valid"],
                self.images_per_class,
            ),
        }
        start, end = split_offsets[self.split]

        rng = np.random.default_rng(self.split_seed)
        split_image_indices = []
        for class_idx in self.class_indices:
            class_start = class_idx * self.images_per_class
            class_images = np.arange(class_start, class_start + self.images_per_class)
            shuffled = rng.permutation(class_images)
            split_image_indices.extend(shuffled[start:end].tolist())

        return np.array(split_image_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        image_index, rep_index = self._sample_index[idx]
        eeg_sample = self.eeg[image_index, rep_index]
        label = image_index // self.images_per_class

        if self.transform:
            eeg_sample = self.transform(eeg_sample)
        if self.target_transform:
            label = self.target_transform(label)

        image_name = self.train_img_files[image_index]
        if "/" in image_name or os.path.sep in image_name:
            rel_path = image_name
        else:
            class_name = image_name.rsplit("_", 1)[0]
            rel_path = os.path.join(class_name, image_name)

        image_path = os.path.join(self.image_root, rel_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        if self.return_image_name:
            return eeg_sample, image, label, image_name

        return eeg_sample, image, label


class EEGImageLatentDataset(EEGImageDataset):
    def __init__(
        self,
        dataset_root: str,
        subject: str = "sub-1",
        split: str = "train",
        class_indices: Optional[Sequence[int]] = None,
        transform=None,
        target_transform=None,
        mmap_mode: Optional[str] = "r",
        latent_root: str = os.path.join("latents", "img"),
        latent_transform=None,
        split_seed: int = 0,
    ) -> None:
        super().__init__(
            dataset_root=dataset_root,
            subject=subject,
            split=split,
            class_indices=class_indices,
            transform=transform,
            target_transform=target_transform,
            mmap_mode=mmap_mode,
            image_transform=None,
            return_image_name=False,
            split_seed=split_seed,
        )
        self.latent_root = latent_root
        self.latent_transform = latent_transform

    def _resolve_latent_path(self, image_index: int) -> str:
        latent_path = os.path.join(self.latent_root, f"{image_index}.pt")
        if os.path.exists(latent_path):
            return latent_path

        padded_latent_path = os.path.join(self.latent_root, f"{image_index:06d}.pt")
        if os.path.exists(padded_latent_path):
            return padded_latent_path

        raise FileNotFoundError(
            "Latent file not found for image index "
            f"{image_index}. Tried: {latent_path} and {padded_latent_path}"
        )

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        image_index, rep_index = self._sample_index[idx]
        eeg_sample = self.eeg[image_index, rep_index]
        label = image_index // self.images_per_class

        if self.transform:
            eeg_sample = self.transform(eeg_sample)
        if self.target_transform:
            label = self.target_transform(label)

        latent_path = self._resolve_latent_path(int(image_index))
        try:
            image_latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        except TypeError:
            image_latent = torch.load(latent_path, map_location="cpu")
        if self.latent_transform:
            image_latent = self.latent_transform(image_latent)

        return eeg_sample, image_latent, label


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        split: str = "train",
        class_indices: Optional[Sequence[int]] = None,
        image_transform=None,
        target_transform=None,
        return_image_name: bool = False,
        return_image_id: bool = False,
        split_seed: int = 0,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.class_indices = class_indices
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.return_image_name = return_image_name
        self.return_image_id = return_image_id
        self.split_seed = split_seed

        img_metadata_path = os.path.join(self.dataset_root, "THINGS_EEG_2", "image_metadata.npy")
        if not os.path.exists(img_metadata_path):
            raise FileNotFoundError(f"Image metadata not found: {img_metadata_path}")
        img_metadata = np.load(img_metadata_path, allow_pickle=True).item()
        if "train_img_files" not in img_metadata:
            raise KeyError("Expected key 'train_img_files' in image metadata.")
        self.train_img_files = img_metadata["train_img_files"]

        self.image_root = os.path.join(self.dataset_root, "images_THINGS", "object_images")
        self.num_images = len(self.train_img_files)
        self.images_per_class = 10
        if self.num_images % self.images_per_class != 0:
            raise ValueError(
                "Number of images must be divisible by images_per_class; "
                f"got {self.num_images} and {self.images_per_class}."
            )
        self.num_classes = self.num_images // self.images_per_class
        self.class_indices = _resolve_class_indices(self.class_indices, self.num_classes)

        self._split_counts = {"train": 7, "valid": 2, "test": 1} # if changing, also change in EEGImageDataset (~line 108)
        if self.split not in self._split_counts:
            raise ValueError("split must be one of: 'train', 'valid', 'test'.")

        self._split_image_indices = self._build_split_image_indices()

    def _build_split_image_indices(self):
        split_offsets = {
            "train": (0, self._split_counts["train"]),
            "valid": (
                self._split_counts["train"],
                self._split_counts["train"] + self._split_counts["valid"],
            ),
            "test": (
                self._split_counts["train"] + self._split_counts["valid"],
                self.images_per_class,
            ),
        }
        start, end = split_offsets[self.split]

        rng = np.random.default_rng(self.split_seed)
        split_image_indices = []
        for class_idx in self.class_indices:
            class_start = class_idx * self.images_per_class
            class_images = np.arange(class_start, class_start + self.images_per_class)
            shuffled = rng.permutation(class_images)
            split_image_indices.extend(shuffled[start:end].tolist())

        return np.array(split_image_indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._split_image_indices)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index out of range: {idx}")

        image_index = int(self._split_image_indices[idx])
        label = image_index // self.images_per_class
        if self.target_transform:
            label = self.target_transform(label)

        image_name = self.train_img_files[image_index]
        if "/" in image_name or os.path.sep in image_name:
            rel_path = image_name
        else:
            class_name = image_name.rsplit("_", 1)[0]
            rel_path = os.path.join(class_name, image_name)

        image_path = os.path.join(self.image_root, rel_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        if self.return_image_name:
            return image, label, image_name
        
        if self.return_image_id:
            return image, label, image_index

        return image, label
