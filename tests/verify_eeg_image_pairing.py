import os
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.datasets import EEGImageDataset  # noqa: E402


def _resolve_image_path(image_root: str, image_name: str) -> str:
    if "/" in image_name or os.path.sep in image_name:
        rel_path = image_name
    else:
        class_name = image_name.rsplit("_", 1)[0]
        rel_path = os.path.join(class_name, image_name)
    return os.path.join(image_root, rel_path)


def main() -> None:
    dataset_root = os.environ.get("EEG_DATASET_ROOT", str(repo_root / "datasets"))
    subject = os.environ.get("EEG_SUBJECT", "sub-1")
    split_seed = int(os.environ.get("EEG_SPLIT_SEED", "0"))
    strict_image_load = os.environ.get("EEG_STRICT_IMAGE_LOAD", "0") == "1"

    print(f"Dataset root: {dataset_root}")
    print(f"Subject: {subject}")
    print(f"Split seed: {split_seed}")
    print(f"Strict image load: {strict_image_load}")

    ds_train = EEGImageDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="train",
        split_seed=split_seed,
        return_image_name=True,
    )
    ds_valid = EEGImageDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="valid",
        split_seed=split_seed,
        return_image_name=True,
    )
    ds_test = EEGImageDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="test",
        split_seed=split_seed,
        return_image_name=True,
    )

    num_images = ds_train.num_images
    repetitions = ds_train.repetitions
    images_per_class = ds_train.images_per_class

    # Core invariant: number of metadata images must match EEG first dimension.
    assert num_images == len(ds_train.train_img_files), (
        f"Metadata/image count mismatch: {num_images} vs {len(ds_train.train_img_files)}"
    )
    assert ds_train.eeg.shape[0] == num_images, (
        f"EEG/image count mismatch: {ds_train.eeg.shape[0]} vs {num_images}"
    )

    # Core invariant: split partition should cover all global image indices exactly once.
    split_union = np.concatenate(
        [ds_train._split_image_indices, ds_valid._split_image_indices, ds_test._split_image_indices]
    )
    assert split_union.size == num_images, (
        f"Split union size mismatch: {split_union.size} vs {num_images}"
    )
    uniq, counts = np.unique(split_union, return_counts=True)
    assert uniq.size == num_images, "Some global image indices are missing from split partition."
    assert np.all(counts == 1), "Some global image indices appear in multiple splits."
    assert np.array_equal(uniq, np.arange(num_images)), "Split partition is not exactly [0..num_images-1]."

    # Path/name invariant for every global image index.
    for global_idx, image_name in enumerate(ds_train.train_img_files):
        resolved_path = _resolve_image_path(ds_train.image_root, image_name)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Missing image file for global_idx={global_idx}: {resolved_path}"
            )

    # Per-split per-image checks (rep=0): returned name, label, and EEG all match global index.
    split_sets = {
        "train": ds_train,
        "valid": ds_valid,
        "test": ds_test,
    }
    for split_name, ds in split_sets.items():
        for split_pos, global_idx in enumerate(ds._split_image_indices):
            sample_idx = split_pos * repetitions
            eeg_rep, image, label, image_name = ds[sample_idx]

            expected_name = ds.train_img_files[int(global_idx)]
            expected_label = int(global_idx) // images_per_class
            expected_eeg = ds.eeg[int(global_idx), 0]

            if image_name != expected_name:
                raise AssertionError(
                    f"Name mismatch ({split_name}, split_pos={split_pos}, global_idx={global_idx}): "
                    f"got {image_name}, expected {expected_name}"
                )
            if int(label) != expected_label:
                raise AssertionError(
                    f"Label mismatch ({split_name}, split_pos={split_pos}, global_idx={global_idx}): "
                    f"got {label}, expected {expected_label}"
                )
            if not np.array_equal(eeg_rep, expected_eeg):
                raise AssertionError(
                    f"EEG mismatch ({split_name}, split_pos={split_pos}, global_idx={global_idx}, rep=0)."
                )

            # Optional strict check: ensure image object is actually loaded and RGB.
            if strict_image_load:
                if getattr(image, "mode", None) != "RGB":
                    raise AssertionError(
                        f"Image mode mismatch ({split_name}, split_pos={split_pos}, global_idx={global_idx}): "
                        f"{getattr(image, 'mode', None)}"
                    )

    # Repetition invariant for every sample in every split without reloading images repeatedly.
    for split_name, ds in split_sets.items():
        for sample_idx, (global_idx, rep_idx) in enumerate(ds._sample_index):
            expected_label = int(global_idx) // images_per_class
            if expected_label != int(global_idx) // images_per_class:
                raise AssertionError(f"Label formula mismatch at {split_name} sample_idx={sample_idx}")
            if rep_idx < 0 or rep_idx >= repetitions:
                raise AssertionError(
                    f"Invalid repetition index at {split_name} sample_idx={sample_idx}: {rep_idx}"
                )

    print("All pairing checks passed.")
    print("Verified invariants:")
    print("- EEG count matches metadata image count")
    print("- train/valid/test split indices partition global image indices exactly once")
    print("- Every metadata image path exists on disk")
    print("- For every split image (rep=0), __getitem__ returns correct EEG slice, label, and image_name")
    print("- Sample index tuples use valid repetition ranges")


if __name__ == "__main__":
    main()
