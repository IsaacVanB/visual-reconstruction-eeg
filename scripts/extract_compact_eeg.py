import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


IMAGES_PER_CLASS = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract compact THINGS EEG files for selected zero-based class indices. "
            "Writes one preprocessed_eeg_training.npy dict per subject."
        )
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets",
        help="Root containing THINGS_EEG_2.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help=(
            "Output directory for compact files. Subject files are written to "
            "<output-root>/<subject>/preprocessed_eeg_training.npy."
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=["all"],
        help="Subject ids like sub-1 sub-2, or 'all' for every available subject.",
    )
    parser.add_argument(
        "--class-indices",
        type=int,
        nargs="+",
        required=True,
        help=(
            "Zero-based class indices to keep, e.g. 9 525 59. "
            "Each class contributes 10 original stimulus images."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing compact subject files.",
    )
    return parser.parse_args()


def _load_npy_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected {path} to contain a dict, got {type(obj)}")
    return obj


def _discover_subjects(things_root: Path) -> list[str]:
    def subject_sort_key(path: Path) -> tuple[int, str]:
        suffix = path.name.removeprefix("sub-")
        return (int(suffix), path.name) if suffix.isdigit() else (10**9, path.name)

    subjects = []
    for subject_dir in sorted(things_root.glob("sub-*"), key=subject_sort_key):
        eeg_path = subject_dir / "preprocessed_eeg_training.npy"
        if subject_dir.is_dir() and eeg_path.exists():
            subjects.append(subject_dir.name)
    if not subjects:
        raise FileNotFoundError(
            f"No subject files found under {things_root}; expected sub-*/preprocessed_eeg_training.npy."
        )
    return subjects


def _resolve_subjects(things_root: Path, requested: Sequence[str]) -> list[str]:
    requested = [str(subject) for subject in requested]
    if len(requested) == 1 and requested[0].lower() == "all":
        return _discover_subjects(things_root)
    if any(subject.lower() == "all" for subject in requested):
        raise ValueError("Use --subjects all by itself, not mixed with explicit subjects.")
    if len(set(requested)) != len(requested):
        raise ValueError("--subjects contains duplicates.")
    for subject in requested:
        eeg_path = things_root / subject / "preprocessed_eeg_training.npy"
        if not eeg_path.exists():
            raise FileNotFoundError(f"EEG file not found for {subject}: {eeg_path}")
    return requested


def _resolve_image_indices(class_indices: Sequence[int], num_images: int) -> np.ndarray:
    if num_images <= 0 or num_images % IMAGES_PER_CLASS != 0:
        raise ValueError(
            f"Expected num_images to be divisible by {IMAGES_PER_CLASS}, got {num_images}."
        )
    num_classes = num_images // IMAGES_PER_CLASS
    classes = np.asarray([int(x) for x in class_indices], dtype=np.int64)
    if classes.ndim != 1 or classes.size == 0:
        raise ValueError("--class-indices must be a non-empty list.")
    if np.unique(classes).size != classes.size:
        raise ValueError("--class-indices contains duplicates.")
    if np.any(classes < 0) or np.any(classes >= num_classes):
        raise ValueError(f"--class-indices must be in [0, {num_classes - 1}].")

    image_indices = []
    for class_idx in classes.tolist():
        start = int(class_idx) * IMAGES_PER_CLASS
        image_indices.extend(range(start, start + IMAGES_PER_CLASS))
    return np.asarray(image_indices, dtype=np.int64)


def _load_image_metadata(things_root: Path) -> dict[str, Any] | None:
    metadata_path = things_root / "image_metadata.npy"
    if not metadata_path.exists():
        return None
    return _load_npy_dict(metadata_path)


def _get_train_img_files(image_metadata: dict[str, Any] | None) -> list[str] | None:
    if image_metadata is None or image_metadata.get("train_img_files", None) is None:
        return None
    return [str(x) for x in np.asarray(image_metadata["train_img_files"]).tolist()]


def _write_compact_image_metadata(
    image_metadata: dict[str, Any] | None,
    output_root: Path,
    image_indices: np.ndarray,
    overwrite: bool,
) -> str | None:
    if image_metadata is None or image_metadata.get("train_img_files", None) is None:
        return None

    metadata_path = output_root / "image_metadata.npy"
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {metadata_path}. Pass --overwrite to replace it.")

    compact_metadata = dict(image_metadata)
    train_img_files = np.asarray(image_metadata["train_img_files"])
    compact_metadata["train_img_files"] = train_img_files[image_indices]
    compact_metadata["original_image_indices"] = image_indices
    np.save(metadata_path, compact_metadata, allow_pickle=True)
    return str(metadata_path)


def _extract_subject(
    things_root: Path,
    output_root: Path,
    subject: str,
    class_indices: Sequence[int],
    train_img_files: list[str] | None,
    overwrite: bool,
) -> dict[str, Any]:
    input_path = things_root / subject / "preprocessed_eeg_training.npy"
    eeg_bundle = _load_npy_dict(input_path)
    if "preprocessed_eeg_data" not in eeg_bundle:
        raise KeyError(f"Expected key 'preprocessed_eeg_data' in {input_path}")

    eeg = np.asarray(eeg_bundle["preprocessed_eeg_data"])
    if eeg.ndim != 4:
        raise ValueError(
            f"Expected EEG shape [images, repeats, channels, time] for {subject}, got {eeg.shape}"
        )

    image_indices = _resolve_image_indices(class_indices=class_indices, num_images=int(eeg.shape[0]))
    compact_eeg = np.asarray(eeg[image_indices], dtype=np.float32)
    original_labels = (image_indices // IMAGES_PER_CLASS).astype(np.int64)
    compact_labels = np.repeat(
        np.arange(len(class_indices), dtype=np.int64), IMAGES_PER_CLASS
    )

    payload: dict[str, Any] = {
        "preprocessed_eeg_data": compact_eeg,
        "original_image_indices": image_indices,
        "original_class_indices": np.asarray(class_indices, dtype=np.int64),
        "labels": compact_labels,
        "original_labels": original_labels,
        "images_per_class": IMAGES_PER_CLASS,
        "source_subject": subject,
        "source_path": str(input_path),
    }
    if "times" in eeg_bundle:
        payload["times"] = np.asarray(eeg_bundle["times"])
    if "ch_names" in eeg_bundle:
        payload["ch_names"] = np.asarray(eeg_bundle["ch_names"])
    if train_img_files is not None:
        payload["train_img_files"] = np.asarray([train_img_files[int(idx)] for idx in image_indices])

    subject_dir = output_root / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    output_path = subject_dir / "preprocessed_eeg_training.npy"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")
    np.save(output_path, payload, allow_pickle=True)

    return {
        "subject": subject,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "shape": list(compact_eeg.shape),
        "dtype": str(compact_eeg.dtype),
        "num_original_images": int(eeg.shape[0]),
        "num_compact_images": int(compact_eeg.shape[0]),
        "class_indices": [int(x) for x in class_indices],
    }


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    things_root = dataset_root / "THINGS_EEG_2"
    if not things_root.exists():
        raise FileNotFoundError(f"THINGS_EEG_2 not found under dataset root: {dataset_root}")

    class_indices = [int(x) for x in args.class_indices]
    subjects = _resolve_subjects(things_root=things_root, requested=args.subjects)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_metadata = _load_image_metadata(things_root)
    train_img_files = _get_train_img_files(image_metadata)
    first_subject_eeg = _load_npy_dict(
        things_root / subjects[0] / "preprocessed_eeg_training.npy"
    )["preprocessed_eeg_data"]
    shared_image_indices = _resolve_image_indices(
        class_indices=class_indices,
        num_images=int(np.asarray(first_subject_eeg).shape[0]),
    )
    metadata_output_path = _write_compact_image_metadata(
        image_metadata=image_metadata,
        output_root=output_root,
        image_indices=shared_image_indices,
        overwrite=bool(args.overwrite),
    )
    if metadata_output_path is not None:
        print(f"Saved compact image metadata: {metadata_output_path}")

    summaries = []
    for subject in subjects:
        summary = _extract_subject(
            things_root=things_root,
            output_root=output_root,
            subject=subject,
            class_indices=class_indices,
            train_img_files=train_img_files,
            overwrite=bool(args.overwrite),
        )
        summaries.append(summary)
        print(
            f"{subject}: saved {summary['shape']} {summary['dtype']} -> "
            f"{summary['output_path']}"
        )

    manifest = {
        "dataset_root": str(dataset_root),
        "source_things_root": str(things_root),
        "output_root": str(output_root),
        "subjects": subjects,
        "class_indices": class_indices,
        "images_per_class": IMAGES_PER_CLASS,
        "image_metadata_path": metadata_output_path,
        "files": summaries,
    }
    manifest_path = output_root / "compact_eeg_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
