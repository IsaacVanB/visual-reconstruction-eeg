import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.dataloader import build_eeg_dataloader  # noqa: E402


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
    split = os.environ.get("EEG_SPLIT", "train")
    split_seed = int(os.environ.get("EEG_SPLIT_SEED", "0"))
    split_img_idx = int(os.environ.get("EEG_SPLIT_IMG_IDX", "0"))
    sample_rep = int(os.environ.get("EEG_SAMPLE_REP", "0"))
    save_fig = os.environ.get("EEG_SAVE_FIG", "").strip()

    loader = build_eeg_dataloader(
        dataset_root=dataset_root,
        subject=subject,
        split=split,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        split_seed=split_seed,
        return_image_name=True,
    )
    dataset = loader.dataset

    if split_img_idx < 0 or split_img_idx >= len(dataset._split_image_indices):
        raise IndexError(
            f"EEG_SPLIT_IMG_IDX out of range: {split_img_idx} not in [0, {len(dataset._split_image_indices) - 1}]"
        )
    if sample_rep < 0 or sample_rep >= dataset.repetitions:
        raise IndexError(
            f"EEG_SAMPLE_REP out of range: {sample_rep} not in [0, {dataset.repetitions - 1}]"
        )

    train_img_idx = int(dataset._split_image_indices[split_img_idx])
    eeg_data_single_image = dataset.eeg[train_img_idx]
    image_name = dataset.train_img_files[train_img_idx]
    image_concept = image_name.rsplit("_", 1)[0]

    sample_idx = split_img_idx * dataset.repetitions + sample_rep
    eeg_rep_from_dataset, image, label, image_name_from_dataset = dataset[sample_idx]

    np.set_printoptions(threshold=np.inf)

    print(f"Dataset root: {dataset_root}")
    print(f"Subject: {subject}")
    print(f"Split: {split}")
    print(f"Split seed: {split_seed}")
    print(f"Split image index (within {split} split): {split_img_idx}")
    print(f"Resolved train_img_idx (global metadata index): {train_img_idx}")
    print(f"Selected repetition index: {sample_rep}")
    print(f"Label: {label}")
    print()

    print("Training EEG single image data shape:")
    print(eeg_data_single_image.shape)
    print(eeg_data_single_image[0])
    print("(Training EEG repetitions × EEG channels × EEG time points)\n")

    rep_matches = np.array_equal(eeg_rep_from_dataset, eeg_data_single_image[sample_rep])
    print(f"Dataset sample image file: {image_name_from_dataset}")
    print(f"Repetition EEG matches full EEG block at rep={sample_rep}: {rep_matches}")

    image_path = _resolve_image_path(dataset.image_root, image_name)
    print(f"Resolved image path: {image_path}")

    plt.figure(figsize=(6, 6))
    plt.axis("off")
    plt.imshow(image)
    plt.title(
        "Training image (split idx): "
        f"{split_img_idx + 1} | global idx: {train_img_idx + 1}\n"
        f"Image concept: {image_concept}\n"
        f"Image file: {image_name}"
    )

    backend = matplotlib.get_backend().lower()
    is_non_interactive = backend.endswith("agg")

    if save_fig:
        plt.savefig(save_fig, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_fig}")
    elif is_non_interactive:
        fallback_path = repo_root / "outputs" / "inspect_eeg_image_pair.png"
        plt.savefig(fallback_path, dpi=150, bbox_inches="tight")
        print(
            f"Matplotlib backend '{matplotlib.get_backend()}' is non-interactive. "
            f"Saved figure to: {fallback_path}"
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
