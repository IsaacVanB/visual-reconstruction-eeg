import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import build_eeg_dataloader  # noqa: E402


def _print_split_batch_shapes(split: str, dataset_root: str, subject: str, batch_size: int, split_seed: int) -> None:
    loader = build_eeg_dataloader(
        dataset_root=dataset_root,
        subject=subject,
        split=split,
        batch_size=batch_size,
        num_workers=0,
        split_seed=split_seed,
    )
    eeg, images, labels = next(iter(loader))
    print(
        f"{split}: eeg={tuple(eeg.shape)} labels={tuple(labels.shape)} "
        f"num_images={len(images)}"
    )


def main() -> None:
    dataset_root = os.environ.get("EEG_DATASET_ROOT", str(repo_root / "datasets"))
    subject = os.environ.get("EEG_SUBJECT", "sub-1")
    batch_size = int(os.environ.get("EEG_BATCH_SIZE", "8"))
    split_seed = int(os.environ.get("EEG_SPLIT_SEED", "0"))

    print(f"Dataset root: {dataset_root}")
    print(f"Subject: {subject}")
    print(f"Batch size: {batch_size}")
    print(f"Split seed: {split_seed}")

    for split in ("train", "valid", "test"):
        _print_split_batch_shapes(split, dataset_root, subject, batch_size, split_seed)


if __name__ == "__main__":
    main()
