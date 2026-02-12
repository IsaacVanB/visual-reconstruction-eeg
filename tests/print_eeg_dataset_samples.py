import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data.datasets import EEGImageDataset  # noqa: E402


def main() -> None:
    dataset_root = os.environ.get("EEG_DATASET_ROOT", str(repo_root / "datasets"))
    subject = os.environ.get("EEG_SUBJECT", "sub-1")
    split = os.environ.get("EEG_SPLIT", "train")
    split_seed = int(os.environ.get("EEG_SPLIT_SEED", "0"))
    num_samples = int(os.environ.get("EEG_NUM_SAMPLES", "4"))

    print(f"Dataset root: {dataset_root}")
    print(f"Subject: {subject}")
    print(f"Split: {split}")
    print(f"Split seed: {split_seed}")
    print(f"Num samples: {num_samples}")

    dataset = EEGImageDataset(
        dataset_root=dataset_root,
        subject=subject,
        split=split,
        split_seed=split_seed,
    )
    print(f"Dataset length: {len(dataset)}")
    print(f"Num classes: {dataset.num_classes}")

    for i in range(min(num_samples, len(dataset))):
        eeg, image, label = dataset[i]
        print(
            f"[{i}] eeg shape={getattr(eeg, 'shape', None)} "
            f"dtype={getattr(eeg, 'dtype', None)} "
            f"label={label} "
            f"image_size={getattr(image, 'size', None)}"
        )

if __name__ == "__main__":
    main()
