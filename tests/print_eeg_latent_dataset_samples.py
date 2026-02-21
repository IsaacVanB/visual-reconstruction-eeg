import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentDataset, build_eeg_transform  # noqa: E402


def _parse_class_indices(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    dataset_root = os.environ.get("EEG_DATASET_ROOT", str(repo_root / "datasets"))
    latent_root = os.environ.get("EEG_LATENT_ROOT", str(repo_root / "latents" / "img"))
    subject = os.environ.get("EEG_SUBJECT", "sub-1")
    split = os.environ.get("EEG_SPLIT", "train")
    split_seed = int(os.environ.get("EEG_SPLIT_SEED", "0"))
    num_samples = int(os.environ.get("EEG_NUM_SAMPLES", "4"))
    class_indices = _parse_class_indices(os.environ.get("EEG_CLASS_INDICES", ""))

    eeg_tf = build_eeg_transform(normalize_per_sample=True, to_tensor=True)

    print(f"Dataset root: {dataset_root}")
    print(f"Latent root: {latent_root}")
    print(f"Subject: {subject}")
    print(f"Split: {split}")
    print(f"Split seed: {split_seed}")
    print(f"Class indices: {class_indices}")
    print(f"Num samples: {num_samples}")

    dataset = EEGImageLatentDataset(
        dataset_root=dataset_root,
        latent_root=latent_root,
        subject=subject,
        split=split,
        class_indices=class_indices,
        transform=eeg_tf,
        split_seed=split_seed,
    )

    expected_len = (
        len(dataset.class_indices)
        * dataset._split_counts[split]
        * dataset.repetitions
    )
    print(f"Dataset length: {len(dataset)} (expected {expected_len})")
    print(f"Num classes (all): {dataset.num_classes}")
    print(f"Num classes (selected): {len(dataset.class_indices)}")

    if len(dataset) != expected_len:
        raise RuntimeError(
            f"Unexpected dataset length: got {len(dataset)}, expected {expected_len}"
        )

    for i in range(min(num_samples, len(dataset))):
        eeg, latent, label = dataset[i]
        print(
            f"[{i}] eeg shape={tuple(eeg.shape)} dtype={eeg.dtype} "
            f"latent shape={tuple(latent.shape)} dtype={latent.dtype} "
            f"label={label}"
        )


if __name__ == "__main__":
    main()
