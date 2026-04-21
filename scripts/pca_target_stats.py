import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import ImageDataset


DEFAULT_CLASS_INDICES = list(range(0, 200, 2))
DEFAULT_CLASS_INDICES_1000 = list(range(0, 2000, 2))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute PCA-target statistics for train/valid splits: "
            "mean vector, variance vector, and average L2 norm."
        )
    )
    parser.add_argument("--dataset-root", default="datasets")
    parser.add_argument("--latent-root", default="latents/img_pca")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--class-subset",
        choices=["default100", "default1000", "all"],
        default="default100",
    )
    parser.add_argument("--class-indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--output-path",
        default="outputs/pca_target_stats.json",
        help="Where to save full stats JSON.",
    )
    return parser.parse_args()


def _resolve_class_indices(
    dataset_root: Path,
    class_subset: str,
    class_indices: Sequence[int] | None,
) -> Sequence[int] | None:
    if class_indices is not None:
        return [int(x) for x in class_indices]
    if class_subset == "all":
        return None

    if class_subset == "default100":
        requested = DEFAULT_CLASS_INDICES
    else:
        requested = DEFAULT_CLASS_INDICES_1000

    metadata_path = dataset_root / "THINGS_EEG_2" / "image_metadata.npy"
    if not metadata_path.exists():
        return list(requested)
    metadata = np.load(metadata_path, allow_pickle=True).item()
    train_files = metadata.get("train_img_files")
    if train_files is None:
        return list(requested)
    num_images = int(len(train_files))
    images_per_class = 10
    if num_images <= 0 or num_images % images_per_class != 0:
        return list(requested)
    num_classes = num_images // images_per_class
    return [idx for idx in requested if int(idx) < int(num_classes)]


def _resolve_latent_path(latent_root: Path, image_index: int) -> Path:
    padded = latent_root / f"{int(image_index):06d}.pt"
    if padded.exists():
        return padded
    plain = latent_root / f"{int(image_index)}.pt"
    if plain.exists():
        return plain
    raise FileNotFoundError(
        f"Latent file not found for image_index={image_index}. "
        f"Tried: {padded} and {plain}"
    )


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(obj)}")
    return obj.to(torch.float32).flatten()


def _compute_split_stats(
    dataset_root: Path,
    latent_root: Path,
    split: str,
    class_indices: Sequence[int] | None,
    split_seed: int,
) -> dict:
    dataset = ImageDataset(
        dataset_root=str(dataset_root),
        split=split,
        class_indices=class_indices,
        image_transform=None,
        split_seed=split_seed,
    )
    image_indices = [int(x) for x in dataset._split_image_indices.tolist()]
    if not image_indices:
        raise RuntimeError(f"No samples found for split={split}")

    first = _load_tensor(_resolve_latent_path(latent_root=latent_root, image_index=image_indices[0]))
    dim = int(first.numel())

    sum_vec = torch.zeros(dim, dtype=torch.float64)
    sum_sq_vec = torch.zeros(dim, dtype=torch.float64)
    sum_l2 = 0.0
    n = 0

    for image_index in image_indices:
        x = _load_tensor(_resolve_latent_path(latent_root=latent_root, image_index=image_index))
        if x.numel() != dim:
            raise ValueError(
                f"Latent dimension mismatch in split={split}: expected {dim}, got {x.numel()} "
                f"for image_index={image_index}"
            )
        x64 = x.to(torch.float64)
        sum_vec += x64
        sum_sq_vec += x64 * x64
        sum_l2 += float(torch.linalg.vector_norm(x64, ord=2).item())
        n += 1

    mean_vec = sum_vec / max(n, 1)
    var_vec = (sum_sq_vec / max(n, 1)) - (mean_vec * mean_vec)
    var_vec = torch.clamp(var_vec, min=0.0)
    avg_l2 = sum_l2 / max(n, 1)

    return {
        "split": split,
        "num_samples": n,
        "dim": dim,
        "mean": mean_vec.to(torch.float32).tolist(),
        "variance": var_vec.to(torch.float32).tolist(),
        "average_l2_norm": float(avg_l2),
    }


def main():
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    latent_root = Path(args.latent_root)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {dataset_root}")
    if not latent_root.exists():
        raise FileNotFoundError(f"latent-root not found: {latent_root}")

    class_indices = _resolve_class_indices(
        dataset_root=dataset_root,
        class_subset=str(args.class_subset),
        class_indices=args.class_indices,
    )

    train_stats = _compute_split_stats(
        dataset_root=dataset_root,
        latent_root=latent_root,
        split="train",
        class_indices=class_indices,
        split_seed=int(args.split_seed),
    )
    valid_stats = _compute_split_stats(
        dataset_root=dataset_root,
        latent_root=latent_root,
        split="valid",
        class_indices=class_indices,
        split_seed=int(args.split_seed),
    )

    payload = {
        "dataset_root": str(dataset_root.resolve()),
        "latent_root": str(latent_root.resolve()),
        "split_seed": int(args.split_seed),
        "class_subset": str(args.class_subset),
        "class_indices": (list(class_indices) if class_indices is not None else None),
        "train": train_stats,
        "valid": valid_stats,
    }
    output_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved stats to: {output_path}")
    print(
        f"train: n={train_stats['num_samples']} dim={train_stats['dim']} "
        f"avg_l2={train_stats['average_l2_norm']:.6f}"
    )
    print(
        f"valid: n={valid_stats['num_samples']} dim={valid_stats['dim']} "
        f"avg_l2={valid_stats['average_l2_norm']:.6f}"
    )


if __name__ == "__main__":
    main()
