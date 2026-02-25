import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PCA on flattened image embeddings and save transformed embeddings."
    )
    parser.add_argument(
        "--input-dir",
        default="latents/img_full",
        help="Directory containing input .pt embedding files.",
    )
    parser.add_argument(
        "--output-dir",
        default="latents/img_full_pca",
        help="Directory to save PCA-transformed .pt embeddings.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=64,
        help="Number of PCA components to keep.",
    )
    parser.add_argument(
        "--save-dtype",
        default="float16",
        choices=["float16", "float32"],
        help="Dtype for saved transformed embeddings.",
    )
    return parser.parse_args()


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor in {path}, got {type(tensor)}")
    return tensor


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() == ".pt"])
    if not paths:
        raise RuntimeError(f"No .pt files found in {input_dir}")

    flat_rows = []
    for path in paths:
        z = _load_tensor(path).to(dtype=torch.float32, device="cpu")
        flat_rows.append(z.reshape(-1))
    x = torch.stack(flat_rows, dim=0)  # [N, D]

    n_samples, n_features = x.shape
    k = min(args.n_components, n_samples, n_features)
    if k <= 0:
        raise ValueError(f"Invalid PCA components: {args.n_components}")
    if k != args.n_components:
        print(f"Adjusting n_components from {args.n_components} to {k} based on data shape.")

    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    _, _, v_t = torch.linalg.svd(x_centered, full_matrices=False)
    components = v_t[:k]  # [k, D]
    x_pca = x_centered @ components.T  # [N, k]

    out_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    for i, path in enumerate(paths):
        out_path = output_dir / path.name
        torch.save(x_pca[i].to(out_dtype), out_path)

    print(f"Processed {len(paths)} embeddings from {input_dir}")
    print(f"Input flattened shape: ({n_samples}, {n_features})")
    print(f"Output embedding shape per file: ({k},)")
    print(f"Saved PCA embeddings to: {output_dir}")


if __name__ == "__main__":
    main()
