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
        default="float32",
        choices=["float16", "float32"],
        help="Dtype for saved transformed embeddings.",
    )
    parser.add_argument(
        "--pca-params-path",
        default=None,
        help="Optional path for PCA params file. Default: <output-dir>/pca_<k>.pt",
    )
    parser.add_argument(
        "--no-explained-variance",
        action="store_true",
        help="If set, do not store explained_variance in PCA params file.",
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
    # After centering, PCA rank is bounded by N-1.
    max_rank = max(min(n_samples - 1, n_features), 1)
    k = min(args.n_components, max_rank)
    if k <= 0:
        raise ValueError(f"Invalid PCA components: {args.n_components}")
    if k != args.n_components:
        print(
            f"Adjusting n_components from {args.n_components} to {k} "
            f"(max valid rank with centered data is {max_rank})."
        )

    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    _, s, v_t = torch.linalg.svd(x_centered, full_matrices=False)
    components = v_t[:k]  # [k, D]
    x_pca = x_centered @ components.T  # [N, k]

    explained_variance = (s[:k] ** 2) / max(n_samples - 1, 1)

    out_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    for i, path in enumerate(paths):
        out_path = output_dir / path.name
        torch.save(x_pca[i].to(out_dtype), out_path)

    pca_params_path = Path(args.pca_params_path) if args.pca_params_path else output_dir / f"pca_{k}.pt"
    pca_payload = {
        "pca_mean": mean[0].to(torch.float32),          # [D]
        "pca_components": components.to(torch.float32), # [k, D]
    }
    if not args.no_explained_variance:
        pca_payload["explained_variance"] = explained_variance.to(torch.float32)  # [k]
    torch.save(pca_payload, pca_params_path)

    print(f"Processed {len(paths)} embeddings from {input_dir}")
    print(f"Input flattened shape: ({n_samples}, {n_features})")
    print(f"Output embedding shape per file: ({k},)")
    print(f"Saved PCA embeddings to: {output_dir}")
    print(f"Saved PCA params to: {pca_params_path}")


if __name__ == "__main__":
    main()
