import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageLatentDataset
from src.evaluation.eeg_eval_core import (
    build_eeg_transform_from_saved_cfg,
    build_model_for_checkpoint,
    decode_from_pca_prediction,
    filter_sample_index_to_existing_files,
    find_latest_run_dir,
    load_checkpoint,
    load_autoencoder_kl_class,
    load_ground_truth_tensor,
    load_metadata,
    load_pca_projection,
    load_scaling_factor,
    resolve_checkpoint_for_run,
    resolve_decode_latent_scaling_mode,
    resolve_eval_overrides,
    resolve_pca_params_path,
    resolve_torch_device,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate EEG encoder on test split and decode predicted images."
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Path to trained EEG encoder checkpoint.",
    )
    parser.add_argument("--dataset-root", default=None, help="Override dataset root from checkpoint.")
    parser.add_argument("--latent-root", default=None, help="Override latent root from checkpoint.")
    parser.add_argument("--subject", default=None, help="Override subject from checkpoint.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help="Override split seed from checkpoint.",
    )
    parser.add_argument(
        "--class-indices",
        type=int,
        nargs="+",
        default=None,
        help="Override class indices from checkpoint.",
    )
    parser.add_argument(
        "--pca-params-path",
        default=None,
        help=(
            "Path to PCA params with pca_mean and pca_components. "
            "If omitted or missing, auto-detect newest pca_*.pt in --latent-root."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        default="latents/img_full_metadata.json",
        help="Path to metadata json containing scaling_factor.",
    )
    parser.add_argument(
        "--decode-latent-scaling",
        default="auto",
        choices=["auto", "divide", "none"],
        help=(
            "How to adapt predicted VAE latents before decode. "
            "'auto' infers from metadata; 'divide' uses z/scaling_factor; "
            "'none' decodes z directly."
        ),
    )
    parser.add_argument(
        "--vae-name",
        default="stabilityai/sd-vae-ft-mse",
        help="Diffusers VAE model id.",
    )
    parser.add_argument(
        "--latent-shape",
        type=int,
        nargs=3,
        default=[4, 64, 64],
        help="Decoded latent shape after inverse PCA (default: 4 64 64).",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick runs.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Alias for --max-samples.",
    )
    parser.add_argument(
        "--grid-images",
        type=int,
        default=8,
        help="Number of columns in recon_grid.png (ground truth over reconstruction).",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, etc.")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _tensor_to_pil(image_chw_01: torch.Tensor) -> Image.Image:
    image_np = (
        image_chw_01.detach()
        .clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    return Image.fromarray(image_np)


def _tensor_to_uint8_hwc(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().clamp(0.0, 1.0)
    image = (image * 255.0).round().to(dtype=torch.uint8)
    return image.permute(1, 2, 0).numpy()


def _build_reconstruction_grid(
    originals: torch.Tensor, reconstructions: torch.Tensor, max_items: int
) -> Image.Image:
    n = min(max_items, originals.size(0), reconstructions.size(0))
    if n <= 0:
        raise ValueError("No images available to build a reconstruction grid.")

    _, _, h, w = originals.shape
    canvas = np.zeros((2 * h, n * w, 3), dtype=np.uint8)
    for i in range(n):
        canvas[0:h, i * w : (i + 1) * w] = _tensor_to_uint8_hwc(originals[i])
        canvas[h : 2 * h, i * w : (i + 1) * w] = _tensor_to_uint8_hwc(reconstructions[i])
    return Image.fromarray(canvas)


def _iter_eeg_label_batches(dataset: EEGImageLatentDataset, batch_size: int):
    total = len(dataset._sample_index)
    for start in range(0, total, batch_size):
        batch_samples = dataset._sample_index[start : start + batch_size]
        eeg_batch = []
        labels = []
        for image_index, rep_index in batch_samples:
            eeg_sample = dataset.eeg[image_index, rep_index]
            if dataset.transform:
                eeg_sample = dataset.transform(eeg_sample)
            if not torch.is_tensor(eeg_sample):
                eeg_sample = torch.as_tensor(eeg_sample)
            eeg_batch.append(eeg_sample)
            labels.append(int(image_index) // int(dataset.images_per_class))
        yield torch.stack(eeg_batch, dim=0), torch.tensor(labels, dtype=torch.long), batch_samples


def main():
    args = parse_args()

    device_t = resolve_torch_device(args.device)

    runs_base = repo_root / "outputs" / "eeg_encoder"
    run_dir: Path | None = None
    if args.checkpoint_path is None:
        run_dir = find_latest_run_dir(runs_base)
        checkpoint_path = resolve_checkpoint_for_run(run_dir)
        print(f"Auto-selected latest run: {run_dir}")
        print(f"Auto-selected checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = Path(args.checkpoint_path)
        if checkpoint_path.parent.name.startswith("run_"):
            run_dir = checkpoint_path.parent
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt, saved_cfg = load_checkpoint(checkpoint_path)

    if args.max_samples is None:
        if args.num_images is not None:
            args.max_samples = args.num_images
        elif saved_cfg.get("eval_max_samples", None) is not None:
            args.max_samples = int(saved_cfg["eval_max_samples"])

    dataset_root, latent_root, subject, split_seed, class_indices = resolve_eval_overrides(
        saved_cfg=saved_cfg,
        ckpt=ckpt,
        dataset_root=args.dataset_root,
        latent_root=args.latent_root,
        subject=args.subject,
        split_seed=args.split_seed,
        class_indices=args.class_indices,
    )

    eeg_tf = build_eeg_transform_from_saved_cfg(saved_cfg)
    dataset = EEGImageLatentDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="test",
        class_indices=class_indices,
        transform=eeg_tf,
        latent_root=latent_root,
        split_seed=split_seed,
    )
    image_root = Path(dataset_root) / "images_THINGS" / "object_images"
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    filtered_samples, missing_test_images = filter_sample_index_to_existing_files(
        sample_index=dataset._sample_index,
        train_img_files=dataset.train_img_files,
        image_root=image_root,
    )
    if missing_test_images:
        dataset._sample_index = filtered_samples
        print(
            "Warning: Skipping test samples with missing ground-truth images: "
            f"{len(missing_test_images)} image ids removed."
        )
    if len(dataset) == 0:
        raise RuntimeError("No test samples remain after filtering missing ground-truth images.")

    if len(dataset._sample_index) == 0:
        raise RuntimeError("No test samples remain after filtering.")
    first_image_index, first_rep_index = dataset._sample_index[0]
    sample_eeg = dataset.eeg[first_image_index, first_rep_index]
    if dataset.transform:
        sample_eeg = dataset.transform(sample_eeg)
    if not torch.is_tensor(sample_eeg):
        sample_eeg = torch.as_tensor(sample_eeg)
    sample_latent = torch.zeros(int(saved_cfg.get("output_dim", 512)), dtype=torch.float32)
    model = build_model_for_checkpoint(
        model_state_dict=ckpt["model_state_dict"],
        sample_eeg=sample_eeg,
        sample_latent=sample_latent,
        saved_cfg=saved_cfg,
        device=device_t,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pca_params_path = resolve_pca_params_path(
        pca_params_path=args.pca_params_path,
        latent_root=latent_root,
    )
    pca = load_pca_projection(
        pca_params_path=pca_params_path,
        device=device_t,
    )
    if int(saved_cfg.get("output_dim", pca["k"])) != int(pca["k"]):
        raise ValueError(
            f"Encoder output_dim ({saved_cfg.get('output_dim')}) does not match PCA k ({pca['k']})."
        )

    c, h, w = args.latent_shape
    if c * h * w != int(pca["d"]):
        raise ValueError(
            f"latent-shape {tuple(args.latent_shape)} has {c*h*w} elements, but PCA D={pca['d']}."
        )

    AutoencoderKL = load_autoencoder_kl_class()
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device_t).eval()
    metadata = load_metadata(Path(args.metadata_path))
    scaling_factor = load_scaling_factor(Path(args.metadata_path), vae)
    decode_scaling_mode = resolve_decode_latent_scaling_mode(
        mode_arg=args.decode_latent_scaling,
        metadata=metadata,
    )

    if args.output_dir is None:
        if run_dir is None:
            output_dir = runs_base / "eval"
        else:
            output_dir = run_dir / "eval"
        print(f"Auto-selected output directory: {output_dir}")
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device_t}")
    print(f"Test samples: {len(dataset)}")
    print(f"PCA shape: k={pca['k']}, D={pca['d']}")
    print(f"Scaling factor: {scaling_factor}")
    print(f"PCA standardized: {pca['standardized']}")
    print(f"Decode latent scaling mode: {decode_scaling_mode}")
    saved = 0
    originals_for_grid = []
    recons_for_grid = []
    with torch.no_grad():
        for eeg, labels, batch_samples in _iter_eeg_label_batches(
            dataset=dataset, batch_size=args.batch_size
        ):
            remaining = eeg.size(0)
            if args.max_samples is not None:
                remaining = min(remaining, args.max_samples - saved)
            if remaining <= 0:
                break

            eeg = eeg[:remaining].to(device=device_t, dtype=torch.float32)
            labels = labels[:remaining]
            batch_samples = batch_samples[:remaining]

            pred_pca = model(eeg)
            if pred_pca.shape[1] != int(pca["k"]):
                raise RuntimeError(
                    f"Predicted latent dim {pred_pca.shape[1]} does not match PCA k={pca['k']}."
                )
            recon_01 = decode_from_pca_prediction(
                pred_pca=pred_pca,
                pca=pca,
                latent_shape=(c, h, w),
                vae=vae,
                scaling_factor=scaling_factor,
                decode_scaling_mode=decode_scaling_mode,
            )
            _, _, recon_h, recon_w = recon_01.shape

            for j in range(recon_01.size(0)):
                if args.max_samples is not None and saved >= args.max_samples:
                    break
                image_index, rep_index = batch_samples[j]
                label = int(labels[j].item())
                out_name = (
                    f"label_{label:04d}_img_{int(image_index):06d}_"
                    f"rep_{int(rep_index)}.png"
                )
                _tensor_to_pil(recon_01[j]).save(output_dir / out_name)
                if len(originals_for_grid) < args.grid_images:
                    image_name = dataset.train_img_files[int(image_index)]
                    gt = load_ground_truth_tensor(
                        image_root=image_root,
                        image_name=image_name,
                        width=int(recon_w),
                        height=int(recon_h),
                    )
                    originals_for_grid.append(gt)
                    recons_for_grid.append(recon_01[j].detach().cpu())
                saved += 1
            if args.max_samples is not None and saved >= args.max_samples:
                break
            if saved % 100 == 0 and saved > 0:
                print(f"Saved {saved} images...")

    if originals_for_grid:
        originals_t = torch.stack(originals_for_grid, dim=0)
        recons_t = torch.stack(recons_for_grid, dim=0)
        grid = _build_reconstruction_grid(
            originals=originals_t,
            reconstructions=recons_t,
            max_items=args.grid_images,
        )
        grid_path = output_dir / "recon_grid.png"
        grid.save(grid_path)
        print(f"Saved recon grid: {grid_path}")

    print(f"Saved decoded images: {saved}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
