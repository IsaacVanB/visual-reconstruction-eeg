#!/usr/bin/env python3
"""Compare Stable Diffusion prompts with and without an EEG feature image."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import EEGLabelDataset
from src.evaluation.eeg_eval_core import (
    build_eeg_transform_from_saved_cfg,
    build_model_for_checkpoint,
    decode_from_lowres_vae_prediction,
    decode_from_pca_prediction,
    load_autoencoder_kl_class,
    load_checkpoint,
    load_ground_truth_tensor,
    load_metadata,
    load_pca_projection,
    load_scaling_factor,
    resolve_decode_latent_scaling_mode,
    resolve_image_path,
    resolve_pca_params_path,
    resolve_torch_device,
)
from src.evaluation.eval_eeg_encoder import (
    _compute_gt_lowres_vae_latent_from_image,
    _compute_gt_pca_latent_from_image,
)
from src.evaluation.generate_eeg_sd_grid import (
    _build_grid,
    _encoder_target_type,
    _load_sd_pipelines,
    _load_ssim_fn,
    _load_target_zscore_stats,
    _load_ground_truth,
    _pil_to_tensor_01,
    _resolve_best_checkpoint,
    _resolve_lowres_shapes,
    _tensor_to_pil,
    _unnormalize_lowres_target,
)
from src.evaluation.statistics import (
    paired_bootstrap_mean_difference_ci,
    paired_permutation_test_greater,
)


def normalize_subject(value: str) -> str:
    """Return a subject id in the repository's ``sub-N`` form."""
    return value if str(value).startswith("sub-") else f"sub-{value}"


def parse_prompt_file(path: Path | None) -> list[str]:
    """Read nonempty, non-comment prompt lines from a text file."""
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resolve_prompts(cli_prompts: Sequence[str] | None, prompt_file: Path | None) -> list[str]:
    """Combine repeated ``--prompt`` values and prompt-file lines."""
    prompts = [str(prompt).strip() for prompt in (cli_prompts or []) if str(prompt).strip()]
    prompts.extend(parse_prompt_file(prompt_file))
    if not prompts:
        raise ValueError("Provide at least one --prompt or --prompt-file.")
    return prompts


def resolve_repetitions(requested: Sequence[int], available: int) -> tuple[int, ...]:
    """Validate repetition indices without silently changing their meaning."""
    repetitions = tuple(int(value) for value in requested)
    if not repetitions:
        raise ValueError("--repetitions must contain at least one index.")
    if len(set(repetitions)) != len(repetitions):
        raise ValueError("--repetitions contains duplicates.")
    invalid = [value for value in repetitions if value < 0 or value >= available]
    if invalid:
        raise IndexError(f"Repetitions {invalid} are outside [0, {available - 1}].")
    return repetitions


def resolve_seeds(explicit: Sequence[int] | None, count: int, start: int) -> list[int]:
    """Return explicit seeds or ``count`` consecutive seeds beginning at ``start``."""
    if explicit is not None:
        seeds = [int(seed) for seed in explicit]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("--seeds must be nonempty and contain no duplicates.")
        return seeds
    if count < 1:
        raise ValueError("--num-seeds must be at least 1.")
    return list(range(int(start), int(start) + int(count)))


def create_output_dir(base: Path) -> Path:
    """Create a unique timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"run_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = base / f"run_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def safe_prompt_name(prompt: str, index: int) -> str:
    """Create a short stable directory name for one prompt."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", prompt).strip("_").lower()[:60]
    return f"prompt_{index + 1:02d}_{slug or 'text'}"


def select_eeg_and_image(
    dataset_root: str,
    subject: str,
    image_index: int,
    repetitions: Sequence[int],
) -> tuple[EEGLabelDataset, np.ndarray, str, Path]:
    """Load one condition and average exactly the selected EEG repetitions."""
    dataset = EEGLabelDataset(
        dataset_root=dataset_root,
        subject=subject,
        split="train",
        transform=None,
    )
    if image_index < 0 or image_index >= dataset.num_images:
        raise IndexError(f"--condition must be in [0, {dataset.num_images - 1}].")
    resolved_repetitions = resolve_repetitions(repetitions, dataset.repetitions)
    eeg = np.asarray(dataset.eeg[image_index, list(resolved_repetitions)], dtype=np.float32)
    eeg_mean = eeg.mean(axis=0)
    image_name = str(dataset.train_img_files[image_index])
    image_path = resolve_image_path(
        Path(dataset_root) / "images_THINGS" / "object_images", image_name
    )
    return dataset, eeg_mean, image_name, image_path


def load_encoder_and_reconstructions(
    checkpoint_path: Path,
    dataset_root: str,
    latent_root_override: str | None,
    pca_params_path: str | None,
    metadata_path: Path,
    latent_shape: tuple[int, int, int],
    eeg_np: np.ndarray,
    ground_truth_tensor: torch.Tensor,
    vae,
    device: torch.device,
    decode_scaling_arg: str,
) -> tuple[Image.Image, Image.Image, dict[str, Any]]:
    """Create the ideal target reconstruction and model-predicted feature image."""
    checkpoint, config = load_checkpoint(checkpoint_path)
    latent_root = str(latent_root_override or config.get("latent_root", config.get("image_latent_root", "")))
    if not latent_root:
        raise ValueError("Latent root is missing from both CLI and encoder checkpoint.")
    eeg_transform = build_eeg_transform_from_saved_cfg(config)
    sample_eeg = eeg_transform(eeg_np)
    output_dim = int(config["output_dim"])
    model = build_model_for_checkpoint(
        model_state_dict=checkpoint["model_state_dict"],
        sample_eeg=sample_eeg,
        sample_latent=torch.zeros(output_dim, dtype=torch.float32),
        saved_cfg=config,
        device=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        predicted = model(sample_eeg.unsqueeze(0).to(device=device, dtype=torch.float32))

    metadata = load_metadata(metadata_path)
    scaling_factor = load_scaling_factor(metadata_path, vae)
    decode_mode = resolve_decode_latent_scaling_mode(decode_scaling_arg, metadata)
    target_type = _encoder_target_type(config)
    c, h, w = latent_shape
    target_details: dict[str, Any]

    with torch.no_grad():
        if target_type == "pca":
            resolved_pca_path = resolve_pca_params_path(pca_params_path, latent_root)
            pca = load_pca_projection(resolved_pca_path, device)
            ideal_target = _compute_gt_pca_latent_from_image(
                gt_01_chw=ground_truth_tensor,
                vae=vae,
                pca=pca,
                device=device,
            )
            ideal_decoded = decode_from_pca_prediction(
                ideal_target, pca, latent_shape, vae, scaling_factor, decode_mode
            )[0]
            predicted_decoded = decode_from_pca_prediction(
                predicted, pca, latent_shape, vae, scaling_factor, decode_mode
            )[0]
            target_details = {"pca_params_path": str(resolved_pca_path), "pca_components": int(pca["k"])}
        else:
            lowres_shape, full_shape = _resolve_lowres_shapes(config, latent_shape)
            target_stats = _load_target_zscore_stats(config, lowres_shape, device)
            ideal_target = _compute_gt_lowres_vae_latent_from_image(
                gt_01_chw=ground_truth_tensor,
                vae=vae,
                lowres_shape=lowres_shape,
                device=device,
                downsample_mode=str(config.get("target_downsample_mode", "area")),
            )
            predicted_target = _unnormalize_lowres_target(predicted, target_stats)
            ideal_decoded = decode_from_lowres_vae_prediction(
                ideal_target, lowres_shape, full_shape, vae, scaling_factor, decode_mode
            )[0]
            predicted_decoded = decode_from_lowres_vae_prediction(
                predicted_target, lowres_shape, full_shape, vae, scaling_factor, decode_mode
            )[0]
            target_details = {
                "lowres_shape": list(lowres_shape),
                "full_latent_shape": list(full_shape),
                "target_downsample_mode": str(config.get("target_downsample_mode", "area")),
                "target_zscore_restored": target_stats is not None,
            }

    details = {
        "encoder_checkpoint": str(checkpoint_path),
        "encoder_target_type": target_type,
        "encoder_output_dim": output_dim,
        "encoder_config": config,
        "latent_root": latent_root,
        "vae_model": metadata.get("model_id"),
        "vae_scaling_factor": scaling_factor,
        "decode_latent_scaling_mode": decode_mode,
        **target_details,
    }
    return _tensor_to_pil(ideal_decoded), _tensor_to_pil(predicted_decoded), details


def generate_prompt_grid(
    prompt: str,
    seeds: Sequence[int],
    ground_truth: Image.Image,
    feature_image: Image.Image,
    text_pipe,
    img2img_pipe,
    ssim_fn,
    device: torch.device,
    negative_prompt: str | None,
    strength: float,
    guidance_scale: float,
    steps: int,
    image_size: int,
    output_dir: Path,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Generate paired prompt-only/img2img outputs and a seed-labeled SSIM grid."""
    label_only_images: list[Image.Image] = []
    label_feature_images: list[Image.Image] = []
    label_only_captions: list[str] = []
    label_feature_captions: list[str] = []
    results: list[dict[str, Any]] = []
    ground_truth_tensor = _pil_to_tensor_01(ground_truth, device)
    label_only_dir = output_dir / "label_only"
    label_feature_dir = output_dir / "label_plus_feature"
    label_only_dir.mkdir(parents=True, exist_ok=True)
    label_feature_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        label_only = text_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            height=image_size,
            width=image_size,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        generator = torch.Generator(device=device).manual_seed(int(seed))
        label_feature = img2img_pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=feature_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]
        ssim_label = float(
            ssim_fn(_pil_to_tensor_01(label_only, device), ground_truth_tensor, data_range=1.0)
            .detach().cpu().item()
        )
        ssim_feature = float(
            ssim_fn(_pil_to_tensor_01(label_feature, device), ground_truth_tensor, data_range=1.0)
            .detach().cpu().item()
        )
        label_only_path = label_only_dir / f"seed_{int(seed)}.png"
        label_feature_path = label_feature_dir / f"seed_{int(seed)}.png"
        label_only.save(label_only_path)
        label_feature.save(label_feature_path)
        label_only_images.append(label_only)
        label_feature_images.append(label_feature)
        label_only_captions.append(f"SSIM {ssim_label:.3f}")
        label_feature_captions.append(f"SSIM {ssim_feature:.3f}")
        results.append(
            {
                "seed": int(seed),
                "ssim_label_only": ssim_label,
                "ssim_label_feature": ssim_feature,
                "ssim_difference": ssim_feature - ssim_label,
                "label_only_path": str(label_only_path),
                "label_feature_path": str(label_feature_path),
            }
        )

    grid = _build_grid(
        rows=[("Label only", label_only_images), ("Label + feature", label_feature_images)],
        column_labels=[f"Seed {seed}" for seed in seeds],
        cell_captions=[label_only_captions, label_feature_captions],
    )
    return grid, results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, help="Subject number or sub-N id.")
    parser.add_argument("--condition", required=True, type=int, help="Zero-based global image index.")
    parser.add_argument("--repetitions", nargs="+", type=int, default=[0], help="Repetitions to average.")
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, help="Explicit seeds; overrides count/start.")
    parser.add_argument("--prompt", action="append", help="Prompt text; repeat for multiple prompts.")
    parser.add_argument("--prompt-file", type=Path, help="Text file containing one prompt per line.")
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--encoder-runs-dir", default="outputs/eeg_encoder_vae_lowres")
    parser.add_argument("--dataset-root")
    parser.add_argument("--latent-root")
    parser.add_argument("--pca-params-path")
    parser.add_argument("--metadata-path", type=Path, default=Path("latents/img_full_metadata.json"))
    parser.add_argument("--vae-name", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--sd-model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--latent-shape", nargs=3, type=int, default=[4, 64, 64])
    parser.add_argument("--decode-latent-scaling", choices=("auto", "divide", "none"), default="auto")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--guidance-scale", type=float, default=8.5)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument(
        "--ssim-permutation-test-permutations",
        type=int,
        default=10_000,
        help="Random sign-flip permutations for each paired one-sided SSIM test.",
    )
    parser.add_argument(
        "--ssim-permutation-test-seed",
        type=int,
        default=0,
        help="Random seed for the paired SSIM permutation tests.",
    )
    parser.add_argument("--ssim-bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--ssim-bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--ssim-bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--negative-prompt",
        default="low quality, blurry, distorted, deformed, out of frame, missing parts, partial object",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sd_prompt_comparison"))
    return parser.parse_args()


def main() -> None:
    """Run the prompt and seed comparison experiment."""
    args = parse_args()
    prompts = resolve_prompts(args.prompt, args.prompt_file)
    seeds = resolve_seeds(args.seeds, args.num_seeds, args.seed_start)
    subject = normalize_subject(args.subject)
    checkpoint_path = _resolve_best_checkpoint(args.encoder_checkpoint, args.encoder_runs_dir)
    _checkpoint, checkpoint_config = load_checkpoint(checkpoint_path)
    dataset_root = str(args.dataset_root or checkpoint_config.get("dataset_root", "datasets"))
    repetitions = tuple(int(value) for value in args.repetitions)
    dataset, eeg_np, image_name, image_path = select_eeg_and_image(
        dataset_root, subject, args.condition, repetitions
    )
    repetitions = resolve_repetitions(repetitions, dataset.repetitions)
    device = resolve_torch_device(args.device)
    output_dir = create_output_dir(args.output_dir)

    AutoencoderKL = load_autoencoder_kl_class()
    vae = AutoencoderKL.from_pretrained(args.vae_name).to(device).eval()
    ground_truth_tensor = load_ground_truth_tensor(
        image_root=Path(dataset_root) / "images_THINGS" / "object_images",
        image_name=image_name,
        width=args.image_size,
        height=args.image_size,
    )
    ground_truth = _load_ground_truth(image_path, args.image_size)
    ideal_target, model_feature, encoder_details = load_encoder_and_reconstructions(
        checkpoint_path=checkpoint_path,
        dataset_root=dataset_root,
        latent_root_override=args.latent_root,
        pca_params_path=args.pca_params_path,
        metadata_path=args.metadata_path,
        latent_shape=tuple(args.latent_shape),
        eeg_np=eeg_np,
        ground_truth_tensor=ground_truth_tensor,
        vae=vae,
        device=device,
        decode_scaling_arg=args.decode_latent_scaling,
    )
    ground_truth.save(output_dir / "ground_truth.png")
    ideal_target.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC).save(
        output_dir / "ideal_target_reconstruction.png"
    )
    model_feature = model_feature.resize(
        (args.image_size, args.image_size), Image.Resampling.BICUBIC
    )
    model_feature.save(output_dir / "model_low_level_feature.png")

    text_pipe, img2img_pipe = _load_sd_pipelines(args.sd_model_id, device, args.fp16)
    ssim_fn = _load_ssim_fn()
    prompt_summaries: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts):
        prompt_dir = output_dir / safe_prompt_name(prompt, prompt_index)
        prompt_dir.mkdir()
        grid, seed_results = generate_prompt_grid(
            prompt=prompt,
            seeds=seeds,
            ground_truth=ground_truth,
            feature_image=model_feature,
            text_pipe=text_pipe,
            img2img_pipe=img2img_pipe,
            ssim_fn=ssim_fn,
            device=device,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            steps=args.num_inference_steps,
            image_size=args.image_size,
            output_dir=prompt_dir,
        )
        grid_path = prompt_dir / "generated_grid.png"
        grid.save(grid_path)
        permutation_result = paired_permutation_test_greater(
            ssim_features=[row["ssim_label_feature"] for row in seed_results],
            ssim_label_only=[row["ssim_label_only"] for row in seed_results],
            n_permutations=args.ssim_permutation_test_permutations,
            seed=args.ssim_permutation_test_seed,
        )
        bootstrap_result = paired_bootstrap_mean_difference_ci(
            ssim_features=[row["ssim_label_feature"] for row in seed_results],
            ssim_label_only=[row["ssim_label_only"] for row in seed_results],
            confidence=args.ssim_bootstrap_confidence,
            n_bootstrap=args.ssim_bootstrap_iterations,
            seed=args.ssim_bootstrap_seed,
        )
        permutation_result["bootstrap_confidence_interval"] = bootstrap_result
        summary = {
            "prompt": prompt,
            "grid_path": str(grid_path),
            "average_ssim_label_only": float(np.mean([row["ssim_label_only"] for row in seed_results])),
            "average_ssim_label_feature": float(np.mean([row["ssim_label_feature"] for row in seed_results])),
            "average_ssim_difference": float(np.mean([row["ssim_difference"] for row in seed_results])),
            "ssim_paired_permutation_test": permutation_result,
            "seed_results": seed_results,
        }
        (prompt_dir / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (prompt_dir / "ssim_paired_permutation_test.json").write_text(
            json.dumps(permutation_result, indent=2), encoding="utf-8"
        )
        prompt_summaries.append(summary)
        print(
            f"Prompt {prompt_index + 1}/{len(prompts)}: "
            f"label-only SSIM={summary['average_ssim_label_only']:.4f}, "
            f"label+feature SSIM={summary['average_ssim_label_feature']:.4f}, "
            f"mean difference={permutation_result['observed_mean_difference']:.4f}, "
            f"{100 * bootstrap_result['confidence']:.0f}% CI="
            f"[{bootstrap_result['ci_lower']:.4f}, {bootstrap_result['ci_upper']:.4f}], "
            f"p={permutation_result['p_value_one_sided']:.6f}, "
            f"significant={permutation_result['alpha_0_05_significant']}"
        )

    pooled_rows = [
        row
        for prompt_summary in prompt_summaries
        for row in prompt_summary["seed_results"]
    ]
    pooled_permutation_result = paired_permutation_test_greater(
        ssim_features=[row["ssim_label_feature"] for row in pooled_rows],
        ssim_label_only=[row["ssim_label_only"] for row in pooled_rows],
        n_permutations=args.ssim_permutation_test_permutations,
        seed=args.ssim_permutation_test_seed,
    )
    pooled_bootstrap_result = paired_bootstrap_mean_difference_ci(
        ssim_features=[row["ssim_label_feature"] for row in pooled_rows],
        ssim_label_only=[row["ssim_label_only"] for row in pooled_rows],
        confidence=args.ssim_bootstrap_confidence,
        n_bootstrap=args.ssim_bootstrap_iterations,
        seed=args.ssim_bootstrap_seed,
    )
    pooled_permutation_result["bootstrap_confidence_interval"] = pooled_bootstrap_result

    metadata = {
        "subject": subject,
        "condition_image_index": int(args.condition),
        "image_name": image_name,
        "repetitions_averaged": list(repetitions),
        "num_repetitions_averaged": len(repetitions),
        "seeds": seeds,
        "num_seeds": len(seeds),
        "dataset_root": dataset_root,
        "ground_truth_path": str(output_dir / "ground_truth.png"),
        "ideal_target_path": str(output_dir / "ideal_target_reconstruction.png"),
        "model_feature_path": str(output_dir / "model_low_level_feature.png"),
        "stable_diffusion_model": args.sd_model_id,
        "vae_model": args.vae_name,
        "negative_prompt": args.negative_prompt,
        "image_size": args.image_size,
        "strength": args.strength,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "ssim_permutation_test_permutations": args.ssim_permutation_test_permutations,
        "ssim_permutation_test_seed": args.ssim_permutation_test_seed,
        "ssim_bootstrap_iterations": args.ssim_bootstrap_iterations,
        "ssim_bootstrap_confidence": args.ssim_bootstrap_confidence,
        "ssim_bootstrap_seed": args.ssim_bootstrap_seed,
        "pooled_ssim_paired_permutation_test": pooled_permutation_result,
        "pooled_test_note": (
            "Pooled across every prompt/seed pair. Per-prompt tests are stored in each "
            "prompt directory and are preferable when prompts are distinct hypotheses."
        ),
        "device": str(device),
        "fp16": bool(args.fp16 and device.type == "cuda"),
        **encoder_details,
        "prompts": prompt_summaries,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pooled_test_path = output_dir / "ssim_paired_permutation_test_pooled.json"
    pooled_test_path.write_text(
        json.dumps(pooled_permutation_result, indent=2), encoding="utf-8"
    )
    print(f"Saved prompt comparison: {output_dir}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Mean SSIM difference: {pooled_permutation_result['observed_mean_difference']:.6f}")
    print(
        f"{100 * pooled_bootstrap_result['confidence']:.0f}% bootstrap CI: "
        f"[{pooled_bootstrap_result['ci_lower']:.6f}, "
        f"{pooled_bootstrap_result['ci_upper']:.6f}]"
    )
    print(
        "One-sided paired permutation p-value: "
        f"{pooled_permutation_result['p_value_one_sided']:.6f} "
        f"(n={pooled_permutation_result['n']}, "
        f"significant={pooled_permutation_result['alpha_0_05_significant']})"
    )
    print(f"Saved pooled SSIM test: {pooled_test_path}")


if __name__ == "__main__":
    main()
