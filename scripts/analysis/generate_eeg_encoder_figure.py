#!/usr/bin/env python3
"""Generate a publication-friendly diagram of the project's EEG encoder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import EEGEncoderCNN


def infer_stage_shapes(model: EEGEncoderCNN) -> list[tuple[int, ...]]:
    """Run a dummy batch through the real modules and return displayed shapes."""
    shapes: list[tuple[int, ...]] = [(1, model.eeg_channels, model.eeg_timesteps)]
    x = torch.zeros(1, model.eeg_channels, model.eeg_timesteps)
    with torch.no_grad():
        x = x.unsqueeze(1)
        shapes.append(tuple(x.shape))
        for index, layer in enumerate(model.features):
            x = layer(x)
            if index in {3, 7, 12, 17}:
                shapes.append(tuple(x.shape))
        x = x.squeeze(2)
        x = model.pool(x)
        shapes.append(tuple(x.shape))
        x = model.head(x)
        shapes.append(tuple(x.shape))
    return shapes


def shape_text(shape: tuple[int, ...]) -> str:
    """Format a tensor shape using B for the batch dimension."""
    return "[B, " + ", ".join(str(value) for value in shape[1:]) + "]"


def add_box(
    ax,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    title: str,
    details: str,
    shape: str,
    color: str,
) -> None:
    """Draw one rounded architecture stage."""
    patch = FancyBboxPatch(
        (center_x - width / 2, center_y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.025",
        linewidth=1.6,
        edgecolor="#243447",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(center_x, center_y + height * 0.27, title, ha="center", va="center", fontsize=10.3, weight="bold")
    ax.text(center_x, center_y, details, ha="center", va="center", fontsize=7.8, linespacing=1.3)
    ax.text(center_x, center_y - height * 0.31, shape, ha="center", va="center", fontsize=9, family="monospace")


def add_arrow(ax, start_x: float, end_x: float, y: float) -> None:
    """Draw a horizontal arrow between adjacent stages."""
    ax.add_patch(
        FancyArrowPatch(
            (start_x, y),
            (end_x, y),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.4,
            color="#44546a",
        )
    )


def build_figure(output_dim: int, target_type: str, output_path: Path) -> None:
    """Build and save the encoder architecture diagram."""
    model = EEGEncoderCNN(eeg_channels=17, eeg_timesteps=51, output_dim=output_dim).eval()
    shapes = infer_stage_shapes(model)
    target_label = "PCA coefficients" if target_type == "pca" else "Low-res VAE latent (4×8×8)"
    stages = [
        ("EEG input", "17 posterior channels\n0–500 ms at 100 Hz", shape_text(shapes[0]), "#dceeff"),
        ("Add image axis", "Unsqueeze", shape_text(shapes[1]), "#eef5fb"),
        ("Temporal\nfiltering", "Conv2D 1×15\n32 filters\nGroupNorm · GELU\nDropout 0.1", shape_text(shapes[2]), "#dff3e4"),
        ("Spatial\nfiltering", "Depthwise Conv2D 17×1\n32 groups · 64 outputs\nGroupNorm · GELU\nDropout 0.1", shape_text(shapes[3]), "#dff3e4"),
        ("Separable\ntemporal", "Depthwise Conv 1×9\nPointwise 64→128\nGroupNorm · GELU\nDropout 0.1", shape_text(shapes[4]), "#fff0d6"),
        ("Temporal\ndownsample", "Depthwise Conv 1×5\nStride 2\nPointwise 128→128\nGroupNorm · GELU\nDropout 0.1", shape_text(shapes[5]), "#fff0d6"),
        ("Adaptive\npooling", "Squeeze spatial axis\nAdaptive average pool\nto 8 time bins", shape_text(shapes[6]), "#f3e5f5"),
        ("Prediction head", "Flatten 1024\nLinear 1024→512→256→K\nGELU · Dropout 0.2", shape_text(shapes[7]), "#f9dfe7"),
    ]

    fig, ax = plt.subplots(figsize=(21, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    centers = [0.06, 0.185, 0.31, 0.435, 0.56, 0.685, 0.81, 0.935]
    widths = [0.108] * len(centers)
    height = 0.43
    y = 0.52
    for index, ((title, details, shape, color), center, width) in enumerate(zip(stages, centers, widths)):
        add_box(ax, center, y, width, height, title, details, shape, color)
        if index:
            add_arrow(ax, centers[index - 1] + widths[index - 1] / 2 + 0.006, center - width / 2 - 0.006, y)

    ax.text(0.5, 0.91, "EEG Encoder Architecture", ha="center", fontsize=21, weight="bold", color="#1f2d3d")
    ax.text(
        0.5,
        0.855,
        "EEGNet-style depthwise/separable convolutional encoder",
        ha="center",
        fontsize=12.5,
        color="#526579",
    )
    ax.text(
        centers[-1],
        0.225,
        f"K = {output_dim}\n{target_label}",
        ha="center",
        va="top",
        fontsize=10.5,
        weight="bold",
        color="#8b1e3f",
    )
    ax.text(
        0.5,
        0.075,
        "Input preprocessing: 45 Hz low-pass filter → 0–500 ms crop → train-set channel-wise z-score",
        ha="center",
        fontsize=11,
        color="#44546a",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved EEG encoder architecture figure: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse output-target options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("figures/eeg_encoder.png"))
    parser.add_argument("--target-type", choices=("pca", "vae_lowres"), default="vae_lowres")
    parser.add_argument("--output-dim", type=int, help="Defaults to 4 for PCA or 256 for low-res VAE.")
    return parser.parse_args()


def main() -> None:
    """Generate the requested architecture image."""
    args = parse_args()
    output_dim = args.output_dim if args.output_dim is not None else (4 if args.target_type == "pca" else 256)
    if output_dim < 1:
        raise ValueError("--output-dim must be positive.")
    build_figure(output_dim=output_dim, target_type=args.target_type, output_path=args.output)


if __name__ == "__main__":
    main()
