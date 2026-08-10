"""Shared statistical tests for EEG reconstruction evaluations."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def paired_permutation_test_greater(
    ssim_features: Sequence[float] | np.ndarray,
    ssim_label_only: Sequence[float] | np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Run a one-sided paired random-sign permutation test on SSIM scores.

    The hypotheses are::

        H0: mean(ssim_features - ssim_label_only) <= 0
        H1: mean(ssim_features - ssim_label_only) > 0

    Non-finite pairs are removed together. The returned p-value uses the
    standard plus-one correction so it is never zero.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1.")

    features = np.asarray(ssim_features, dtype=float)
    label_only = np.asarray(ssim_label_only, dtype=float)
    if features.shape != label_only.shape:
        raise ValueError("ssim_features and ssim_label_only must have the same shape.")
    if features.ndim != 1:
        raise ValueError("Inputs should be 1D arrays of matched SSIM scores.")

    valid = np.isfinite(features) & np.isfinite(label_only)
    features = features[valid]
    label_only = label_only[valid]
    if features.size == 0:
        raise ValueError("No valid paired SSIM scores remain after removing NaNs/Infs.")

    differences = features - label_only
    observed_mean_difference = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    permuted_mean_differences = np.empty(n_permutations, dtype=float)
    for index in range(n_permutations):
        signs = rng.choice([-1, 1], size=differences.size)
        permuted_mean_differences[index] = np.mean(signs * differences)

    p_value = (np.sum(permuted_mean_differences >= observed_mean_difference) + 1) / (
        n_permutations + 1
    )
    return {
        "n": int(differences.size),
        "observed_mean_ssim_features": float(np.mean(features)),
        "observed_mean_ssim_label_only": float(np.mean(label_only)),
        "observed_mean_difference": observed_mean_difference,
        "p_value_one_sided": float(p_value),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "alternative": "mean(ssim_label_image - ssim_label_only) > 0",
        "alpha_0_05_significant": bool(p_value < 0.05),
    }


def paired_bootstrap_mean_difference_ci(
    ssim_features: Sequence[float] | np.ndarray,
    ssim_label_only: Sequence[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate a percentile bootstrap CI for the paired mean SSIM difference.

    Each bootstrap draw resamples pair indices, preserving the correspondence
    between feature-image and label-only SSIM measurements.
    """
    features = np.asarray(ssim_features, dtype=float)
    label_only = np.asarray(ssim_label_only, dtype=float)
    if features.shape != label_only.shape:
        raise ValueError("ssim_features and ssim_label_only must have the same shape.")
    if features.ndim != 1:
        raise ValueError("Inputs should be 1D arrays of matched SSIM scores.")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be >= 1.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    valid = np.isfinite(features) & np.isfinite(label_only)
    features = features[valid]
    label_only = label_only[valid]
    if features.size == 0:
        raise ValueError("No valid paired SSIM scores remain after removing NaNs/Infs.")

    diffs = features - label_only
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled_indices = rng.choice(diffs.size, size=diffs.size, replace=True)
        bootstrap_means[index] = np.mean(diffs[sampled_indices])

    alpha = 1.0 - confidence
    lower = np.percentile(bootstrap_means, 100.0 * alpha / 2.0)
    upper = np.percentile(bootstrap_means, 100.0 * (1.0 - alpha / 2.0))
    return {
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }
