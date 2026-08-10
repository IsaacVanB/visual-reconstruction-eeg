import numpy as np
import pytest

from src.evaluation.statistics import (
    paired_bootstrap_mean_difference_ci,
    paired_permutation_test_greater,
)


def test_paired_permutation_test_reports_positive_feature_improvement():
    result = paired_permutation_test_greater(
        ssim_features=[0.5, 0.6, 0.7, 0.8],
        ssim_label_only=[0.1, 0.2, 0.3, 0.4],
        n_permutations=1000,
        seed=0,
    )

    assert result["n"] == 4
    assert result["observed_mean_difference"] == pytest.approx(0.4)
    assert 0.0 < result["p_value_one_sided"] <= 1.0


def test_paired_permutation_test_filters_nonfinite_pairs_together():
    result = paired_permutation_test_greater(
        ssim_features=[0.5, np.nan, 0.7],
        ssim_label_only=[0.2, 0.3, np.inf],
        n_permutations=10,
        seed=1,
    )

    assert result["n"] == 1
    assert result["observed_mean_difference"] == pytest.approx(0.3)


def test_paired_permutation_test_validates_inputs():
    with pytest.raises(ValueError, match="same shape"):
        paired_permutation_test_greater([0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="n_permutations"):
        paired_permutation_test_greater([0.1], [0.1], n_permutations=0)


def test_paired_bootstrap_mean_difference_ci_is_paired_and_deterministic():
    kwargs = dict(
        ssim_features=[0.5, 0.6, 0.9],
        ssim_label_only=[0.2, 0.4, 0.3],
        confidence=0.95,
        n_bootstrap=1000,
        seed=7,
    )
    first = paired_bootstrap_mean_difference_ci(**kwargs)
    second = paired_bootstrap_mean_difference_ci(**kwargs)

    assert first == second
    assert first["ci_lower"] <= np.mean([0.3, 0.2, 0.6]) <= first["ci_upper"]
    assert first["confidence"] == 0.95
    assert first["n_bootstrap"] == 1000
    assert first["seed"] == 7


def test_paired_bootstrap_filters_invalid_pairs_and_validates_options():
    result = paired_bootstrap_mean_difference_ci(
        [0.5, np.nan, 0.9], [0.2, 0.4, np.inf], n_bootstrap=10
    )
    assert result["ci_lower"] == pytest.approx(0.3)
    assert result["ci_upper"] == pytest.approx(0.3)

    with pytest.raises(ValueError, match="same shape"):
        paired_bootstrap_mean_difference_ci([0.1], [0.1, 0.2])
    with pytest.raises(ValueError, match="1D"):
        paired_bootstrap_mean_difference_ci([[0.1]], [[0.0]])
    with pytest.raises(ValueError, match="n_bootstrap"):
        paired_bootstrap_mean_difference_ci([0.1], [0.0], n_bootstrap=0)
    with pytest.raises(ValueError, match="confidence"):
        paired_bootstrap_mean_difference_ci([0.1], [0.0], confidence=1.0)
