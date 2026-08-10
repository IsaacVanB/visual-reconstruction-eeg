import numpy as np
import pytest

from src.evaluation.statistics import paired_permutation_test_greater


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
