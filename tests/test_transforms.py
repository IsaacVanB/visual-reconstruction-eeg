import numpy as np
import pytest
import torch

from src.data.transforms import (
    EEGChannelZScoreNormalize,
    EEGPerSampleNormalize,
    build_eeg_transform,
    crop_eeg_time_window,
    resolve_eeg_time_window,
)


def test_per_sample_normalization_has_unit_norm():
    eeg = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    normalized = EEGPerSampleNormalize()(eeg)

    assert normalized.dtype == np.float32
    assert np.linalg.norm(normalized) == pytest.approx(1.0)


def test_per_sample_normalization_leaves_zero_input_finite():
    normalized = EEGPerSampleNormalize()(np.zeros((2, 3), dtype=np.float32))

    assert np.array_equal(normalized, np.zeros((2, 3), dtype=np.float32))
    assert np.isfinite(normalized).all()


def test_channel_zscore_uses_per_channel_statistics():
    eeg = np.array([[2.0, 4.0], [12.0, 16.0]], dtype=np.float32)
    transform = EEGChannelZScoreNormalize(mean=[3.0, 10.0], std=[1.0, 2.0])

    normalized = transform(eeg)

    np.testing.assert_allclose(normalized, [[-1.0, 1.0], [1.0, 3.0]])


def test_resolve_and_crop_time_window():
    times = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=np.float32)
    eeg = np.arange(10, dtype=np.float32).reshape(2, 5)

    window = resolve_eeg_time_window(times, pre_ms=100, post_ms=100)
    cropped = crop_eeg_time_window(eeg, window["start_idx"], window["end_idx"])

    assert window["num_timepoints"] == 3
    np.testing.assert_array_equal(cropped, eeg[:, 1:4])


def test_time_window_requires_both_bounds():
    with pytest.raises(ValueError, match="Both pre_ms and post_ms"):
        resolve_eeg_time_window(np.arange(3), pre_ms=100)


def test_build_eeg_transform_crops_normalizes_and_converts_to_tensor():
    eeg = np.arange(12, dtype=np.float32).reshape(2, 6)
    transform = build_eeg_transform(
        normalize_mode="l2",
        crop_start_idx=1,
        crop_end_idx=4,
        to_tensor=True,
    )

    transformed = transform(eeg)

    assert isinstance(transformed, torch.Tensor)
    assert transformed.shape == (2, 4)
    assert transformed.dtype == torch.float32
    assert torch.linalg.vector_norm(transformed).item() == pytest.approx(1.0)
