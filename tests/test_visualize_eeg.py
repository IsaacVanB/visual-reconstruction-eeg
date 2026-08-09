import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "visualize_eeg.py"
SPEC = importlib.util.spec_from_file_location("visualize_eeg", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_compute_spectrum_recovers_sinusoid_amplitude_and_frequency():
    sampling_rate = 100.0
    times = np.arange(100, dtype=np.float64) / sampling_rate
    waveform = 2.5 + 3.0 * np.sin(2 * np.pi * 10 * times)

    frequencies, amplitude = MODULE.compute_spectrum(waveform, sampling_rate, "linear")

    peak = int(np.argmax(amplitude))
    assert frequencies[peak] == pytest.approx(10.0)
    assert amplitude[peak] == pytest.approx(3.0, rel=0.02)
    assert amplitude[0] < 1e-3


def test_compute_spectrum_uses_true_unpadded_frequency_spacing():
    frequencies, _ = MODULE.compute_spectrum(np.ones(51), 100.0, "linear")

    assert frequencies[1] - frequencies[0] == pytest.approx(100.0 / 51.0)
