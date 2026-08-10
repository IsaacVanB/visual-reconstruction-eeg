import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "analysis"
    / "evaluate_correct_classifications_sd.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_correct_classifications_sd", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FixedClassifier(torch.nn.Module):
    def forward(self, values):
        return values


def test_classify_repetitions_keeps_each_trial_independent():
    logits = np.asarray([[4.0, 1.0], [0.0, 3.0], [2.0, 1.0]], dtype=np.float32)
    results = MODULE.classify_repetitions(
        classifier=FixedClassifier(),
        classifier_transform=torch.as_tensor,
        eeg_repetitions=logits,
        true_label=0,
        device=torch.device("cpu"),
    )

    assert [row["repetition"] for row in results] == [0, 1, 2]
    assert [row["correct"] for row in results] == [True, False, True]


def test_resolve_seeds_defaults_to_requested_consecutive_count():
    assert MODULE.resolve_seeds(None, 5, 10) == [10, 11, 12, 13, 14]
    assert MODULE.resolve_seeds([3, 7], 5, 0) == [3, 7]


def test_resolve_seeds_rejects_invalid_values():
    with pytest.raises(ValueError, match="at least 1"):
        MODULE.resolve_seeds(None, 0, 0)
    with pytest.raises(ValueError, match="unique"):
        MODULE.resolve_seeds([1, 1], 5, 0)
