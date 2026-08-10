import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "analysis" / "compare_sd_prompts.py"
SPEC = importlib.util.spec_from_file_location("compare_sd_prompts", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_resolve_repetitions_preserves_explicit_selection():
    assert MODULE.resolve_repetitions([3, 1], 4) == (3, 1)


def test_resolve_repetitions_rejects_duplicates_and_out_of_range():
    with pytest.raises(ValueError, match="duplicates"):
        MODULE.resolve_repetitions([1, 1], 4)
    with pytest.raises(IndexError, match="outside"):
        MODULE.resolve_repetitions([4], 4)


def test_resolve_seeds_supports_count_or_explicit_values():
    assert MODULE.resolve_seeds(None, count=3, start=10) == [10, 11, 12]
    assert MODULE.resolve_seeds([7, 12], count=99, start=99) == [7, 12]


def test_resolve_prompts_combines_cli_and_file(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("# comment\na red object\n\na blue object\n")

    assert MODULE.resolve_prompts(["a green object"], prompt_file) == [
        "a green object",
        "a red object",
        "a blue object",
    ]
