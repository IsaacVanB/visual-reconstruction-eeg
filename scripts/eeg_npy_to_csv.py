#!/usr/bin/env python3
"""Convert THINGS EEG `.npy` data into human-readable summary files.

This script is designed for files like:
`datasets/THINGS_EEG_2/sub-1/preprocessed_eeg_training.npy`

The source `.npy` stores a pickled dictionary with:
- `preprocessed_eeg_data`: EEG array shaped `(images, repetitions, channels, timepoints)`
- `ch_names`: channel names
- `times`: time values for each timepoint

By default, the script writes:
- a JSON summary of the dataset
- a CSV export for one EEG sample (one image index + one repetition)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_preprocessed_eeg(path: Path) -> dict[str, Any]:
    raw = np.load(path, allow_pickle=True)

    if isinstance(raw, np.ndarray) and raw.dtype == object and raw.ndim == 0:
        raw = raw.item()

    if not isinstance(raw, dict):
        raise TypeError(
            "Expected the `.npy` file to contain a dictionary with EEG metadata and data."
        )

    required_keys = {"preprocessed_eeg_data", "ch_names", "times"}
    missing_keys = required_keys.difference(raw.keys())
    if missing_keys:
        raise KeyError(f"Missing expected keys: {sorted(missing_keys)}")

    return raw


def _default_summary_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_summary.json")


def _default_sample_path(input_path: Path, image_index: int, repetition: int) -> Path:
    return input_path.with_name(
        f"{input_path.stem}_image_{image_index}_rep_{repetition}.csv"
    )


def _write_summary(
    output_path: Path,
    input_path: Path,
    eeg: np.ndarray,
    ch_names: list[str],
    times: np.ndarray,
) -> None:
    summary = {
        "source_file": str(input_path),
        "dtype": str(eeg.dtype),
        "shape": list(eeg.shape),
        "num_images": int(eeg.shape[0]),
        "num_repetitions": int(eeg.shape[1]),
        "num_channels": int(eeg.shape[2]),
        "num_timepoints": int(eeg.shape[3]),
        "channel_names": ch_names,
        "time_start": float(times[0]) if len(times) else None,
        "time_end": float(times[-1]) if len(times) else None,
        "time_count": int(len(times)),
        "preview_first_sample_first_channel_first_10_values": [
            float(value) for value in eeg[0, 0, 0, :10]
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


def _write_sample_csv(
    output_path: Path,
    eeg: np.ndarray,
    ch_names: list[str],
    times: np.ndarray,
    image_index: int,
    repetition: int,
) -> None:
    if image_index < 0 or image_index >= eeg.shape[0]:
        raise IndexError(f"image_index must be in [0, {eeg.shape[0] - 1}].")
    if repetition < 0 or repetition >= eeg.shape[1]:
        raise IndexError(f"repetition must be in [0, {eeg.shape[1] - 1}].")

    sample = eeg[image_index, repetition]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image_index", "repetition", "channel_index", "channel_name", "time_index", "time", "value"]
        )

        for channel_index in range(sample.shape[0]):
            channel_name = ch_names[channel_index] if channel_index < len(ch_names) else ""
            for time_index in range(sample.shape[1]):
                time_value = float(times[time_index]) if time_index < len(times) else ""
                writer.writerow(
                    [
                        image_index,
                        repetition,
                        channel_index,
                        channel_name,
                        time_index,
                        time_value,
                        float(sample[channel_index, time_index]),
                    ]
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a preprocessed EEG `.npy` file into readable JSON/CSV files."
    )
    parser.add_argument("input_path", type=Path, help="Path to `preprocessed_eeg_training.npy`.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Where to write the JSON summary. Defaults next to the input file.",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=None,
        help="Where to write the sample CSV. Defaults next to the input file.",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Image index to export to CSV. Default: 0.",
    )
    parser.add_argument(
        "--repetition",
        type=int,
        default=0,
        help="Repetition index to export to CSV. Default: 0.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only the JSON summary and skip the sample CSV.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input_path.resolve()

    eeg_bundle = _load_preprocessed_eeg(input_path)
    eeg = np.asarray(eeg_bundle["preprocessed_eeg_data"])
    ch_names = [str(name) for name in np.asarray(eeg_bundle["ch_names"]).tolist()]
    times = np.asarray(eeg_bundle["times"])

    if eeg.ndim != 4:
        raise ValueError(
            f"Expected EEG data with 4 dimensions `(images, repetitions, channels, timepoints)`, got {eeg.shape}."
        )

    summary_output = args.summary_output.resolve() if args.summary_output else _default_summary_path(input_path)
    _write_summary(summary_output, input_path, eeg, ch_names, times)
    print(f"Wrote JSON summary to: {summary_output}")

    if not args.summary_only:
        sample_output = (
            args.sample_output.resolve()
            if args.sample_output
            else _default_sample_path(input_path, args.image_index, args.repetition)
        )
        _write_sample_csv(
            sample_output,
            eeg,
            ch_names,
            times,
            image_index=args.image_index,
            repetition=args.repetition,
        )
        print(
            "Wrote sample CSV to: "
            f"{sample_output} (image_index={args.image_index}, repetition={args.repetition})"
        )


if __name__ == "__main__":
    main()
