#!/usr/bin/env python3
"""Visualize matching original and additionally processed THINGS-EEG2 samples.

The repository stores EEG as [image, repetition, channel, time].  Averaging and
the project's extra signal transforms are performed at runtime; this utility
does the same without changing the source arrays.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import EEGLabelDataset, build_eeg_transform, resolve_eeg_time_window


DISPLAY_CHANNELS = (
    "O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
)


@dataclass(frozen=True)
class EEGSample:
    """One selected EEG waveform and the metadata needed to validate it."""

    data: np.ndarray
    times_s: np.ndarray
    channels: tuple[str, ...]
    dataset_name: str
    subject: str
    image_index: int
    image_name: str
    repetition: int | None
    sampling_rate_hz: float


def normalize_subject(value: str) -> str:
    """Accept either ``1`` or ``sub-1`` and return the repository form."""
    value = str(value)
    return value if value.startswith("sub-") else f"sub-{value}"


def load_payload(dataset_root: Path, subject: str) -> dict[str, Any]:
    """Load mapping metadata not retained as attributes by the dataset class."""
    path = dataset_root / "THINGS_EEG_2" / subject / "preprocessed_eeg_training.npy"
    if not path.exists():
        raise FileNotFoundError(f"EEG file not found: {path}")
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object and payload.ndim == 0:
        payload = payload.item()
    return payload if isinstance(payload, dict) else {}


def load_dataset(dataset_root: Path, subject: str) -> tuple[EEGLabelDataset, dict[str, Any]]:
    """Load a subject through the project's existing EEG dataset implementation."""
    dataset = EEGLabelDataset(
        dataset_root=str(dataset_root), subject=subject, split="train", transform=None
    )
    return dataset, load_payload(dataset_root, subject)


def resolve_local_image_index(
    dataset: EEGLabelDataset, payload: dict[str, Any], requested_image_index: int
) -> int:
    """Map a global THINGS image index into a full or compact EEG dataset."""
    original_indices = payload.get("original_image_indices")
    if original_indices is None:
        if not 0 <= requested_image_index < dataset.num_images:
            raise IndexError(f"Image index must be in [0, {dataset.num_images - 1}].")
        return requested_image_index
    matches = np.flatnonzero(np.asarray(original_indices, dtype=np.int64) == requested_image_index)
    if matches.size != 1:
        raise ValueError(
            f"Global image index {requested_image_index} occurs {matches.size} times in compact dataset."
        )
    return int(matches[0])


def sampling_rate(times_s: np.ndarray) -> float:
    """Validate the stored uniform time vector and return its sampling rate."""
    intervals = np.diff(np.asarray(times_s, dtype=np.float64))
    if intervals.size == 0 or not np.allclose(intervals, intervals[0], rtol=1e-5, atol=1e-9):
        raise ValueError("EEG timestamps are not uniformly sampled.")
    return float(1.0 / intervals[0])


def reorder_channels(data: np.ndarray, stored_names: Sequence[str]) -> np.ndarray:
    """Return EEG channels in the canonical 17-channel posterior ordering."""
    stored = [str(name) for name in stored_names]
    if len(stored) != len(set(stored)):
        raise ValueError("Stored EEG channel names contain duplicates.")
    missing = [name for name in DISPLAY_CHANNELS if name not in stored]
    if missing:
        raise ValueError(f"Dataset is missing required channels: {', '.join(missing)}")
    return np.asarray(data, dtype=np.float32)[[stored.index(name) for name in DISPLAY_CHANNELS]].copy()


def select_original_sample(
    dataset: EEGLabelDataset,
    payload: dict[str, Any],
    subject: str,
    image_index: int,
    repetition: int,
) -> EEGSample:
    """Select one unmodified repetition by global image index."""
    local_index = resolve_local_image_index(dataset, payload, image_index)
    if not 0 <= repetition < dataset.repetitions:
        raise IndexError(f"Repetition must be in [0, {dataset.repetitions - 1}].")
    names = dataset.ch_names or DISPLAY_CHANNELS
    image_name = str(dataset.train_img_files[local_index])
    return EEGSample(
        data=reorder_channels(dataset.eeg[local_index, repetition], names),
        times_s=np.asarray(dataset.times, dtype=np.float64).copy(),
        channels=DISPLAY_CHANNELS,
        dataset_name="original THINGS-EEG2 preprocessed",
        subject=subject,
        image_index=image_index,
        image_name=image_name,
        repetition=repetition,
        sampling_rate_hz=sampling_rate(dataset.times),
    )


def select_processed_sample(
    dataset: EEGLabelDataset,
    payload: dict[str, Any],
    subject: str,
    image_index: int,
    repetition: int,
    average_repetitions: bool,
    cutoff_hz: float | None,
    window_pre_ms: float | None,
    window_post_ms: float | None,
) -> EEGSample:
    """Select the matching trial or average, then apply non-mutating runtime processing."""
    local_index = resolve_local_image_index(dataset, payload, image_index)
    names = dataset.ch_names or DISPLAY_CHANNELS
    if average_repetitions:
        source = np.asarray(dataset.eeg[local_index], dtype=np.float32).mean(axis=0)
        processed_repetition = None
        dataset_name = "processed (mean of all repetitions)"
    else:
        if not 0 <= repetition < dataset.repetitions:
            raise IndexError(f"Repetition must be in [0, {dataset.repetitions - 1}].")
        source = np.asarray(dataset.eeg[local_index, repetition], dtype=np.float32)
        processed_repetition = repetition
        dataset_name = "processed repetition"

    times = np.asarray(dataset.times, dtype=np.float64)
    transform_kwargs: dict[str, Any] = {}
    if cutoff_hz is not None:
        transform_kwargs.update(lowpass_cutoff_hz=cutoff_hz, sampling_rate_hz=sampling_rate(times))
    if window_pre_ms is not None or window_post_ms is not None:
        window = resolve_eeg_time_window(times, window_pre_ms, window_post_ms)
        if window is None:
            raise RuntimeError("Expected a resolved EEG window.")
        transform_kwargs.update(crop_start_idx=window["start_idx"], crop_end_idx=window["end_idx"])
        times = times[window["start_idx"] : window["end_idx"] + 1]
    processed = build_eeg_transform(
        normalize_mode="none", to_tensor=False, **transform_kwargs
    )(source)
    return EEGSample(
        data=reorder_channels(processed, names),
        times_s=times.copy(),
        channels=DISPLAY_CHANNELS,
        dataset_name=dataset_name,
        subject=subject,
        image_index=image_index,
        image_name=str(dataset.train_img_files[local_index]),
        repetition=processed_repetition,
        sampling_rate_hz=sampling_rate(dataset.times),
    )


def validate_correspondence(original: EEGSample, processed: EEGSample) -> None:
    """Reject comparisons that do not identify the same subject and stimulus."""
    if (original.subject, original.image_index, original.image_name) != (
        processed.subject, processed.image_index, processed.image_name
    ):
        raise ValueError("Original and processed samples do not identify the same stimulus.")
    if processed.repetition is not None and original.repetition != processed.repetition:
        raise ValueError("Original and processed samples use different repetitions.")


def print_metadata(sample: EEGSample) -> None:
    """Print useful identifying and numerical metadata for a plotted sample."""
    repetition = "none (averaged)" if sample.repetition is None else str(sample.repetition)
    print(
        f"dataset={sample.dataset_name} | subject={sample.subject} | "
        f"condition/image={sample.image_index} ({sample.image_name}) | repetition={repetition} | "
        f"shape={sample.data.shape} | sampling_frequency={sample.sampling_rate_hz:g} Hz | "
        f"voltage_range=[{sample.data.min():.6g}, {sample.data.max():.6g}]"
    )


def stacked_limits(samples: Sequence[EEGSample]) -> tuple[float, float]:
    """Calculate one channel amplitude range shared by stacked panels."""
    peak = max(float(np.max(np.abs(sample.data))) for sample in samples)
    return -peak, peak


def plot_stacked(samples: Sequence[EEGSample]) -> plt.Figure:
    """Plot all channels with vertical offsets and identical scaling across panels."""
    low, high = stacked_limits(samples)
    span = max(high - low, np.finfo(float).eps)
    offsets = np.arange(len(DISPLAY_CHANNELS) - 1, -1, -1) * span * 1.25
    fig, axes = plt.subplots(1, len(samples), figsize=(8 * len(samples), 10), squeeze=False)
    for ax, sample in zip(axes[0], samples):
        for index, offset in enumerate(offsets):
            ax.plot(sample.times_s * 1000, sample.data[index] + offset, linewidth=0.9)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(offsets, DISPLAY_CHANNELS)
        ax.set_xlabel("Time relative to stimulus onset (ms)")
        ax.set_title(sample.dataset_name)
        ax.grid(axis="x", alpha=0.2)
    axes[0, 0].set_ylabel("Channel (vertically offset)")
    fig.suptitle(f"{samples[0].subject}, image {samples[0].image_index}: {samples[0].image_name}")
    fig.tight_layout()
    return fig


def compute_spectrum(
    waveform: np.ndarray, sampling_rate_hz: float, scale: str = "linear"
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a mean-removed, Hann-windowed, one-sided EEG spectrum.

    Linear output is a one-sided FFT amplitude spectrum. Dividing by the Hann
    window sum corrects its coherent gain, and interior rFFT bins are doubled
    to account for the omitted negative-frequency half. ``db`` returns power
    in dB relative to one squared input unit. No zero-padding is used, so the
    frequency spacing reflects the true record duration.
    """
    values = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("At least two samples are required to compute a spectrum.")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive.")
    if scale not in {"linear", "db"}:
        raise ValueError("Spectrum scale must be 'linear' or 'db'.")

    centered = values - values.mean()
    window = np.hanning(values.size)
    spectrum = np.fft.rfft(centered * window)
    amplitude = np.abs(spectrum) / window.sum()
    if values.size % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / sampling_rate_hz)
    if scale == "db":
        floor = np.finfo(np.float64).tiny
        amplitude = 10.0 * np.log10(np.maximum(amplitude**2, floor))
    return frequencies, amplitude


def _plot_spectra_on_axis(
    ax: plt.Axes,
    original: EEGSample,
    processed: EEGSample,
    channel: str,
    scale: str,
    freq_max: float,
) -> None:
    """Plot corresponding original and processed channel spectra on one axis."""
    if channel not in DISPLAY_CHANNELS:
        raise ValueError(f"Unknown channel: {channel}")
    index = DISPLAY_CHANNELS.index(channel)
    original_freq, original_spectrum = compute_spectrum(
        original.data[index], original.sampling_rate_hz, scale
    )
    processed_freq, processed_spectrum = compute_spectrum(
        processed.data[index], processed.sampling_rate_hz, scale
    )
    ax.plot(original_freq, original_spectrum, label="Original", linewidth=1.2)
    ax.plot(processed_freq, processed_spectrum, label="Processed", linewidth=1.2)
    ax.set_xlim(0, freq_max)
    ax.set_ylabel("Power (dB re 1 unit²)" if scale == "db" else "FFT amplitude")
    ax.set_title(channel)
    ax.grid(alpha=0.2)
    ax.legend()


def plot_fft_comparison(
    original: EEGSample,
    processed: EEGSample,
    selected_channels: Sequence[str],
    scale: str,
    freq_max: float,
) -> plt.Figure:
    """Overlay original and processed spectra for each selected channel."""
    if not selected_channels:
        raise ValueError("Select at least one channel.")
    fig, axes = plt.subplots(
        len(selected_channels), 1, figsize=(11, 3 * len(selected_channels)), squeeze=False
    )
    for ax, channel in zip(axes[:, 0], selected_channels):
        _plot_spectra_on_axis(ax, original, processed, channel, scale, freq_max)
    axes[-1, 0].set_xlabel("Frequency (Hz)")
    fig.suptitle(f"EEG spectra — {original.subject}, image {original.image_index}: {original.image_name}")
    fig.tight_layout()
    return fig


def plot_full_comparison(
    original: EEGSample,
    processed: EEGSample,
    selected_channels: Sequence[str],
    scale: str,
    freq_max: float,
) -> plt.Figure:
    """Plot time-domain overlays with each channel's spectrum directly below."""
    if not selected_channels:
        raise ValueError("Select at least one channel.")
    unknown = [name for name in selected_channels if name not in DISPLAY_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown channel(s): {', '.join(unknown)}")
    fig, axes = plt.subplots(
        len(selected_channels) * 2,
        1,
        figsize=(11, 5.5 * len(selected_channels)),
        squeeze=False,
    )
    for row, channel in enumerate(selected_channels):
        index = DISPLAY_CHANNELS.index(channel)
        time_ax = axes[row * 2, 0]
        spectrum_ax = axes[row * 2 + 1, 0]
        time_ax.plot(original.times_s * 1000, original.data[index], label="Original", linewidth=1.2)
        time_ax.plot(processed.times_s * 1000, processed.data[index], label="Processed", linewidth=1.2)
        time_ax.axvline(0, color="black", linestyle="--", linewidth=1)
        time_ax.set_ylabel(f"{channel} amplitude")
        time_ax.set_xlabel("Time relative to stimulus onset (ms)")
        time_ax.grid(alpha=0.2)
        time_ax.legend()
        _plot_spectra_on_axis(spectrum_ax, original, processed, channel, scale, freq_max)
        spectrum_ax.set_xlabel("Frequency (Hz)")
    fig.suptitle(f"EEG time and frequency comparison — {original.subject}, image {original.image_index}")
    fig.tight_layout()
    return fig


def plot_stacked_with_spectra(
    samples: Sequence[EEGSample],
    original: EEGSample,
    processed: EEGSample,
    selected_channels: Sequence[str],
    scale: str,
    freq_max: float,
) -> plt.Figure:
    """Plot stacked waveform panels with selected-channel spectra underneath."""
    low, high = stacked_limits(samples)
    span = max(high - low, np.finfo(float).eps)
    offsets = np.arange(len(DISPLAY_CHANNELS) - 1, -1, -1) * span * 1.25
    columns = max(len(samples), len(selected_channels))
    fig = plt.figure(figsize=(8 * columns, 13))
    grid = fig.add_gridspec(2, columns, height_ratios=(3.2, 1))
    for column, sample in enumerate(samples):
        ax = fig.add_subplot(grid[0, column])
        for index, offset in enumerate(offsets):
            ax.plot(sample.times_s * 1000, sample.data[index] + offset, linewidth=0.9)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_yticks(offsets, DISPLAY_CHANNELS)
        ax.set_xlabel("Time relative to stimulus onset (ms)")
        ax.set_title(sample.dataset_name)
        ax.grid(axis="x", alpha=0.2)
        if column == 0:
            ax.set_ylabel("Channel (vertically offset)")
    for column in range(len(samples), columns):
        fig.add_subplot(grid[0, column]).axis("off")
    if len(selected_channels) == 1:
        ax = fig.add_subplot(grid[1, :])
        _plot_spectra_on_axis(
            ax, original, processed, selected_channels[0], scale, freq_max
        )
        ax.set_xlabel("Frequency (Hz)")
    else:
        for column, channel in enumerate(selected_channels):
            ax = fig.add_subplot(grid[1, column])
            _plot_spectra_on_axis(ax, original, processed, channel, scale, freq_max)
            ax.set_xlabel("Frequency (Hz)")
        for column in range(len(selected_channels), columns):
            fig.add_subplot(grid[1, column]).axis("off")
    fig.suptitle(f"{original.subject}, image {original.image_index}: {original.image_name}")
    fig.tight_layout()
    return fig


def plot_channel_comparison(
    original: EEGSample, processed: EEGSample, selected_channels: Sequence[str]
) -> plt.Figure:
    """Overlay original and processed traces for each requested channel."""
    unknown = [name for name in selected_channels if name not in DISPLAY_CHANNELS]
    if unknown:
        raise ValueError(f"Unknown channel(s): {', '.join(unknown)}")
    fig, axes = plt.subplots(len(selected_channels), 1, figsize=(11, 3 * len(selected_channels)), squeeze=False)
    for ax, channel in zip(axes[:, 0], selected_channels):
        index = DISPLAY_CHANNELS.index(channel)
        ax.plot(original.times_s * 1000, original.data[index], label="Original", linewidth=1.2)
        ax.plot(processed.times_s * 1000, processed.data[index], label="Processed", linewidth=1.2)
        combined = np.concatenate((original.data[index], processed.data[index]))
        margin = max(float(np.ptp(combined)) * 0.05, np.finfo(float).eps)
        ax.set_ylim(float(combined.min()) - margin, float(combined.max()) + margin)
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_ylabel(channel)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[-1, 0].set_xlabel("Time relative to stimulus onset (ms)")
    fig.suptitle(f"{original.subject}, image {original.image_index}: {original.image_name}")
    fig.tight_layout()
    return fig


def display_figure(figure: plt.Figure) -> None:
    """Show a figure interactively or open a rendered PNG when using Agg.

    Minimal Linux environments often install Matplotlib without Tk or Qt. In
    that case ``plt.show()`` only emits a warning, so retain a temporary image
    and hand it to the desktop's default image viewer.
    """
    backend = str(matplotlib.get_backend()).lower()
    if backend not in {"agg", "pdf", "ps", "svg", "cairo", "template"}:
        plt.show()
        return

    opener = shutil.which("xdg-open")
    if opener is None:
        raise RuntimeError(
            "Matplotlib is using a non-interactive backend and xdg-open is unavailable. "
            "Pass --save PATH to write the figure instead."
        )
    output_dir = Path(tempfile.mkdtemp(prefix="visualize-eeg-"))
    output_path = output_dir / "eeg_waveforms.png"
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    subprocess.Popen(
        [opener, str(output_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"Matplotlib backend '{backend}' is non-interactive; opened: {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument(
        "--processed-dataset-root", type=Path,
        help="Optional full/compact dataset root used for the processed sample (default: --dataset-root).",
    )
    parser.add_argument("--subject", required=True, help="Subject number or id, e.g. 1 or sub-1.")
    parser.add_argument("--condition", type=int, required=True, help="Zero-based global training image index.")
    parser.add_argument("--repetition", type=int, default=0, help="Zero-based repetition for the original sample.")
    parser.add_argument("--mode", choices=("stacked", "compare", "fft", "full"), default="stacked")
    parser.add_argument("--channels", nargs="+", default=["O1"], help="Channels used by --mode compare.")
    parser.add_argument(
        "--show-dataset", choices=("both", "original", "processed"), default="both",
        help="Panels shown in stacked mode.",
    )
    parser.add_argument(
        "--processed-repetitions", choices=("average", "selected"), default="selected",
        help="Average all processed repetitions or process the selected repetition.",
    )
    parser.add_argument("--lowpass-cutoff-hz", type=float, default=45.0, help="Cutoff in Hz; use 0 to disable.")
    parser.add_argument("--window-pre-ms", type=float, default=0.0)
    parser.add_argument("--window-post-ms", type=float, default=500.0)
    parser.add_argument("--full-window", action="store_true", help="Disable the default 0–500 ms crop.")
    parser.add_argument(
        "--freq-max", type=float, default=50.0,
        help="Maximum displayed frequency in Hz (default: 50).",
    )
    parser.add_argument(
        "--spectrum-scale", choices=("linear", "db"), default="linear",
        help="Plot one-sided FFT amplitude or power in dB.",
    )
    parser.add_argument("--save", type=Path, help="Save the figure instead of opening an interactive window.")
    return parser.parse_args()


def main() -> None:
    """Load matching signals, validate them, print metadata, and create the plot."""
    args = parse_args()
    subject = normalize_subject(args.subject)
    original_dataset, original_payload = load_dataset(args.dataset_root, subject)
    processed_root = args.processed_dataset_root or args.dataset_root
    processed_dataset, processed_payload = load_dataset(processed_root, subject)
    original = select_original_sample(
        original_dataset, original_payload, subject, args.condition, args.repetition
    )
    processed = select_processed_sample(
        processed_dataset,
        processed_payload,
        subject,
        args.condition,
        args.repetition,
        average_repetitions=args.processed_repetitions == "average",
        cutoff_hz=None if args.lowpass_cutoff_hz == 0 else args.lowpass_cutoff_hz,
        window_pre_ms=None if args.full_window else args.window_pre_ms,
        window_post_ms=None if args.full_window else args.window_post_ms,
    )
    validate_correspondence(original, processed)
    print_metadata(original)
    print_metadata(processed)

    nyquist_max = max(original.sampling_rate_hz, processed.sampling_rate_hz) / 2
    if args.freq_max <= 0 or args.freq_max > nyquist_max + 1e-6:
        raise ValueError(f"--freq-max must be in (0, {nyquist_max:g}] Hz.")
    if args.mode == "compare":
        figure = plot_channel_comparison(original, processed, args.channels)
    elif args.mode == "fft":
        figure = plot_fft_comparison(
            original, processed, args.channels, args.spectrum_scale, args.freq_max
        )
    elif args.mode == "full":
        figure = plot_full_comparison(
            original, processed, args.channels, args.spectrum_scale, args.freq_max
        )
    else:
        choices = {"original": [original], "processed": [processed], "both": [original, processed]}
        figure = plot_stacked_with_spectra(
            choices[args.show_dataset],
            original,
            processed,
            args.channels,
            args.spectrum_scale,
            args.freq_max,
        )
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=160, bbox_inches="tight")
        print(f"Saved figure: {args.save}")
        plt.close(figure)
    else:
        display_figure(figure)


if __name__ == "__main__":
    main()
