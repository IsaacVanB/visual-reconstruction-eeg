# Example Usages

This file documents active, tracked code paths for the VAE-based EEG encoder
pipeline and the EEG classifier pipeline.

Current encoder target types:

- `pca`: predict PCA-compressed SD-VAE latents. Default config: `configs/eeg_encoder.yaml`.
- `vae_lowres`: predict downsampled SD-VAE latent grids directly. Default config:
  `configs/eeg_encoder_vae_lowres.yaml`.

Class subset presets supported:

- EEG encoder training: `default100`, `default1000`, `all`
- VAE latent extraction: `default100`, `default800`, `all`
- EEG classifier training: `classifier20`, or explicit original THINGS zero-based
  `--class-indices`

Subject selection:

- `--subject sub-1`: train/evaluate one subject
- `--subjects sub-1 sub-2 sub-3`: train/evaluate explicit subjects
- `--subjects all` or `subjects: all`: use every available
  `datasets/THINGS_EEG_2/sub-*/preprocessed_eeg_training.npy`

Current EEG preprocessing defaults for encoder and classifier training are a
45 Hz low-pass filter, a 0–500 ms stimulus-relative crop, and channel-wise
z-score normalization using training-set statistics. Repeat handling is set
separately by `averaging_mode` (encoder) or `sample_mode` (classifier).

## Minimum Runbook

### Image Latents

Extract SD-VAE full latents and PCA-4 targets matching `configs/eeg_encoder.yaml`:

```bash
python scripts/vae_extract_image_embeds.py \
  --dataset-root datasets \
  --output-root latents \
  --embedding-type both \
  --full-dir-name img_full \
  --pca-dir-name img_pca_4_all \
  --n-components 4 \
  --standardize-pca \
  --pca-scope train \
  --vae-name stabilityai/sd-vae-ft-mse \
  --image-size 512 \
  --class-subset all \
  --split-seed 0 \
  --device cuda
```

For a larger PCA experiment, change both `--pca-dir-name`/`--n-components`
here and `image_latent_root`/`output_dim` in the encoder config or CLI.

The low-res VAE encoder uses the full latent directory (`latents/img_full`) and
does not need PCA files.

### PCA EEG Encoder

Train + evaluate the PCA encoder. The runner creates a timestamped run
directory under the config's `output_dir` and defaults to the best checkpoint:

```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --config configs/eeg_encoder.yaml \
  --eval --max-samples 16 --grid-images 8 \
  --baseline --image-size 512 --batch-size 4
```

Train only:

```bash
python scripts/train_eeg_encoder.py \
  --config configs/eeg_encoder.yaml \
  --dataset-root datasets \
  --image-latent-root latents/img_pca_4_all \
  --output-dim 4 \
  --subjects sub-1 sub-2 sub-3 \
  --subject-chunk-size 3 \
  --output-dir outputs/eeg_encoder/run_YYYYMMDD_HHMMSS \
  --device cuda
```

Evaluate one PCA checkpoint:

```bash
python src/evaluation/eval_eeg_encoder.py \
  --checkpoint-path outputs/eeg_encoder/run_YYYYMMDD_HHMMSS/eeg_encoder_best_YYYYMMDD_HHMMSS.pt \
  --dataset-root datasets \
  --latent-root latents/img_pca_4_all \
  --pca-params-path latents/img_pca_4_all/pca_4.pt \
  --metadata-path latents/img_full_metadata.json \
  --decode-latent-scaling auto \
  --vae-name stabilityai/sd-vae-ft-mse \
  --latent-shape 4 64 64 \
  --batch-size 4 \
  --max-samples 16 \
  --grid-images 8 \
  --device cuda \
  --output-dir outputs/eeg_encoder/run_YYYYMMDD_HHMMSS/eval
```

### Low-Res VAE EEG Encoder

Train + evaluate the direct low-res VAE latent encoder:

```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --config configs/eeg_encoder_vae_lowres.yaml \
  --eval --max-samples 16 --grid-images 8
```

The runner skips `eval_eeg_with_mean_baselines.py` for `vae_lowres` checkpoints
because that baseline script currently supports PCA targets only.

Train only:

```bash
python scripts/train_eeg_encoder.py \
  --config configs/eeg_encoder_vae_lowres.yaml \
  --dataset-root datasets \
  --image-latent-root latents/img_full \
  --target-type vae_lowres \
  --target-latent-size 8 \
  --output-dim 256 \
  --subjects sub-1 sub-2 sub-3 \
  --subject-chunk-size 3 \
  --output-dir outputs/eeg_encoder_vae_lowres/run_YYYYMMDD_HHMMSS \
  --device cuda
```

Evaluate one low-res VAE checkpoint:

```bash
python src/evaluation/eval_eeg_encoder.py \
  --checkpoint-path outputs/eeg_encoder_vae_lowres/run_YYYYMMDD_HHMMSS/eeg_encoder_best_YYYYMMDD_HHMMSS.pt \
  --dataset-root datasets \
  --latent-root latents/img_full \
  --metadata-path latents/img_full_metadata.json \
  --decode-latent-scaling auto \
  --vae-name stabilityai/sd-vae-ft-mse \
  --latent-shape 4 64 64 \
  --batch-size 4 \
  --max-samples 16 \
  --grid-images 8 \
  --device cuda \
  --output-dir outputs/eeg_encoder_vae_lowres/run_YYYYMMDD_HHMMSS/eval
```

### EEG Classifier

The default classifier config points at the compact dataset root
`datasets/classifier20` and currently uses 30 explicit original THINGS class
ids. The `classifier20` directory and artifact names are retained for backward
compatibility and should not be interpreted as the current class count. Create
the compact dataset from the full THINGS EEG files:

```bash
python scripts/extract_compact_eeg.py \
  --dataset-root datasets \
  --output-root datasets/classifier20/THINGS_EEG_2 \
  --subjects all \
  --class-indices 1055 9 178 853 435 1476 1461 59 431 977 114 596 1246 390 1314 1220 1509 1511 1487 1631 274 1408 645 705 736 955 1011 1036 1548 1583
```

Train the default CNN classifier on all compact subjects:

```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --dataset-root datasets/classifier20 \
  --subjects all \
  --subject-chunk-size 1 \
  --output-dir outputs/eeg_classifier \
  --device cuda
```

Train an EEGNet classifier variant:

```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --dataset-root datasets/classifier20 \
  --subjects all \
  --model-architecture eegnet \
  --batch-size 32 \
  --epochs 30 \
  --output-dir outputs/eeg_classifier \
  --device cuda
```

Per-epoch train/test evaluation is disabled by default for speed. Final test
evaluation always runs once after training. Use `--evaluate-train-each-epoch` or
`--evaluate-test-each-epoch` only when you need those diagnostics.

Evaluate a classifier checkpoint and save confusion matrix artifacts:

```bash
python scripts/eval_eeg_classifier.py \
  --checkpoint-path outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --device cuda
```

Classifier training writes each run to:

```text
outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/
```

Evaluation saves artifacts to:

```text
outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eval/
```

### Stable Diffusion Grid

Generate an img2img Stable Diffusion grid from classifier labels and EEG
encoder VAE reconstructions. The script supports both `pca` and `vae_lowres`
encoder checkpoints and restores the appropriate target processing from the
checkpoint. The example below uses PCA. Use the full dataset root so EEG image
indices line up with the full latent directory, even if the classifier was
trained from compact files.

```bash
python scripts/generate_eeg_sd_grid.py \
  --classifier-checkpoint outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --encoder-checkpoint outputs/eeg_encoder/run_YYYYMMDD_HHMMSS/eeg_encoder_best_YYYYMMDD_HHMMSS.pt \
  --dataset-root datasets \
  --latent-root latents/img_pca_4_all \
  --pca-params-path latents/img_pca_4_all/pca_4.pt \
  --metadata-path latents/img_full_metadata.json \
  --subjects all \
  --max-samples 20 \
  --device cuda \
  --fp16
```

## Configs

`configs/eeg_encoder.yaml`  
Default PCA encoder config. Current defaults use:

- `target_type: pca`
- `image_latent_root: latents/img_pca_4_all`
- `output_dim: 4`
- `subjects: [sub-1, sub-2, sub-3]`
- `averaging_mode: none`
- `eeg_lowpass_cutoff_hz: 45`
- `eeg_window_pre_ms: 0`, `eeg_window_post_ms: 500`
- `eeg_normalization: zscore`

```bash
python scripts/train_eeg_encoder.py --config configs/eeg_encoder.yaml
```

`configs/eeg_encoder_vae_lowres.yaml`  
Default low-res VAE latent encoder config. Current defaults use:

- `target_type: vae_lowres`
- `image_latent_root: latents/img_full`
- `target_latent_size: 8`
- `output_dim: 256`
- `averaging_mode: all`
- `eeg_lowpass_cutoff_hz: 45`
- `eeg_window_pre_ms: 0`, `eeg_window_post_ms: 500`
- `eeg_normalization: zscore`

```bash
python scripts/train_eeg_encoder.py --config configs/eeg_encoder_vae_lowres.yaml
```

`configs/eeg_classifier.yaml`  
Default classifier config for compact EEG data. Current defaults use:

- `dataset_root: datasets/classifier20`
- `subjects: all`
- 30 explicit `class_indices` (legacy paths/artifacts still say `classifier20`)
- `model_architecture: cnn`
- `subject_chunk_size: 1`
- `sample_mode: repetitions`
- `eeg_lowpass_cutoff_hz: 45`
- `eeg_window_pre_ms: 0`, `eeg_window_post_ms: 500`
- `eeg_normalization: zscore`

```bash
python scripts/train_eeg_classifier.py --config configs/eeg_classifier.yaml
```

## Scripts

`scripts/vae_extract_image_embeds.py`  
Extract SD-VAE full latents and optionally PCA latents.

`scripts/vae_latent_decode.py`  
Reconstruct an image from one PCA latent by inverse standardization, inverse
PCA, and SD-VAE decode.

```bash
python scripts/vae_latent_decode.py \
  --latent-path latents/img_pca_4_all/000000.pt \
  --pca-params-path latents/img_pca_4_all/pca_4.pt \
  --latent-shape 4 64 64 \
  --metadata-path latents/img_full_metadata.json \
  --output-path outputs/vae_recon_debug.png \
  --device cuda
```

`scripts/train_eeg_encoder.py`  
CLI wrapper for EEG encoder training.

`scripts/train_eeg_classifier.py`  
CLI wrapper for classifier training. Supports `cnn` and `eegnet`
architectures, explicit `--class-indices`, multi-subject training, and subject
chunking.

`scripts/eval_eeg_classifier.py`  
Wrapper for `src/evaluation/eval_eeg_classifier.py`.

`scripts/extract_compact_eeg.py`  
Extract selected THINGS EEG classes into smaller per-subject `.npy` files. Each
output file keeps `preprocessed_eeg_data` and adds compact-label metadata,
original labels, original image indices, and original class indices.

`scripts/run_eeg_encoder_experiment.sh`  
Runner for train -> reconstruction eval -> optional PCA baseline metrics.

`scripts/generate_eeg_sd_grid.py`  
Wrapper for `src/evaluation/generate_eeg_sd_grid.py`.

`scripts/analysis/visualize_eeg.py`

Compare matching original THINGS-EEG2 trials with the repository's additional
runtime processing. It uses the stored channel names and exact time vector,
validates subject/image/repetition correspondence, and supports stacked,
waveform-only, channel-overlay, FFT-only, and combined time/frequency views. By default the
processed signal uses the selected repetition without averaging, applies the
45 Hz low-pass filter, and crops to 0–500 ms. Visualization intentionally does
not apply training-set z-score normalization, so waveform amplitudes remain in
the loaded data's units.

Default stacked comparison with the selected channel's FFT below:

```bash
python scripts/analysis/visualize_eeg.py \
  --subject 1 \
  --condition 100 \
  --repetition 1 \
  --channels Oz
```

Original waveform-only side-by-side layout:

```bash
python scripts/analysis/visualize_eeg.py \
  --subject 1 \
  --condition 100 \
  --repetition 1 \
  --mode waveforms
```

Compare selected channels in the frequency domain and save the figure:

```bash
python scripts/analysis/visualize_eeg.py \
  --subject 1 \
  --condition 100 \
  --repetition 1 \
  --mode fft \
  --channels O1 Oz POz \
  --spectrum-scale db \
  --freq-max 50 \
  --save outputs/eeg_waveforms/sub1_condition100_rep1_fft.png
```

Show time-domain and FFT comparisons for all 17 posterior channels:

```bash
python scripts/analysis/visualize_eeg.py \
  --subject 1 \
  --condition 100 \
  --repetition 1 \
  --mode full \
  --channels O1 Oz O2 PO7 PO3 POz PO4 PO8 P7 P5 P3 P1 Pz P2 P4 P6 P8 \
  --save outputs/eeg_waveforms/sub1_condition100_rep1_all_channels.png
```

Use `--processed-repetitions average` to average all four trials explicitly,
`--full-window` to keep the stored -200–790 ms interval, or
`--lowpass-cutoff-hz 0` to disable the visualization filter. If Matplotlib has
no interactive GUI backend, the script renders a temporary PNG and opens it
with `xdg-open`; `--save` always writes directly to the requested path.

`scripts/pca_target_stats.py`  
Compute summary stats over PCA latent targets for train/valid split.

```bash
python scripts/pca_target_stats.py \
  --dataset-root datasets \
  --latent-root latents/img_pca_4_all \
  --class-subset all \
  --split-seed 0 \
  --output-path outputs/pca_target_stats.json
```

`scripts/eval_mean_image_baseline.py`  
Compute global/class mean-image baseline metrics on raw images.

```bash
python scripts/eval_mean_image_baseline.py \
  --dataset-root datasets \
  --split-seed 0 \
  --class-indices 0 2 4 6 8 \
  --mean-mode class \
  --image-size 512 \
  --batch-size 16 \
  --device cuda \
  --output-dir outputs/eeg_encoder/mean_baseline
```

## Source: Data

`src/data/datasets.py`  
Core dataset classes (`EEGImageDataset`, `EEGImageLatentDataset`, averaging
variants, `ImageDataset`).

```python
from src.data import EEGImageLatentDataset

ds = EEGImageLatentDataset(
    dataset_root="datasets",
    subject="sub-1",
    split="train",
    class_indices=[0, 2, 4],
    latent_root="latents/img_pca_4_all",
    split_seed=0,
)
eeg, latent, label = ds[0]
```

`src/data/dataloader.py` and `src/data/transforms.py`  
Helpers for loader construction and EEG/image transforms.

## Source: Models

`src/models/eeg_encoder.py`  
CNN encoder mapping EEG `[B,C,T]` to latent target vector `[B,K]`.

```python
import torch
from src.models import EEGEncoderCNN

model = EEGEncoderCNN(eeg_channels=17, eeg_timesteps=51, output_dim=4)
y = model(torch.randn(8, 17, 51))
```

`src/models/eeg_classifier.py`  
CNN and EEGNet classifiers mapping EEG `[B,C,T]` to class logits.

```python
import torch
from src.models import build_eeg_classifier_model

model = build_eeg_classifier_model(
    architecture="cnn",
    eeg_channels=17,
    eeg_timesteps=51,
    num_classes=30,
)
logits = model(torch.randn(8, 17, 51))
```

## Source: Training

`src/training/train_eeg_encoder.py`  
Shared training loop used by `scripts/train_eeg_encoder.py`.

```python
from src.training import load_eeg_encoder_config, train_eeg_encoder

cfg = load_eeg_encoder_config("configs/eeg_encoder.yaml")
train_eeg_encoder(cfg)
```

`src/training/train_eeg_classifier.py`  
Training loop used by `scripts/train_eeg_classifier.py`.

```python
from src.training import load_eeg_classifier_config, train_eeg_classifier

cfg = load_eeg_classifier_config("configs/eeg_classifier.yaml")
train_eeg_classifier(cfg)
```

## Source: Evaluation

`src/evaluation/eval_eeg_encoder.py`  
Runs an EEG encoder on the test split, decodes predicted latents, and saves
individual reconstructions plus `recon_grid.png`. Supports both `pca` and
`vae_lowres` checkpoints.

`src/evaluation/eval_eeg_with_mean_baselines.py`  
Computes SSIM/LPIPS for PCA model predictions vs global/class mean-image
baselines.

`src/evaluation/generate_eeg_sd_grid.py`  
Generates classifier-label-conditioned Stable Diffusion img2img grids and
summary metrics from classifier plus PCA or low-resolution VAE encoder
checkpoints.

`src/evaluation/eeg_eval_core.py`  
Shared checkpoint/model/loader/eval utility functions used by evaluation
scripts.

`src/evaluation/eval_eeg_classifier.py`  
Loads a classifier checkpoint, restores saved preprocessing stats/config, and
saves confusion matrix outputs.

## Colab

`notes/colab_eeg_classifier_training.md`  
Recommended Colab workflow for classifier training. Use Colab as a wrapper
around the repo scripts instead of copying training code into notebook cells.

## Tests

```bash
pytest
```

## Experiment and Validation Utilities

`scripts/validation/print_eeg_latent_dataset_samples.py`

```bash
EEG_DATASET_ROOT=datasets EEG_LATENT_ROOT=latents/img_pca_4_all EEG_SPLIT=test python scripts/validation/print_eeg_latent_dataset_samples.py
```

`scripts/experiments/stable_diffusion_img2img.py`
Stable Diffusion img2img utility for prompt and inference-parameter experiments.
