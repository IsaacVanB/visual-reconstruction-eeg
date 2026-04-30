# Example Usages

This file documents active, tracked code paths for the VAE-based EEG encoder
pipeline and the `classifier20` EEG classifier pipeline.

Class subset presets supported:
- EEG encoder training: `default100`, `default1000`, `all`
- VAE latent extraction: `default100`, `default800`, `all`
- EEG classifier training: `classifier20`

Classifier subject selection:
- `--subject sub-1`: train/evaluate one subject
- `--subjects sub-1 sub-2 sub-3`: train/evaluate explicit subjects
- `--subjects all` or `subjects: all`: use every available
  `datasets/THINGS_EEG_2/sub-*/preprocessed_eeg_training.npy`

## Minimum Runbook

### VAE EEG Encoder

1) Extract VAE targets (full + PCA-128 with standardization):
```bash
python scripts/vae_extract_image_embeds.py \
  --dataset-root datasets \
  --output-root latents \
  --embedding-type both \
  --full-dir-name img_full \
  --pca-dir-name img_pca_128 \
  --n-components 128 \
  --standardize-pca \
  --pca-scope train \
  --vae-name stabilityai/sd-vae-ft-mse \
  --image-size 512 \
  --class-subset all \
  --split-seed 0 \
  --device cuda
```

2) Train + run VAE reconstruction eval + SSIM/LPIPS baseline eval:
```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --config configs/eeg_encoder.yaml \
  --eval --max-samples 16 --grid-images 8 \
  --baseline --image-size 512
```

3) (Optional) Reconstruct one PCA latent directly for debugging:
```bash
python scripts/vae_latent_decode.py \
  --latent-path latents/img_pca_128/000000.pt \
  --pca-params-path latents/img_pca_128/pca_128.pt \
  --latent-shape 4 64 64 \
  --metadata-path latents/img_full_metadata.json \
  --output-path outputs/vae_recon_debug.png \
  --device cuda
```

### EEG Classifier

Train the notebook-compatible 20-class classifier on one subject:
```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --dataset-root datasets \
  --subject sub-1 \
  --output-dir outputs/eeg_classifier \
  --device cuda
```

Train on all available subjects:
```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --dataset-root datasets \
  --subjects all \
  --output-dir outputs/eeg_classifier \
  --device cuda
```

Classifier training writes each run to:
```text
outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/
```

Evaluate a classifier checkpoint and save confusion matrix artifacts:
```bash
python scripts/eval_eeg_classifier.py \
  --checkpoint-path outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --device cuda
```

Evaluation saves artifacts to:
```text
outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eval/
```

## Config

`configs/eeg_encoder.yaml`  
Default config for EEG encoder training against VAE PCA latents.
```bash
python scripts/train_eeg_encoder.py --config configs/eeg_encoder.yaml
```

`configs/eeg_classifier.yaml`  
Default config for classifier20 EEG classification.
```bash
python scripts/train_eeg_classifier.py --config configs/eeg_classifier.yaml
```

Useful overrides:
```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --subjects all \
  --batch-size 16 \
  --epochs 30 \
  --device cuda
```

## Scripts

`scripts/vae_extract_image_embeds.py`  
Extract SD-VAE full latents and optionally PCA latents (with optional PCA z-score standardization).
```bash
python scripts/vae_extract_image_embeds.py \
  --dataset-root datasets \
  --output-root latents \
  --embedding-type both \
  --full-dir-name img_full \
  --pca-dir-name img_pca_128 \
  --n-components 128 \
  --standardize-pca \
  --pca-scope train \
  --vae-name stabilityai/sd-vae-ft-mse \
  --image-size 512 \
  --class-subset all \
  --split-seed 0 \
  --device cuda
```

`scripts/vae_latent_decode.py`  
Reconstruct an image from PCA latent by inverse standardization + inverse PCA + VAE decode.
```bash
python scripts/vae_latent_decode.py \
  --latent-path latents/img_pca_128/000000.pt \
  --pca-params-path latents/img_pca_128/pca_128.pt \
  --latent-shape 4 64 64 \
  --metadata-path latents/img_full_metadata.json \
  --output-path outputs/vae_recon.png \
  --device cuda
```

`scripts/train_eeg_encoder.py`  
CLI wrapper for EEG encoder training.
```bash
python scripts/train_eeg_encoder.py \
  --config configs/eeg_encoder.yaml \
  --dataset-root datasets \
  --latent-root latents/img_pca_128 \
  --output-dim 128 \
  --epochs 20 \
  --output-dir outputs/eeg_encoder \
  --device cuda
```

`scripts/train_eeg_classifier.py`  
CLI wrapper for classifier20 EEG classifier training. Uses the same EEG
normalization/windowing style as the main `src` pipeline and streams subjects
one at a time for multi-subject runs.
```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --dataset-root datasets \
  --subjects all \
  --batch-size 16 \
  --epochs 30 \
  --output-dir outputs/eeg_classifier \
  --device cuda
```

Single-subject run:
```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --subject sub-1 \
  --device cuda
```

Explicit multi-subject run:
```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --subjects sub-1 sub-2 sub-3 \
  --device cuda
```

`scripts/eval_eeg_classifier.py`  
Evaluate a classifier20 checkpoint and write confusion matrix artifacts.
```bash
python scripts/eval_eeg_classifier.py \
  --checkpoint-path outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --device cuda
```

Optional quick evaluation:
```bash
python scripts/eval_eeg_classifier.py \
  --checkpoint-path outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --max-samples 16 \
  --device cuda
```

`scripts/run_eeg_encoder_experiment.sh`  
Runner for train -> VAE reconstruction eval -> SSIM/LPIPS baseline eval.
```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --config configs/eeg_encoder.yaml \
  --eval --max-samples 16 --grid-images 8 \
  --baseline --image-size 512
```

`scripts/pca_target_stats.py`  
Compute summary stats over latent targets for train/valid split.
```bash
python scripts/pca_target_stats.py \
  --dataset-root datasets \
  --latent-root latents/img_pca_128 \
  --class-subset all \
  --split-seed 0 \
  --output-path outputs/pca_target_stats_vae.json
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
Core dataset classes (`EEGImageDataset`, `EEGImageLatentDataset`, averaging variants, `ImageDataset`).
```python
from src.data import EEGImageLatentDataset

ds = EEGImageLatentDataset(
    dataset_root="datasets",
    subject="sub-1",
    split="train",
    class_indices=[0, 2, 4],
    latent_root="latents/img_pca_128",
    split_seed=0,
)
eeg, latent, label = ds[0]
```

`src/data/dataloader.py` and `src/data/transforms.py`  
Helpers for loader construction and EEG/image transforms.

## Source: Models

`src/models/eeg_encoder.py`  
CNN encoder mapping EEG `[B,C,T]` -> latent target vector `[B,K]`.
```python
import torch
from src.models import EEGEncoderCNN

model = EEGEncoderCNN(eeg_channels=17, eeg_timesteps=100, output_dim=128)
y = model(torch.randn(8, 17, 100))
```

`src/models/eeg_classifier.py`  
Notebook-style 2D CNN classifier for EEG `[B,C,T]` -> 20-class logits.
```python
import torch
from src.models import EEGClassifier20CNN

model = EEGClassifier20CNN(eeg_channels=17, eeg_timesteps=51, num_classes=20)
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
Runs EEG encoder on test split, inverse PCA, SD-VAE decode, and saves reconstructions + grid.
```bash
python src/evaluation/eval_eeg_encoder.py \
  --checkpoint-path outputs/eeg_encoder/run_YYYYMMDD_HHMMSS/eeg_encoder_*.pt \
  --dataset-root datasets \
  --latent-root latents/img_pca_128 \
  --pca-params-path latents/img_pca_128/pca_128.pt \
  --metadata-path latents/img_full_metadata.json \
  --decode-latent-scaling auto \
  --vae-name stabilityai/sd-vae-ft-mse \
  --latent-shape 4 64 64 \
  --batch-size 4 \
  --max-samples 16 \
  --grid-images 8 \
  --device cuda \
  --output-dir outputs/eeg_encoder/eval
```

`src/evaluation/eval_eeg_with_mean_baselines.py`  
Computes SSIM/LPIPS for model predictions vs global/class mean-image baselines.
```bash
python src/evaluation/eval_eeg_with_mean_baselines.py \
  --checkpoint-path outputs/eeg_encoder/run_YYYYMMDD_HHMMSS/eeg_encoder_*.pt \
  --dataset-root datasets \
  --latent-root latents/img_pca_128 \
  --pca-params-path latents/img_pca_128/pca_128.pt \
  --metadata-path latents/img_full_metadata.json \
  --decode-latent-scaling auto \
  --vae-name stabilityai/sd-vae-ft-mse \
  --latent-shape 4 64 64 \
  --image-size 512 \
  --batch-size 4 \
  --max-samples 16 \
  --device cuda \
  --lpips-net alex \
  --output-dir outputs/eeg_encoder/eval \
  --metrics-name eeg_vs_baselines_metrics.json
```

`src/evaluation/eeg_eval_core.py`  
Shared checkpoint/model/loader/eval utility functions used by evaluation scripts.

`src/evaluation/eval_eeg_classifier.py`  
Loads a classifier20 checkpoint, restores saved preprocessing stats/config, and
saves confusion matrix outputs.
```bash
python src/evaluation/eval_eeg_classifier.py \
  --checkpoint-path outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --device cuda
```

## Colab

`notes/colab_eeg_classifier_training.md`  
Recommended Colab workflow for classifier training. Use Colab as a wrapper around
the repo scripts instead of copying training code into notebook cells.

## Tests and Utilities

`tests/print_eeg_latent_dataset_samples.py`  
```bash
EEG_DATASET_ROOT=datasets EEG_LATENT_ROOT=latents/img_pca_128 EEG_SPLIT=test python tests/print_eeg_latent_dataset_samples.py
```

`tests/inspect_eeg_image_pair.py`  
```bash
EEG_DATASET_ROOT=datasets EEG_SPLIT=train EEG_SPLIT_IMG_IDX=0 EEG_SAMPLE_REP=0 python tests/inspect_eeg_image_pair.py
```

`tests/verify_eeg_image_pairing.py`  
```bash
EEG_DATASET_ROOT=datasets EEG_SUBJECT=sub-1 EEG_SPLIT_SEED=0 python tests/verify_eeg_image_pairing.py
```

`tests/stable_diffusion_test.py`  
Legacy image-polish utility (not required for core VAE train/eval pipeline).
