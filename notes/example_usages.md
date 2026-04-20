# Example Usages

This file documents active, tracked code paths for the current DINO-based pipeline.

## Minimum Runbook

1) Extract DINO targets (full + PCA-128 with standardization):
```bash
python scripts/dino_extract_image_embeds.py \
  --dataset-root datasets \
  --output-root latents \
  --embedding-type both \
  --full-dir-name img_dino_full \
  --pca-dir-name img_dino_pca_128 \
  --n-components 128 \
  --standardize-pca \
  --pca-scope train \
  --dino-repo-root dino-sae \
  --sae-checkpoint dino-sae/ema_model_step_470000.pt \
  --image-size 256 \
  --class-subset all \
  --split-seed 0 \
  --device cuda
```

2) Train + run DINO reconstruction eval + SSIM/LPIPS baseline eval:
```bash
bash scripts/run_eeg_encoder_dino_experiment.sh \
  --config configs/eeg_encoder_dino.yaml \
  --eval --max-samples 16 --grid-images 8 \
  --baseline --output-mode zero_one --image-size 256
```

3) (Optional) Reconstruct one PCA latent directly for debugging:
```bash
python scripts/dino_latent_reconstruct.py \
  --latent-path latents/img_dino_pca_128/000000.pt \
  --pca-params-path latents/img_dino_pca_128/pca_128.pt \
  --latent-shape 1024 16 16 \
  --output-path outputs/dino_recon_debug.png \
  --dino-repo-root dino-sae \
  --sae-checkpoint dino-sae/ema_model_step_470000.pt \
  --output-mode zero_one \
  --device cuda
```

## Config

`configs/eeg_encoder_dino.yaml`  
Default config for EEG encoder training against DINO PCA latents.
```bash
python scripts/train_eeg_encoder_dino.py --config configs/eeg_encoder_dino.yaml
```

## Scripts

`scripts/dino_extract_image_embeds.py`  
Extract DINO-SAE full latent grids and optionally PCA latents (with optional PCA z-score standardization).
```bash
python scripts/dino_extract_image_embeds.py \
  --dataset-root datasets \
  --output-root latents \
  --embedding-type both \
  --full-dir-name img_dino_full \
  --pca-dir-name img_dino_pca_128 \
  --n-components 128 \
  --standardize-pca \
  --pca-scope train \
  --dino-repo-root dino-sae \
  --sae-checkpoint dino-sae/ema_model_step_470000.pt \
  --image-size 256 \
  --class-subset all \
  --split-seed 0 \
  --device cuda
```

`scripts/dino_latent_reconstruct.py`  
Reconstruct an image from either full DINO latent or PCA latent (+ inverse standardization + inverse PCA).
```bash
python scripts/dino_latent_reconstruct.py \
  --latent-path latents/img_dino_pca_128/000000.pt \
  --pca-params-path latents/img_dino_pca_128/pca_128.pt \
  --latent-shape 1024 16 16 \
  --output-path outputs/dino_recon.png \
  --dino-repo-root dino-sae \
  --sae-checkpoint dino-sae/ema_model_step_470000.pt \
  --output-mode zero_one \
  --device cuda
```

`scripts/train_eeg_encoder_dino.py`  
CLI wrapper for EEG encoder training.
```bash
python scripts/train_eeg_encoder_dino.py \
  --config configs/eeg_encoder_dino.yaml \
  --dataset-root datasets \
  --latent-root latents/img_dino_pca_128 \
  --output-dim 128 \
  --epochs 20 \
  --output-dir outputs/eeg_encoder_dino \
  --device cuda
```

`scripts/run_eeg_encoder_dino_experiment.sh`  
Runner for train -> reconstruction eval -> SSIM/LPIPS baseline eval.
```bash
bash scripts/run_eeg_encoder_dino_experiment.sh \
  --config configs/eeg_encoder_dino.yaml \
  --eval --max-samples 16 --grid-images 8 \
  --baseline --output-mode zero_one --image-size 256
```

`scripts/pca_target_stats.py`  
Compute summary stats over latent targets for train/valid split.
```bash
python scripts/pca_target_stats.py \
  --dataset-root datasets \
  --latent-root latents/img_dino_pca_128 \
  --split-seed 0 \
  --output-path outputs/pca_target_stats_dino.json
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
    class_indices=[0, 1, 2],
    latent_root="latents/img_dino_pca_128",
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

## Source: Training

`src/training/train_eeg_encoder.py`  
Shared training loop used by the DINO training wrapper.
```python
from src.training import load_eeg_encoder_config, train_eeg_encoder

cfg = load_eeg_encoder_config("configs/eeg_encoder_dino.yaml")
train_eeg_encoder(cfg)
```

## Source: Evaluation

`src/evaluation/eval_eeg_encoder_dino.py`  
Runs EEG encoder on test split, inverse PCA, DINO decode, and saves reconstructions + grid.
```bash
python src/evaluation/eval_eeg_encoder_dino.py \
  --checkpoint-path outputs/eeg_encoder_dino/run_YYYYMMDD_HHMMSS/eeg_encoder_*.pt \
  --dataset-root datasets \
  --latent-root latents/img_dino_pca_128 \
  --pca-params-path latents/img_dino_pca_128/pca_128.pt \
  --latent-shape 1024 16 16 \
  --dino-repo-root dino-sae \
  --sae-checkpoint dino-sae/ema_model_step_470000.pt \
  --output-mode zero_one \
  --batch-size 4 \
  --max-samples 16 \
  --grid-images 8 \
  --device cuda \
  --output-dir outputs/eeg_encoder_dino/eval
```

`src/evaluation/eval_eeg_with_mean_baselines_dino.py`  
Computes SSIM/LPIPS for model predictions vs global/class mean-image baselines.
```bash
python src/evaluation/eval_eeg_with_mean_baselines_dino.py \
  --checkpoint-path outputs/eeg_encoder_dino/run_YYYYMMDD_HHMMSS/eeg_encoder_*.pt \
  --dataset-root datasets \
  --latent-root latents/img_dino_pca_128 \
  --pca-params-path latents/img_dino_pca_128/pca_128.pt \
  --latent-shape 1024 16 16 \
  --dino-repo-root dino-sae \
  --sae-checkpoint dino-sae/ema_model_step_470000.pt \
  --output-mode zero_one \
  --image-size 256 \
  --batch-size 4 \
  --max-samples 16 \
  --device cuda \
  --lpips-net alex \
  --output-dir outputs/eeg_encoder_dino/eval \
  --metrics-name eeg_vs_baselines_metrics.json
```

`src/evaluation/eeg_eval_core.py`  
Shared checkpoint/model/loader/eval utility functions used by evaluation scripts.

## Tests and Utilities

`tests/print_eeg_latent_dataset_samples.py`  
```bash
EEG_DATASET_ROOT=datasets EEG_LATENT_ROOT=latents/img_dino_pca_128 EEG_SPLIT=test python tests/print_eeg_latent_dataset_samples.py
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
Legacy image-polish utility (not part of the DINO core training/eval pipeline).
