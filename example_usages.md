# Example Usages

This file documents each tracked file in the repo and how to use it.

## Table of Contents

- [Root Files](#root-files)
- [Config Files](#config-files)
- [Scripts](#scripts)
- [Source: Data](#source-data)
- [Source: Models](#source-models)
- [Source: Training](#source-training)
- [Source: Evaluation](#source-evaluation)
- [Tests and Debug Utilities](#tests-and-debug-utilities)
- [Notes](#notes)
- [Notebooks](#notebooks)
- [Figures](#figures)

## Root Files

`README.md`  
Project overview and setup instructions.
Example usage:
```bash
cat README.md
```

`LICENSE`  
Repository license text.
Example usage:
```bash
cat LICENSE
```

`requirements.txt`  
Python dependencies for training/evaluation scripts.
Example usage:
```bash
pip install -r requirements.txt
```

## Config Files

`configs/dummy_vae.yaml`  
Default config for dummy VAE training.
Example usage:
```bash
python scripts/train_dummy_vae.py --config configs/dummy_vae.yaml
```

`configs/eeg_encoder.yaml`  
Default config for EEG encoder training against PCA latents.
Example usage:
```bash
python scripts/train_eeg_encoder.py --config configs/eeg_encoder.yaml
```

## Scripts

`scripts/train_dummy_vae.py`  
CLI wrapper for `src/training/train_dummy_vae.py`.
Example usage (all CLI params):
```bash
python scripts/train_dummy_vae.py \
  --config configs/dummy_vae.yaml \
  --dataset-root datasets \
  --epochs 10 \
  --batch-size 32 \
  --num-workers 0 \
  --latent-dim 128 \
  --lr 0.001 \
  --kl-weight 0.001 \
  --image-size 128 \
  --split-seed 0 \
  --output-dir outputs/dummy_vae \
  --device cuda
```

`scripts/train_eeg_encoder.py`  
CLI wrapper for EEG encoder training.
Example usage (all CLI params):
```bash
python scripts/train_eeg_encoder.py \
  --config configs/eeg_encoder.yaml \
  --dataset-root datasets \
  --latent-root latents/img_pca \
  --subject sub-1 \
  --class-indices 0 2 4 6 8 \
  --split-seed 0 \
  --output-dim 128 \
  --temporal-filters 32 \
  --depth-multiplier 2 \
  --temporal-kernel1 51 \
  --temporal-kernel3 13 \
  --pool1 2 \
  --pool3 5 \
  --dropout 0.3 \
  --batch-size 16 \
  --num-workers 0 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --epochs 20 \
  --output-dir outputs/eeg_encoder \
  --device cuda
```

`scripts/extract_image_embeds.py`  
Unified embedding pipeline:
- extracts full SD-VAE latents (`img_full`)
- optionally converts them to PCA latents (`img_pca`)
- writes metadata + PCA params
Example usage (all CLI params):
```bash
python scripts/extract_image_embeds.py \
  --dataset-root datasets \
  --output-root latents \
  --embedding-type both \
  --vae-name stabilityai/sd-vae-ft-mse \
  --image-size 512 \
  --device cuda \
  --class-indices 0 2 4 6 8 \
  --full-dir-name img_full \
  --pca-dir-name img_pca \
  --n-components 128 \
  --pca-save-dtype float32 \
  --pca-params-path latents/img_pca/pca_128.pt
```
Optional PCA flag:
```bash
python scripts/extract_image_embeds.py --embedding-type pca --output-root latents --n-components 128 --no-explained-variance
```

`scripts/latent_decode.py`  
Decodes PCA-space latent vectors by inverse PCA + inverse scaling + SD VAE decode.
Example usage (all CLI params):
```bash
python scripts/latent_decode.py \
  --latent-path latents/img_pca/000090.pt \
  --pca-params-path latents/img_pca/pca_128.pt \
  --output-path outputs/latent_decode.png \
  --vae-name stabilityai/sd-vae-ft-mse \
  --latent-shape 4 64 64 \
  --metadata-path latents/img_full_metadata.json \
  --device cuda
```
If latent file stores a dict:
```bash
python scripts/latent_decode.py --latent-path some_pred.pt --latent-key prediction
```

`scripts/run_eeg_encoder_experiment.sh`  
End-to-end experiment runner: train -> evaluate -> mean-image baseline.
Example usage:
```bash
bash scripts/run_eeg_encoder_experiment.sh
```
Runner options:
```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --output-base outputs/eeg_encoder \
  --run-name my_run \
  --skip-eval \
  --skip-baseline
```
Forwarding train/eval/baseline args:
```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --config configs/eeg_encoder.yaml \
  --epochs 40 \
  --eval --max-samples 16 --grid-images 8 \
  --baseline --image-size 256 --batch-size 64
```

## Source: Data

`src/data/__init__.py`  
Exports dataset, dataloader, and transform builders.
Example usage:
```python
from src.data import EEGImageDataset, EEGImageLatentDataset, build_eeg_dataloader
```

`src/data/datasets.py`  
Defines:
- `EEGImageDataset`: EEG + paired image + label
- `EEGImageLatentDataset`: EEG + precomputed latent + label
- `ImageDataset`: image + label
Example usage:
```python
from src.data import EEGImageLatentDataset

ds = EEGImageLatentDataset(
    dataset_root="datasets",
    subject="sub-1",
    split="train",
    class_indices=[0, 2, 4],
    latent_root="latents/img_pca",
    split_seed=0,
)
eeg, latent, label = ds[0]
```

`src/data/dataloader.py`  
Builders for `DataLoader` objects.
Example usage:
```python
from src.data import build_eeg_dataloader

loader = build_eeg_dataloader(
    dataset_root="datasets",
    split="train",
    batch_size=32,
    split_seed=0,
)
```

`src/data/transforms.py`  
Custom transform utilities for EEG/image preprocessing.
Example usage:
```python
from src.data import build_eeg_transform, build_image_transform

eeg_tf = build_eeg_transform(normalize_per_sample=True, to_tensor=True)
img_tf = build_image_transform(image_size=(256, 256), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
```

## Source: Models

`src/models/__init__.py`  
Exports model classes.
Example usage:
```python
from src.models import ConvVAE, EEGEncoderCNN
```

`src/models/vae.py`  
Convolutional VAE used for pipeline prototyping.
Example usage:
```python
from src.models import ConvVAE

model = ConvVAE(image_size=(128, 128), latent_dim=128)
```

`src/models/eeg_encoder.py`  
CNN encoder for EEG input `[B, C, T]` with output embedding `[B, k]`.
Example usage:
```python
import torch
from src.models import EEGEncoderCNN

model = EEGEncoderCNN(eeg_channels=17, eeg_timesteps=100, output_dim=128)
x = torch.randn(8, 17, 100)
y = model(x)  # [8, 128]
```

## Source: Training

`src/training/__init__.py`  
Exports train/config loaders for VAE and EEG encoder.
Example usage:
```python
from src.training import load_eeg_encoder_config, train_eeg_encoder
```

`src/training/train_dummy_vae.py`  
Core dummy-VAE training loop, config loader, artifact saving.
Example usage:
```python
from src.training.train_dummy_vae import load_dummy_vae_config, train_dummy_vae

cfg = load_dummy_vae_config("configs/dummy_vae.yaml")
train_dummy_vae(cfg)
```

`src/training/train_eeg_encoder.py`  
Core EEG-encoder training loop against latent targets (MSE), with timestamped artifacts.
Example usage:
```python
from src.training.train_eeg_encoder import load_eeg_encoder_config, train_eeg_encoder

cfg = load_eeg_encoder_config("configs/eeg_encoder.yaml")
train_eeg_encoder(cfg)
```

## Source: Evaluation

`src/evaluation/eval_dummy_vae.py`  
Loads trained dummy VAE and writes reconstruction grid.
Example usage (all CLI params):
```bash
python src/evaluation/eval_dummy_vae.py \
  --checkpoint-path outputs/dummy_vae/dummy_vae.pt \
  --dataset-root datasets \
  --split valid \
  --batch-size 16 \
  --num-workers 0 \
  --image-size 64 \
  --split-seed 0 \
  --num-images 8 \
  --output-path outputs/dummy_vae/recon_grid.png
```

`src/evaluation/eval_eeg_encoder.py`  
Runs EEG encoder on test set, inverse-PCA + VAE decode, saves reconstructions and a recon grid.
Example usage (all CLI params):
```bash
python src/evaluation/eval_eeg_encoder.py \
  --checkpoint-path outputs/eeg_encoder/eeg_encoder.pt \
  --dataset-root datasets \
  --latent-root latents/img_pca \
  --subject sub-1 \
  --split-seed 0 \
  --class-indices 0 2 4 6 8 \
  --pca-params-path latents/img_pca/pca_128.pt \
  --metadata-path latents/img_full_metadata.json \
  --vae-name stabilityai/sd-vae-ft-mse \
  --latent-shape 4 64 64 \
  --batch-size 4 \
  --num-workers 0 \
  --max-samples 16 \
  --num-images 16 \
  --grid-images 8 \
  --device cuda \
  --output-dir outputs/decoded_eeg_img
```

`src/evaluation/eval_mean_image_baseline.py`  
Computes mean-image baseline and reports SSIM/LPIPS on test split.
Example usage (all CLI params):
```bash
python src/evaluation/eval_mean_image_baseline.py \
  --dataset-root datasets \
  --split-seed 0 \
  --class-indices 0 2 4 6 8 \
  --image-size 256 \
  --batch-size 32 \
  --num-workers 0 \
  --device cuda \
  --lpips-net alex \
  --output-dir outputs/eeg_encoder/mean_baseline \
  --mean-image-name mean_image.png \
  --metrics-name baseline_metrics.json
```

## Tests and Debug Utilities

`tests/print_eeg_latent_dataset_samples.py`  
Print sample shapes from `EEGImageLatentDataset` and validates expected length.
Example usage:
```bash
EEG_DATASET_ROOT=datasets EEG_LATENT_ROOT=latents/img_pca EEG_SPLIT=test EEG_CLASS_INDICES=0,2,4 python tests/print_eeg_latent_dataset_samples.py
```

`tests/inspect_eeg_image_pair.py`  
Deep inspection utility for one split image and repetition; can render/save selected image.
Example usage:
```bash
EEG_DATASET_ROOT=datasets EEG_SPLIT=train EEG_SPLIT_IMG_IDX=0 EEG_SAMPLE_REP=0 EEG_SAVE_FIG=outputs/inspect_pair.png python tests/inspect_eeg_image_pair.py
```

`tests/verify_eeg_image_pairing.py`  
Invariant checks for EEG-image alignment and split partitioning.
Example usage:
```bash
EEG_DATASET_ROOT=datasets EEG_SUBJECT=sub-1 EEG_SPLIT_SEED=0 python tests/verify_eeg_image_pairing.py
```

`tests/stable_diffusion_test.py`  
Standalone img2img “polish” script (supports single image or directory input).
Example usage (all CLI params):
```bash
python tests/stable_diffusion_test.py \
  --input_dir outputs/sd_vae_recon \
  --output_dir outputs/sd_polished \
  --model_id runwayml/stable-diffusion-v1-5 \
  --size 512 \
  --prompt "" \
  --negative_prompt "" \
  --strength 0.2 \
  --guidance_scale 5.0 \
  --steps 25 \
  --seed 0 \
  --fp16
```

## Notes

`notes/EEG_ENCODER_NOTES.md`  
Design checklist and change-safety guide for EEG encoder experiments.
Example usage:
```bash
cat notes/EEG_ENCODER_NOTES.md
```

`notes/VAE_architecture_notes.md`  
Design checklist for safely modifying VAE architecture/training.
Example usage:
```bash
cat notes/VAE_architecture_notes.md
```

`notes/repo_structure.txt`  
Planned/target repo layout and guidance.
Example usage:
```bash
cat notes/repo_structure.txt
```

## Notebooks

`notebooks/AE-ex1.ipynb`  
Autoencoder experiment notebook.
Example usage:
```bash
jupyter lab notebooks/AE-ex1.ipynb
```

`notebooks/VAE-ex1.ipynb`  
VAE experiment notebook.
Example usage:
```bash
jupyter lab notebooks/VAE-ex1.ipynb
```

`notebooks/g4g-vae-ex.ipynb`  
Additional VAE exploration notebook.
Example usage:
```bash
jupyter lab notebooks/g4g-vae-ex.ipynb
```

`notebooks/loading_data.ipynb`  
Notebook for THINGS-EEG data loading inspection.
Example usage:
```bash
jupyter lab notebooks/loading_data.ipynb
```

## Figures

`figures/model.jpg`  
Architecture diagram used in README.
Example usage:
```bash
xdg-open figures/model.jpg
```

`figures/model.png`  
Alternative format of architecture diagram.
Example usage:
```bash
xdg-open figures/model.png
```
