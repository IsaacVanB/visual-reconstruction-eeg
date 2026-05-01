# Visual Reconstruction with EEG

Reconstructing stimulus images from EEG data, with an emphasis on shape, color, and image structure over semantic accuracy. The main pipeline trains an EEG encoder to predict PCA-compressed Stable Diffusion VAE latents, then decodes those predictions back into images.

![Model architecture](figures/model.jpg)

## Results

Example EEG-conditioned Stable Diffusion reconstructions:

![EEG Stable Diffusion reconstruction grid](figures/eeg_sd_grid.png)

## Setup

1. Create and activate a Python environment. We used Python 3.12.3.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. Download [THINGS EEG2](https://osf.io/3jk45/overview). We used the [preprocessed data](https://osf.io/anp5v/overview).

3. Download [THINGS images](https://osf.io/jum2f/files/rdxy2).

4. Arrange the data under `datasets/`:

   ```text
   datasets/
     THINGS_EEG_2/
       image_metadata.npy
       sub-1/
         preprocessed_eeg_training.npy
       sub-2/
         preprocessed_eeg_training.npy
       ...
     images_THINGS/
       object_images/
         <class_name>/
           <image files>
   ```

5. Extract image targets for the EEG encoder. This command writes full SD-VAE latents and standardized PCA latents to `latents/`.

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

   If you use a different PCA dimensionality, set `image_latent_root` and `output_dim` in `configs/eeg_encoder.yaml` to the same dimension.

## Quick Training And Testing

Train the EEG encoder:

```bash
python scripts/train_eeg_encoder.py \
  --config configs/eeg_encoder.yaml \
  --dataset-root datasets \
  --image-latent-root latents/img_pca_128 \
  --output-dim 128 \
  --output-dir outputs/eeg_encoder \
  --device cuda
```

Evaluate an encoder checkpoint and save decoded reconstructions:

```bash
python src/evaluation/eval_eeg_encoder.py \
  --checkpoint-path outputs/eeg_encoder/eeg_encoder_best_YYYYMMDD_HHMMSS.pt \
  --latent-root latents/img_pca_128 \
  --max-samples 16 \
  --grid-images 8 \
  --device cuda
```

For a full train-then-evaluate run with a timestamped output directory:

```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --config configs/eeg_encoder.yaml \
  --eval --max-samples 16 --grid-images 8 \
  --baseline --image-size 512
```

Train the 20-class EEG classifier:

```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --dataset-root datasets \
  --subjects all \
  --output-dir outputs/eeg_classifier \
  --device cuda
```

Evaluate a classifier checkpoint:

```bash
python scripts/eval_eeg_classifier.py \
  --checkpoint-path outputs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --device cuda
```

Training artifacts are written under `outputs/eeg_encoder/` and `outputs/eeg_classifier/`. Evaluation writes decoded images, reconstruction grids, metrics, predictions, and confusion matrices under each run's `eval/` directory.
