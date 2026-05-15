# Visual Reconstruction with EEG

Reconstructing stimulus images from EEG data, with an emphasis on shape, color, and image structure over semantic accuracy. The main pipeline trains an EEG encoder to predict downsampled Stable Diffusion VAE latents, then decodes those predictions back into images.

![Model architecture](figures/model.jpg)

## Results

Example reconstructions:

![EEG Stable Diffusion reconstruction grid](figures/eeg_sd_grid.png)

"Label only" images are reconstructed using Stable Diffusion text to image, with text coming from the EEG classifier.  
"Label + EEG image" images are reconstructed using Stable Diffusion image to image, with text coming from the EEG classifier and EEG image coming from the encoder.

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

5. Extract image targets for the EEG encoder. The low-res pipeline uses full
   SD-VAE latents from `latents/img_full`; the training dataset downsamples
   them to the configured low-resolution target size at runtime.

   ```bash
   python scripts/vae_extract_image_embeds.py \
     --dataset-root datasets \
     --output-root latents \
     --embedding-type full \
     --full-dir-name img_full \
     --vae-name stabilityai/sd-vae-ft-mse \
     --image-size 512 \
     --class-subset all \
     --split-seed 0 \
     --device cuda
   ```

   The default low-res config expects `image_latent_root: latents/img_full`,
   `target_latent_size: 8`, and `output_dim: 256`.

## Quick Training And Testing

Train the EEG encoder:

```bash
python scripts/train_eeg_encoder.py \
  --config configs/eeg_encoder_vae_lowres.yaml \
  --dataset-root datasets \
  --image-latent-root latents/img_full \
  --target-type vae_lowres \
  --target-latent-size 8 \
  --output-dim 256 \
  --output-dir outputs/eeg_encoder_vae_lowres/run_YYYYMMDD_HHMMSS \
  --device cuda
```

Evaluate an encoder checkpoint and save decoded reconstructions:

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

For a full train-then-evaluate run with a timestamped output directory:

```bash
bash scripts/run_eeg_encoder_experiment.sh \
  --config configs/eeg_encoder_vae_lowres.yaml \
  --eval --max-samples 16 --grid-images 8
```

`scripts/run_eeg_encoder_experiment.sh` skips the SSIM/LPIPS baseline step for
`vae_lowres` checkpoints because the baseline comparison script does not support
low-res VAE targets yet.

Train the EEG classifier:

```bash
python scripts/train_eeg_classifier.py \
  --config configs/eeg_classifier.yaml \
  --dataset-root datasets/classifier20 \
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

Training artifacts are written under `outputs/eeg_encoder_vae_lowres/` and `outputs/eeg_classifier/`. Evaluation writes decoded images, reconstruction grids, metrics, predictions, and confusion matrices under each run's `eval/` directory.
