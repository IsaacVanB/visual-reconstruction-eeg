# Walking Through Training End-to-End

***★** = stuff to change*

---
## Before Training

- `scripts/extract_image_embeds.py`
  - Loads image dataset, SD’s VAE
  - **★ Preprocesses images (line 161) ★**
  - Encodes images to latents w/ VAE
  - Saves latents as `.pt` files w/ metadata
  - Runs PCA on full latents
  - Saves PCA latents as `.pt` files w/ metadata

- `run_eeg_encoder_experiment.sh`
  - Runs training and both evaluations
  - **★ Currently overcomplicated ★**
---

## Training Pipeline

- Start w/ `scripts/train_eeg_encoder.py`
  - Loads config
    - `eeg_encoder.yaml`
      - Holds data locations, class indices, output stuff: PCA dim
      - **★ Has encoder architecture, optimization params? ★**
  - Calls `train_eeg_encoder` from `training/train_eeg_encoder.py`
    - Loads config, takes arguments to override
    - Creates dataloader, uses `EEGImageLatentDataset` (`src/data/datasets.py`)
      - Uses EEG transform from `src/data/transforms.py`
        - Can normalize, convert to tensor
      - Returns paired EEG sample, image latent, label
      - Dataloader has stuff about #workers, pin memory, etc.
    - Creates `EEGEncoderCNN` from `src/models/eeg_encoder.py`
      - **★ Takes model architecture variables as parameters? ★**
      - Contains model blocks (conv layers, normalizations, activation functions, etc.)
      - And projection head
      - Forward pass function
    - Creates optimizer
    - Runs epochs
      - Calculates loss
    - **★ Saves artifacts: model info, metrics, graphs ★**
      - Save model itself too, architecture and all

---

## Evaluation

- Evaluate w/ files in `src/evaluation`
  - `eval_eeg_encoder.py` (IMAGES / QUALITATIVE)
    - Loads saved model checkpoint
    - Loads `EEGImageLatentDataset`, creates dataloader
    - Loads `EEGEncoderCNN`
      - **★ Loads architecture variables from saved config ★**
    - Loads PCA params, mean, components
    - Loads Stable Diffusion VAE
    - Main for loop (line 308):
      - Encodes EEG to latent space
      - Scales EEG latent to SD VAE-compatible space
      - Decodes to image
      - Saves images and ground truth figure
  - `eval_eeg_with_mean_baselines.py`
    - **★ Remove 1D CNN class, related unnecessary code ★**
    - Loads model checkpoint
      - Uses most recent if none specified
    - Loads test dataset (`EEGImageLatentDataset`), dataloader
    - Loads PCA stuff, SD’s VAE
    - Loads train dataset
    - Computes global and class means on training dataset
    - Encodes EEG to latent space, rescales, reconstructs image
    - Loads ground truth image (target_01)
    - Computes SSIM and LPIPS between target and:
      - Model (`recon_01`)
      - Global mean (`global_pred_01`)
      - Class mean (`class_pred_01`)
    - Saves info, parameters, metrics