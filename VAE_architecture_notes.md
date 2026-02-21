# VAE Architecture Notes

This note is a practical checklist for changing the VAE architecture without breaking training.

## 1) Where To Change What

- Model architecture:
  - `src/models/vae.py`
  - Edit `hidden_dims`, conv blocks, activation types, normalization layers, latent heads.
- Training objective and optimization:
  - `src/training/train_dummy_vae.py`
  - Edit `vae_loss`, `kl_weight`, optimizer, scheduling logic.
- Data and experiment settings:
  - `configs/dummy_vae.yaml`
  - Edit `image_size`, `latent_dim`, `lr`, `batch_size`, `class_indices`, etc.

## 2) Hard Constraints (Do Not Break)

- Decoder output shape must match input image shape:
  - input: `[B, C, H, W]`
  - recon: `[B, C, H, W]`
- Output channels must equal `in_channels` (currently `3` for RGB).
- Output activation must match image normalization:
  - `Sigmoid` output -> targets in `[0, 1]`
  - `Tanh` output -> targets in `[-1, 1]`
- `image_size` in config must match what model expects (the model reads it from config in training).

## 3) Safe Knobs To Tune First

- `latent_dim`:
  - lower: stronger compression, usually blurrier reconstructions
  - higher: easier reconstructions, weaker compression pressure
- `hidden_dims` depth/width:
  - wider/deeper: better capacity, higher memory and compute
- `kl_weight`:
  - lower: better recon, weaker latent regularization
  - higher: stronger regularization, risk of posterior collapse
- `image_size`:
  - smaller (`64`) for fast iteration
  - larger (`128`/`256`) once pipeline is stable

## 4) Change Matrix

- If you change final activation:
  - update image transform normalization accordingly.
- If you change downsampling depth/strides:
  - verify decoder upsamples by the same total factor.
  - verify one forward pass returns exact input shape.
- If you change latent dimensionality:
  - checkpoint compatibility changes (old checkpoints may not load).
- If you change loss from MSE to BCE/L1:
  - re-check output range and transform range.

## 5) Quick Validation Checklist After Any Model Edit

1. Run one training epoch with small config.
2. Confirm no shape errors in forward/backward.
3. Check loss terms are finite (not NaN/inf).
4. Run `src/evaluation/eval_dummy_vae.py` and inspect reconstruction grid.
5. Confirm checkpoint can be saved and loaded.

## 6) Suggested Experiment Order

1. Keep architecture fixed, tune `kl_weight` and `latent_dim`.
2. Increase/decrease `hidden_dims` width.
3. Add/remove one encoder/decoder block.
4. Try alternate losses or KL schedules.
5. Move to larger `image_size`.

## 7) Common Failure Modes

- Reconstructions are all gray/noisy:
  - usually loss/activation/normalization mismatch or KL too high.
- KL quickly goes near zero:
  - posterior collapse; reduce decoder capacity or adjust KL schedule/weight.
- Training unstable or exploding:
  - lower LR, reduce model width/depth, check input normalization.
- Shape mismatch errors:
  - stride/padding asymmetry between encoder and decoder.

