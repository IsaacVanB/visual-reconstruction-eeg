# EEG Encoder Notes

This note is a practical checklist for modifying the EEG encoder and training loop safely.

## 1) Where To Change What

- Encoder architecture:
  - `src/models/eeg_encoder.py`
  - Change conv blocks, kernels, pooling, activations, dropout, head MLP.
- Training objective + optimization:
  - `src/training/train_eeg_encoder.py`
  - Change loss, optimizer, LR/WD, batch size, schedules.
- Experiment settings:
  - `configs/eeg_encoder.yaml` (or your chosen config path)
  - Change data roots, class subsets, model hyperparams.

## 2) Hard Constraints (Do Not Break)

- EEG input shape contract is `[B, C, T]`.
- Model output dim must equal latent target dim exactly:
  - `output_dim == sample_latent.numel()`.
- Data/latent consistency:
  - `dataset_root`, `latent_root`, and PCA metadata must come from compatible preprocessing.
- Grouped conv constraint in `block2`:
  - with `groups=temporal_filters`, block2 out channels must be divisible by `temporal_filters`.

## 3) Safe Knobs To Tune First

- Capacity:
  - `temporal_filters`, `depth_multiplier`, head hidden width.
- Temporal receptive field:
  - `temporal_kernel1`, `temporal_kernel3`.
- Temporal compression:
  - `pool1`, `pool3`.
- Regularization:
  - `dropout`, `weight_decay`.
- Optimization:
  - `lr`, `batch_size`, optimizer choice.

## 4) Change Matrix

- If you change target latent dimension (e.g., PCA k):
  - update `output_dim` to match.
  - existing checkpoints may no longer load.
- If you change EEG normalization:
  - keep train/eval transforms consistent.
- If you change architecture depth/strides/pooling:
  - verify flattened feature dim remains valid (current model infers it dynamically).
- If you change loss:
  - update evaluation interpretation accordingly (MSE vs cosine vs L1 metrics).

## 5) Quick Validation Checklist After Any Edit

1. Run one short training job (1-2 epochs).
2. Confirm no shape mismatch errors.
3. Confirm loss is finite and trending down on train.
4. Check valid loss is reasonable vs train (no immediate divergence).
5. Save/load checkpoint and run `src/evaluation/eval_eeg_encoder.py`.

## 6) Common Failure Modes

- Shape mismatch at training step:
  - usually `output_dim` mismatch with latent target dim.
- Loss not decreasing:
  - LR too high/low, insufficient capacity, or data-normalization mismatch.
- Overfitting quickly:
  - increase dropout/weight decay, reduce capacity, add class diversity.
- Poor decoded image quality in downstream eval:
  - latent predictor underfit, or mismatch between encoder output and PCA/decoder assumptions.

## 7) Suggested Experiment Order

1. Keep loss fixed (MSE), tune `lr`, `batch_size`, `weight_decay`.
2. Tune `temporal_filters` and `depth_multiplier`.
3. Tune temporal kernels/pooling.
4. Try alternate losses (Huber, cosine).
5. Expand class subset and longer training schedules.

