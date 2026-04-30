# Running EEG Classifier Training In Google Colab

This note describes the recommended way to train the `classifier20` EEG classifier
from a Google Colab notebook.

The important principle is: use the notebook as an orchestration layer, not as a
fork of the training code. Keep the model, dataset, preprocessing, checkpointing,
and evaluation logic in `src/` and call the existing scripts from notebook cells.
That keeps Colab runs aligned with local runs and avoids notebook-only code drift.

## Recommended Colab Flow

### 1. Enable a GPU runtime

In Colab:

```text
Runtime -> Change runtime type -> Hardware accelerator -> GPU
```

Then verify PyTorch can see CUDA:

```python
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

### 2. Mount Google Drive

Use Drive for persistent datasets, checkpoints, and outputs:

```python
from google.colab import drive

drive.mount("/content/drive")
```

### 3. Clone or upload the repo

If the repo is available from GitHub:

```bash
%cd /content
!git clone <your-repo-url> visual-reconstruction-eeg
%cd /content/visual-reconstruction-eeg
```

If you upload the repo another way, make sure the current working directory is the
repo root before running scripts:

```bash
%cd /content/visual-reconstruction-eeg
```

### 4. Install dependencies

```bash
!pip install -r requirements.txt
```

If Colab already has compatible PyTorch installed, avoid reinstalling PyTorch
unless necessary. Reinstalling large GPU packages can waste runtime time and
occasionally create CUDA version mismatches.

### 5. Place the dataset

The classifier expects this dataset structure:

```text
<dataset_root>/
  THINGS_EEG_2/
    image_metadata.npy
    sub-1/preprocessed_eeg_training.npy
    sub-2/preprocessed_eeg_training.npy
    ...
  images_THINGS/
    object_images/
    ...
```

For Drive-backed data, a common layout is:

```text
/content/drive/MyDrive/eeg_data/datasets/
  THINGS_EEG_2/
  images_THINGS/
```

You can point training directly at this path with `--dataset-root`.

## Training From A Notebook

### Single subject

```bash
!python scripts/train_eeg_classifier.py \
  --dataset-root /content/drive/MyDrive/eeg_data/datasets \
  --subject sub-1 \
  --output-dir /content/drive/MyDrive/eeg_runs/eeg_classifier \
  --device cuda
```

### Multiple explicit subjects

```bash
!python scripts/train_eeg_classifier.py \
  --dataset-root /content/drive/MyDrive/eeg_data/datasets \
  --subjects sub-1 sub-2 sub-3 \
  --output-dir /content/drive/MyDrive/eeg_runs/eeg_classifier \
  --device cuda
```

### All available subjects

```bash
!python scripts/train_eeg_classifier.py \
  --dataset-root /content/drive/MyDrive/eeg_data/datasets \
  --subjects all \
  --output-dir /content/drive/MyDrive/eeg_runs/eeg_classifier \
  --device cuda
```

`--subjects all` resolves every available
`THINGS_EEG_2/sub-*/preprocessed_eeg_training.npy` file under `dataset_root`.

By default, each training run writes to a fresh timestamped directory:

```text
<output-dir>/run_YYYYMMDD_HHMMSS/
```

For example:

```text
/content/drive/MyDrive/eeg_runs/eeg_classifier/run_20260429_220217/
```

## Evaluation From A Notebook

After training, evaluate the best checkpoint:

```bash
!python scripts/eval_eeg_classifier.py \
  --checkpoint-path /content/drive/MyDrive/eeg_runs/eeg_classifier/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --device cuda
```

Evaluation artifacts are saved by default to:

```text
<run_dir>/eval/
```

The evaluator writes:

```text
confusion_matrix.png
confusion_matrix_normalized.png
confusion_matrix.npy
confusion_matrix.csv
confusion_matrix.json
predictions.csv
classification_summary.json
```

You can display the confusion matrix in the notebook:

```python
from IPython.display import Image, display

display(Image("/content/drive/MyDrive/eeg_runs/eeg_classifier/run_YYYYMMDD_HHMMSS/eval/confusion_matrix.png"))
```

## Performance And Memory Advice

### Prefer local `/content` for active training data when possible

Google Drive is persistent but can be slow for repeated reads. The classifier now
streams one subject at a time to avoid high peak RAM, but training can still be
I/O-bound if every subject is read from Drive repeatedly.

Best performance pattern:

```bash
!mkdir -p /content/eeg_data
!cp -r /content/drive/MyDrive/eeg_data/datasets /content/eeg_data/datasets
```

Then train from local Colab storage and save outputs to Drive:

```bash
!python scripts/train_eeg_classifier.py \
  --dataset-root /content/eeg_data/datasets \
  --subjects all \
  --output-dir /content/drive/MyDrive/eeg_runs/eeg_classifier \
  --device cuda
```

This uses fast local disk for dataset reads while preserving checkpoints and
metrics in Drive.

### Keep `num_workers` at 0 first

The config default is:

```yaml
num_workers: 0
```

Keep that setting for initial Colab runs, especially with `subjects: all`. Extra
workers can multiply dataset memory use because each worker may load its own data
copy.

### Use the streamed multi-subject path

For `subjects: all`, the training code streams subjects sequentially instead of
holding every subject dataset in memory at the same time. This is specifically to
avoid laptop or Colab RAM spikes.

Expected sample counts for `classifier20` with `sample_mode: repetitions`:

```text
1 subject:
  train: 560
  valid: 160
  test: 80

10 subjects:
  train: 5600
  valid: 1600
  test: 800
```

### Save outputs to Drive

Colab runtimes are temporary. Anything left only under `/content` can disappear
when the runtime resets. Use a Drive-backed `--output-dir` for training runs:

```bash
--output-dir /content/drive/MyDrive/eeg_runs/eeg_classifier
```

If you train using local `/content` outputs for speed, copy the run directory to
Drive before ending the session.

### Use short smoke runs first

Before launching a long all-subject run, verify paths and GPU setup:

```bash
!python scripts/train_eeg_classifier.py \
  --dataset-root /content/eeg_data/datasets \
  --subjects sub-1 \
  --epochs 1 \
  --batch-size 8 \
  --output-dir /content/drive/MyDrive/eeg_runs/eeg_classifier_smoke \
  --device cuda
```

Then test evaluation:

```bash
!python scripts/eval_eeg_classifier.py \
  --checkpoint-path /content/drive/MyDrive/eeg_runs/eeg_classifier_smoke/run_YYYYMMDD_HHMMSS/eeg_classifier20_best_YYYYMMDD_HHMMSS.pt \
  --split test \
  --max-samples 16 \
  --device cuda
```

## Config-Based Runs

You can also edit `configs/eeg_classifier.yaml` in the notebook and run:

```bash
!python scripts/train_eeg_classifier.py --config configs/eeg_classifier.yaml
```

For Colab, typical config changes are:

```yaml
dataset_root: /content/eeg_data/datasets
subjects: all
device: cuda
output_dir: /content/drive/MyDrive/eeg_runs/eeg_classifier
num_workers: 0
```

CLI arguments override config values, so you can keep the YAML generic and pass
Colab-specific paths from notebook cells.

## Troubleshooting

### `Killed`

This usually means the OS killed the process due to RAM pressure.

Use:

```yaml
num_workers: 0
subjects: all
```

and make sure you are on the current streamed-subject implementation. If it still
happens, reduce batch size:

```bash
--batch-size 8
```

### CUDA is not used

Check:

```python
import torch
print(torch.cuda.is_available())
```

Then pass:

```bash
--device cuda
```

If CUDA is unavailable, switch the Colab runtime type to GPU.

### Dataset path errors

Print and inspect the expected files:

```bash
!find /content/drive/MyDrive/eeg_data/datasets/THINGS_EEG_2 -maxdepth 2 -name preprocessed_eeg_training.npy | head
!ls /content/drive/MyDrive/eeg_data/datasets/THINGS_EEG_2/image_metadata.npy
```

Then make sure `--dataset-root` points to the directory that contains
`THINGS_EEG_2`, not to `THINGS_EEG_2` itself.

### Drive is slow

Copy the dataset to `/content` for the active run, and write only outputs to
Drive. This is usually the biggest speed improvement for Colab training.
