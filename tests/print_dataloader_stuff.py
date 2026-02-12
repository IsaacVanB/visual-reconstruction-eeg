import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import build_eeg_dataloader, build_eeg_transform, build_image_transform

eeg_tf = build_eeg_transform(normalize_per_sample=True, to_tensor=True)
img_tf = build_image_transform(
    image_size=(256, 256),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)

loader = build_eeg_dataloader(
    dataset_root="datasets",
    split="train",
    transform=eeg_tf,
    image_transform=img_tf,
    batch_size=32,
    split_seed=0
)


eeg, images, labels = next(iter(loader))
print(
    f"eeg={tuple(eeg.shape)} labels={tuple(labels.shape)} "
    f"num_images={len(images)}"
)