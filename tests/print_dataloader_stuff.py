import os
import sys
from pathlib import Path

from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.data import EEGImageDataset, build_eeg_transform, build_image_transform


def _parse_class_indices(raw: str):
    raw = raw.strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]

eeg_tf = build_eeg_transform(normalize_per_sample=True, to_tensor=True)
img_tf = build_image_transform(
    image_size=(256, 256),
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
)
class_indices = list(range(0, 200, 2))
#_parse_class_indices(os.environ.get("EEG_CLASS_INDICES", ""))
split = os.environ.get("EEG_SPLIT", "train")
batch_size = int(os.environ.get("EEG_BATCH_SIZE", "32"))
split_seed = int(os.environ.get("EEG_SPLIT_SEED", "0"))

dataset = EEGImageDataset(
    dataset_root="datasets",
    split=split,
    class_indices=class_indices,
    transform=eeg_tf,
    image_transform=img_tf,
    split_seed=split_seed,
)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

eeg, images, labels = next(iter(loader))
print(f"split={split}")
print(f"class_indices={class_indices}")
print(f"dataset_len={len(dataset)}")
print(
    f"eeg={tuple(eeg.shape)} images={tuple(images.shape)} labels={tuple(labels.shape)}"
)
