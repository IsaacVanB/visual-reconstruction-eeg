from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def synthetic_dataset_root(tmp_path: Path) -> Path:
    """Create the smallest representative two-class THINGS-EEG dataset."""
    eeg_dir = tmp_path / "THINGS_EEG_2" / "sub-1"
    eeg_dir.mkdir(parents=True)

    num_images = 20
    eeg = np.arange(num_images * 4 * 17 * 100, dtype=np.float32).reshape(
        num_images, 4, 17, 100
    )
    np.save(eeg_dir / "preprocessed_eeg_training.npy", eeg)

    image_names = []
    image_root = tmp_path / "images_THINGS" / "object_images"
    for class_idx in range(2):
        class_name = f"class_{class_idx}"
        class_dir = image_root / class_name
        class_dir.mkdir(parents=True)
        for image_idx in range(10):
            image_name = f"{class_name}_{image_idx:02d}.png"
            image_names.append(image_name)
            Image.new(
                "RGB",
                (4, 4),
                color=(class_idx * 100, image_idx * 10, 50),
            ).save(class_dir / image_name)

    metadata_path = tmp_path / "THINGS_EEG_2" / "image_metadata.npy"
    np.save(metadata_path, {"train_img_files": image_names})
    return tmp_path
