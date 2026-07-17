import numpy as np
import pytest

from src.data.datasets import EEGImageDataset, EEGLabelDataset


@pytest.fixture
def split_datasets(synthetic_dataset_root):
    common = {
        "dataset_root": str(synthetic_dataset_root),
        "subject": "sub-1",
        "split_seed": 42,
        "return_image_name": True,
    }
    return {
        split: EEGImageDataset(split=split, **common)
        for split in ("train", "valid", "test")
    }


def test_splits_partition_every_image_once(split_datasets):
    split_indices = np.concatenate(
        [split_datasets[split]._split_image_indices for split in ("train", "valid", "test")]
    )

    assert len(split_indices) == 20
    assert np.array_equal(np.sort(split_indices), np.arange(20))
    assert len(np.unique(split_indices)) == len(split_indices)


@pytest.mark.parametrize(
    ("split", "images_per_class"),
    [("train", 7), ("valid", 2), ("test", 1)],
)
def test_split_lengths_include_all_repetitions(
    synthetic_dataset_root, split, images_per_class
):
    dataset = EEGImageDataset(
        dataset_root=str(synthetic_dataset_root),
        split=split,
    )

    assert len(dataset) == 2 * images_per_class * 4


def test_getitem_keeps_eeg_image_name_and_label_paired(split_datasets):
    for dataset in split_datasets.values():
        for split_position, global_index in enumerate(dataset._split_image_indices):
            sample_index = split_position * dataset.repetitions
            eeg, image, label, image_name = dataset[sample_index]

            assert np.array_equal(eeg, dataset.eeg[global_index, 0])
            assert image_name == dataset.train_img_files[global_index]
            assert label == global_index // dataset.images_per_class
            assert image.mode == "RGB"


def test_split_is_reproducible_for_same_seed(synthetic_dataset_root):
    kwargs = {
        "dataset_root": str(synthetic_dataset_root),
        "split": "train",
        "split_seed": 7,
    }

    first = EEGLabelDataset(**kwargs)
    second = EEGLabelDataset(**kwargs)

    assert np.array_equal(first._split_image_indices, second._split_image_indices)


def test_class_filter_only_returns_selected_class(synthetic_dataset_root):
    dataset = EEGLabelDataset(
        dataset_root=str(synthetic_dataset_root),
        split="train",
        class_indices=[1],
    )

    assert len(dataset) == 7 * 4
    assert {dataset[index][1] for index in range(len(dataset))} == {1}


@pytest.mark.parametrize("class_indices", [[], [2], [0, 0]])
def test_invalid_class_filters_are_rejected(synthetic_dataset_root, class_indices):
    with pytest.raises(ValueError):
        EEGLabelDataset(
            dataset_root=str(synthetic_dataset_root),
            class_indices=class_indices,
        )
