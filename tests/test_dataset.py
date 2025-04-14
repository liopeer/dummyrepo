import pytest
from classification.dataset import ClothingDataset

@pytest.fixture
def dataset():
    return ClothingDataset("dataset_clothing_images")

def test_dataset_initialization(dataset):
    assert len(dataset) > 0
    assert len(dataset.classes) > 0
    assert hasattr(dataset, 'class_to_idx')

def test_dataset_getitem(dataset):
    img, label = dataset[0]
    assert img.shape == (3, 224, 224)  # Check image dimensions
    assert isinstance(label, int)
    assert 0 <= label < len(dataset.classes)
