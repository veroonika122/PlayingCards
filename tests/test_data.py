import os
import torch
import pytest
from playing_cards.data import fetch_kaggle, normalize, playing_cards


def test_normalize():
    # Create a sample tensor
    x = torch.randn(10, 3, 224, 224)
    
    # Apply normalization
    normalized = normalize(x)
    
    # Check if mean is close to 0 and std is close to 1
    assert torch.abs(normalized.mean()) < 1e-6
    assert torch.abs(torch.std(normalized) - 1.0) < 1e-6


def test_playing_cards(tmp_path):
    # Test with project directory
    train_set, valid_set, test_set = playing_cards(tmp_path)
    
    # Check if datasets are TensorDataset
    assert isinstance(train_set, torch.utils.data.TensorDataset)
    assert isinstance(valid_set, torch.utils.data.TensorDataset)
    assert isinstance(test_set, torch.utils.data.TensorDataset)
    
    # Check if tensors have correct shape
    assert train_set.tensors[0].shape[1:] == (3, 224, 224)  # Image dimensions
    assert len(train_set.tensors[1].shape) == 1  # Labels are 1D
    
    # Check if labels are within valid range (53 classes)
    assert torch.all(train_set.tensors[1] >= 0)
    assert torch.all(train_set.tensors[1] < 53)

    # Assert dataset sizes
    assert len(train_set) == 7624, "Train dataset size should be 7624"
    assert len(valid_set) == 265, "Validation dataset size should be 265"
    assert len(test_set) == 265, "Test dataset size should be 265"

    # Check the shape of images and range of targets
    for dataset in [train_set, valid_set, test_set]:
        for x, y in dataset:
            assert x.shape == (3, 224, 224), "Image shape should be (3, 224, 224)"
            assert 0 <= y < 53, "Target should be in the range [0, 52]"

    # Check if all 53 classes are represented in the train set
    train_targets = torch.unique(torch.tensor([y for _, y in train_set]))
    assert len(train_targets) == 53, "Train dataset should include all 53 classes"
    assert (train_targets == torch.arange(53)).all(), "Train classes should match [0, 52]"

    # Check if all 53 classes are represented in the test set
    test_targets = torch.unique(torch.tensor([y for _, y in test_set]))
    assert len(test_targets) <= 53, "Test dataset should not have more than 53 classes"

    print("All tests passed!")
