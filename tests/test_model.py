import torch
import pytest
from playing_cards.model import CNN, HuggingfaceResnet, TimmResNet


def test_model_initialization():
    """Test if the models can be initialized without errors."""
    modelCNN = CNN()
    modelTimm = TimmResNet()
    modelHuggingface = HuggingfaceResnet()

    assert isinstance(modelCNN, torch.nn.Module), "ModelCNN should be an instance of torch.nn.Module."
    assert isinstance(modelTimm, torch.nn.Module), "ModelTimm should be an instance of torch.nn.Module."
    assert isinstance(modelHuggingface, torch.nn.Module), "ModelHuggingface should be an instance of torch.nn.Module."


def test_forward_pass_convolution():
    """Test the forward pass of the ModelConvolution."""
    model = CNN()
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels (RGB), 224x224 image
    output = model(dummy_input)
    assert output.shape == (1, 53), f"Expected output shape (1, 53), but got {output.shape}."


def test_forward_pass_resnet():
    """Test the forward pass of the Hugging Face ResNet model."""
    resnet_model = TimmResNet()
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch size of 1, 3 channels (RGB), 224x224 image
    output = resnet_model(dummy_input)  # Access logits for classification
    assert output.shape == (1, 53), f"Expected output shape (1, 53), but got {output.shape}."


def test_parameter_count():
    """Test if ModelConvolution has trainable parameters."""
    model = TimmResNet()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert num_params > 0, "ModelConvolution should have trainable parameters."


def test_resnet_pretrained_weights():
    """Test if the Hugging Face ResNet model loads pretrained weights correctly."""
    resnet_model = TimmResNet()
    assert hasattr(resnet_model, "state_dict"), "Hugging Face ResNet should have a state_dict method."
    state_dict = resnet_model.state_dict()
    assert len(state_dict) > 0, "State dict should not be empty."


def test_invalid_input_shape():
    """Test if ModelConvolution raises an error for invalid input shapes."""
    model = TimmResNet()
    invalid_input = torch.randn(1, 1, 224, 224)  # Only 1 channel instead of 3
    try:
        _ = model(invalid_input)
        assert False, "Model should raise an error for invalid input shape."
    except RuntimeError:
        pass  # Expected behavior


def test_cnn_forward():
    model = CNN()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 53)


def test_huggingface_resnet_forward():
    model = HuggingfaceResnet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 53)


def test_timm_resnet_forward():
    model = TimmResNet()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, 53)


def test_invalid_input_shape():
    model = CNN()
    x = torch.randn(1, 3, 32, 32)  # Wrong input size
    with pytest.raises(RuntimeError):
        model(x)


def test_batch_processing():
    model = CNN()
    x = torch.randn(4, 3, 224, 224)  # Batch size of 4
    y = model(x)
    assert y.shape == (4, 53)


def test_model_parameters():
    model = CNN()
    # Check if model has trainable parameters
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) > 0


def test_dropout_layers():
    model = CNN()
    # Check if model contains dropout layers
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout


if __name__ == "__main__":
    # Run tests
    test_model_initialization()
    test_forward_pass_convolution()
    test_forward_pass_resnet()
    test_parameter_count()
    test_dropout_layers()
    test_resnet_pretrained_weights()
    test_invalid_input_shape()
    test_cnn_forward()
    test_huggingface_resnet_forward()
    test_timm_resnet_forward()
    test_batch_processing()
    test_model_parameters()
    print("All tests passed!")
