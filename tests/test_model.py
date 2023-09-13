import torch
import pytest



import sys
print(sys.path)


from gender_classification.code.model import Network
# Create a fixture to instantiate the model
@pytest.fixture
def model():
    return Network(classes=2)  # You can customize classes as needed

# Define a test for the forward pass
def test_forward_pass(model):
    # Create a dummy input tensor with the expected shape
    input_tensor = torch.randn(1, 3, 224, 224)  # Assumes 3-channel RGB images and 224x224 size

    # Ensure that the forward pass works without errors
    try:
        output1, output2 = model(input_tensor)
    except Exception as e:
        pytest.fail(f"Forward pass failed with an exception: {str(e)}")

    # Check the shapes of the outputs, you may need to adjust these depending on your model
    assert output1.shape == torch.Size([1, 1])  # Check the shape of x1
    assert output2.shape == torch.Size([1, 2])  # Check the shape of x2


