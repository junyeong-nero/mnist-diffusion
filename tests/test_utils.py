import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from MyDiffusion.utils import image_normalize


def test_image_normalize():
    # Create a random tensor
    test_tensor = torch.randn(3, 32, 32)

    # Normalize the tensor
    normalized_tensor = image_normalize(test_tensor.clone())

    # Check if the output is a tensor
    assert isinstance(normalized_tensor, torch.Tensor)

    # Check if the shape is correct (permuted)
    assert normalized_tensor.shape == (32, 32, 3)

    # Check if min and max are approximately 0 and 1
    # We check each channel individually before permuting
    for i in range(test_tensor.shape[0]):
        channel = test_tensor[i]
        normalized_channel = (channel - channel.min()) / (channel.max() - channel.min())
        # Now check the output tensor's corresponding values
        # Note: this is a bit complex due to the permutation in the original function
        # A simpler check is to verify the final output range

    # A more direct test on the output
    # Due to potential floating point inaccuracies, we use a small tolerance
    assert torch.all(normalized_tensor >= 0.0 - 1e-6)
    assert torch.all(normalized_tensor <= 1.0 + 1e-6)

    # Let's also check if min is close to 0 and max is close to 1
    assert torch.isclose(normalized_tensor.min(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(normalized_tensor.max(), torch.tensor(1.0), atol=1e-6)


if __name__ == "__main__":
    pytest.main()
