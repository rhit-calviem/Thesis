"""Test script to verify the OmniModel can run."""
import torch
from models import OmniModel, Model, ShallowFeatureExtraction, ImageReconstruction
from config import UPSCALE_FACTOR

def test_components():
    """Test individual components."""
    print("Testing individual components...")
    
    # Test Shallow Feature Extraction
    print("\n1. Testing ShallowFeatureExtraction...")
    shallow = ShallowFeatureExtraction(in_channels=3, out_channels=64)
    x = torch.randn(1, 3, 64, 64)
    out = shallow(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (1, 64, 64, 64), f"Expected (1, 64, 64, 64), got {out.shape}"
    print("   ✓ ShallowFeatureExtraction working!")
    
    # Test Image Reconstruction
    print("\n2. Testing ImageReconstruction...")
    reconstruction = ImageReconstruction(in_channels=64, out_channels=3, upscale_factor=4)
    x = torch.randn(1, 64, 64, 64)
    out = reconstruction(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {out.shape}"
    print("   ✓ ImageReconstruction working!")
    
    # Test OmniModel
    print("\n3. Testing OmniModel...")
    model = OmniModel(
        num_local_blocks=6,
        channels=64,
        upscale_factor=4,
        in_channels=3,
        out_channels=3
    )
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == (1, 3, 256, 256), f"Expected (1, 3, 256, 256), got {out.shape}"
    print("   ✓ OmniModel working!")
    
    # Test Model class for compatibility
    print("\n4. Testing Model class (backward compatibility)...")
    model = Model(upscale_factor=UPSCALE_FACTOR)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    expected_shape = (1, 3, 64 * UPSCALE_FACTOR, 64 * UPSCALE_FACTOR)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    print(f"   Expected shape: {expected_shape}")
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print("   ✓ Model class working!")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)

def test_forward_pass():
    """Test full forward pass with realistic input."""
    print("\nTesting full forward pass...")
    model = OmniModel(num_local_blocks=6, channels=64, upscale_factor=4)
    
    # Simulate a batch of LR images
    lr_images = torch.randn(4, 3, 32, 32)  # batch of 4, 32x32 LR images
    print(f"Input LR images shape: {lr_images.shape}")
    
    model.eval()
    with torch.no_grad():
        hr_images = model(lr_images)
    
    print(f"Output HR images shape: {hr_images.shape}")
    print(f"   Expected: (4, 3, 128, 128)")
    assert hr_images.shape == (4, 3, 128, 128)
    print("   ✓ Full forward pass successful!")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal model parameters: {num_params:,}")

if __name__ == "__main__":
    print("="*50)
    print("Testing OmniModel Implementation")
    print("="*50)
    
    test_components()
    test_forward_pass()
    
    print("\n✅ All tests completed successfully!")
    print("The model is ready to run.")


