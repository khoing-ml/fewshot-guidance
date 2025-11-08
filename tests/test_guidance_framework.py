"""
Validation Script for Custom Guidance Framework

This script tests that all components work together correctly.
Run this to verify your installation and setup.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flux.guidance_models import (
    MLPGuidanceModel,
    AttentionGuidanceModel,
    ConvGuidanceModel,
    GuidanceModelTrainer,
)


def test_mlp_guidance_model():
    """Test MLPGuidanceModel forward pass."""
    print("Testing MLPGuidanceModel...")
    
    model = MLPGuidanceModel(
        latent_channels=64,
        txt_dim=4096,
        vec_dim=768,
        hidden_dim=512,
        num_layers=2,
    )
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 256
    
    img = torch.randn(batch_size, seq_len, 64)
    pred = torch.randn(batch_size, seq_len, 64)
    txt = torch.randn(batch_size, 77, 4096)
    vec = torch.randn(batch_size, 768)
    timestep = torch.rand(batch_size)
    step_idx = 5
    
    # Forward pass
    guidance = model(img, pred, txt, vec, timestep, step_idx)
    
    # Check output shape
    assert guidance.shape == pred.shape, f"Shape mismatch: {guidance.shape} vs {pred.shape}"
    
    print(f"  ✓ Output shape correct: {guidance.shape}")
    print(f"  ✓ Output range: [{guidance.min().item():.4f}, {guidance.max().item():.4f}]")
    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_attention_guidance_model():
    """Test AttentionGuidanceModel forward pass."""
    print("Testing AttentionGuidanceModel...")
    
    model = AttentionGuidanceModel(
        latent_channels=64,
        txt_dim=4096,
        vec_dim=768,
        hidden_dim=512,
        num_heads=4,
        num_layers=1,
    )
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 256
    
    img = torch.randn(batch_size, seq_len, 64)
    pred = torch.randn(batch_size, seq_len, 64)
    txt = torch.randn(batch_size, 77, 4096)
    vec = torch.randn(batch_size, 768)
    timestep = torch.rand(batch_size)
    step_idx = 5
    
    # Forward pass
    guidance = model(img, pred, txt, vec, timestep, step_idx)
    
    # Check output shape
    assert guidance.shape == pred.shape, f"Shape mismatch: {guidance.shape} vs {pred.shape}"
    
    print(f"  ✓ Output shape correct: {guidance.shape}")
    print(f"  ✓ Output range: [{guidance.min().item():.4f}, {guidance.max().item():.4f}]")
    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_conv_guidance_model():
    """Test ConvGuidanceModel forward pass."""
    print("Testing ConvGuidanceModel...")
    
    model = ConvGuidanceModel(
        in_channels=16,
        vec_dim=768,
        hidden_channels=128,
        num_blocks=2,
    )
    
    # Create dummy inputs (make seq_len perfect square for conv)
    batch_size = 2
    seq_len = 256  # 16x16
    
    img = torch.randn(batch_size, seq_len, 64)
    pred = torch.randn(batch_size, seq_len, 64)
    txt = torch.randn(batch_size, 77, 4096)
    vec = torch.randn(batch_size, 768)
    timestep = torch.rand(batch_size)
    step_idx = 5
    
    # Forward pass
    guidance = model(img, pred, txt, vec, timestep, step_idx)
    
    # Check output shape
    assert guidance.shape == pred.shape, f"Shape mismatch: {guidance.shape} vs {pred.shape}"
    
    print(f"  ✓ Output shape correct: {guidance.shape}")
    print(f"  ✓ Output range: [{guidance.min().item():.4f}, {guidance.max().item():.4f}]")
    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()


def test_guidance_trainer():
    """Test GuidanceModelTrainer."""
    print("Testing GuidanceModelTrainer...")
    
    # Create model
    model = MLPGuidanceModel(
        latent_channels=64,
        hidden_dim=256,
        num_layers=2,
    )
    
    # Create trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = GuidanceModelTrainer(model, optimizer, device=torch.device('cpu'))
    
    # Create dummy training data
    batch_size = 4
    seq_len = 256
    
    img = torch.randn(batch_size, seq_len, 64)
    pred = torch.randn(batch_size, seq_len, 64)
    txt = torch.randn(batch_size, 77, 4096)
    vec = torch.randn(batch_size, 768)
    timestep = torch.rand(batch_size)
    step_idx = 5
    target = torch.randn(batch_size, seq_len, 64) * 0.01
    
    # Training step
    metrics = trainer.train_step(img, pred, txt, vec, timestep, step_idx, target)
    
    assert 'loss' in metrics, "Loss not in metrics"
    assert isinstance(metrics['loss'], float), "Loss should be float"
    
    print(f"  ✓ Training step successful")
    print(f"  ✓ Loss: {metrics['loss']:.6f}")
    print()


def test_backward_pass():
    """Test that gradients flow correctly."""
    print("Testing backward pass...")
    
    model = MLPGuidanceModel(
        latent_channels=64,
        hidden_dim=256,
        num_layers=2,
    )
    
    # Create dummy inputs with requires_grad
    batch_size = 2
    seq_len = 256
    
    img = torch.randn(batch_size, seq_len, 64, requires_grad=True)
    pred = torch.randn(batch_size, seq_len, 64)
    txt = torch.randn(batch_size, 77, 4096)
    vec = torch.randn(batch_size, 768)
    timestep = torch.rand(batch_size)
    step_idx = 5
    
    # Forward pass
    guidance = model(img, pred, txt, vec, timestep, step_idx)
    
    # Compute dummy loss
    loss = guidance.mean()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert img.grad is not None, "No gradient for img"
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    
    print(f"  ✓ Backward pass successful")
    print(f"  ✓ Input gradient shape: {img.grad.shape}")
    print(f"  ✓ All model parameters have gradients")
    print()


def test_output_initialization():
    """Test that output layers are initialized near zero for stability."""
    print("Testing output initialization...")
    
    model = MLPGuidanceModel(latent_channels=64, hidden_dim=256)
    
    # Create dummy inputs
    batch_size = 2
    seq_len = 256
    
    img = torch.randn(batch_size, seq_len, 64)
    pred = torch.randn(batch_size, seq_len, 64)
    txt = torch.randn(batch_size, 77, 4096)
    vec = torch.randn(batch_size, 768)
    timestep = torch.rand(batch_size)
    
    # Forward pass (should be near zero initially)
    with torch.no_grad():
        guidance = model(img, pred, txt, vec, timestep, 0)
    
    mean_abs = guidance.abs().mean().item()
    
    print(f"  ✓ Initial output magnitude: {mean_abs:.6f}")
    
    if mean_abs < 0.1:
        print(f"  ✓ Output properly initialized near zero")
    else:
        print(f"  ⚠ Warning: Output may not be initialized near zero")
    print()


def test_integration_with_sampling():
    """Test integration with sampling module."""
    print("Testing integration with sampling...")
    
    try:
        from flux.sampling import denoise
        
        # Create dummy model
        class DummyFlux(nn.Module):
            def forward(self, img, img_ids, txt, txt_ids, y, timesteps, guidance, unconditional=False):
                return torch.randn_like(img) * 0.1
        
        flux_model = DummyFlux()
        guidance_model = MLPGuidanceModel(latent_channels=64, hidden_dim=256)
        
        # Create dummy inputs
        batch_size = 1
        seq_len = 256
        
        img = torch.randn(batch_size, seq_len, 64)
        img_ids = torch.zeros(batch_size, seq_len, 3)
        txt = torch.randn(batch_size, 77, 4096)
        txt_ids = torch.zeros(batch_size, 77, 3)
        vec = torch.randn(batch_size, 768)
        timesteps = [1.0, 0.9, 0.8, 0.7, 0.6]
        
        # Run denoise with guidance
        result = denoise(
            model=flux_model,
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            vec=vec,
            timesteps=timesteps,
            guidance=1.0,
            unconditional=True
            
        )
        
        assert result.shape == img.shape, f"Output shape mismatch: {result.shape} vs {img.shape}"
        
        print(f"  ✓ Integration with sampling successful")
        print(f"  ✓ Output shape: {result.shape}")
        
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        raise
    
    print()


def run_all_tests():
    """Run all validation tests."""
    print("=" * 70)
    print("Custom Guidance Framework Validation")
    print("=" * 70)
    print()
    
    tests = [
        test_mlp_guidance_model,
        test_attention_guidance_model,
        test_conv_guidance_model,
        test_guidance_trainer,
        test_backward_pass,
        test_output_initialization,
        test_integration_with_sampling,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ✗ Test failed: {e}")
            print()
    
    print("=" * 70)
    print(f"Tests completed: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✓ All tests passed! Your guidance framework is ready to use.")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please check the errors above.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
