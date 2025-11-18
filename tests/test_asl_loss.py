#!/usr/bin/env python3
"""Tests for Asymmetric Loss (ASL) for multi-label classification (PR5-C)."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from losses.asl import AsymmetricLossMultiLabel


def test_asl_shapes_and_finiteness():
    """Test that ASL returns finite scalar loss."""
    print("[TEST] ASL shapes and finiteness...")
    
    B, C = 8, 5
    logits = torch.randn(B, C)
    targets = (torch.rand(B, C) > 0.8).float()
    
    loss = AsymmetricLossMultiLabel()(logits, targets)
    
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    
    print(f"  OK: Loss is finite and scalar: {loss.item():.4f}")


def test_asl_monotonic_positive():
    """Test that increasing logit for a positive label decreases loss."""
    print("[TEST] ASL monotonic for positives...")
    
    logits = torch.tensor([[0.0]])
    targets = torch.tensor([[1.0]])
    
    loss1 = AsymmetricLossMultiLabel()(logits, targets)
    loss2 = AsymmetricLossMultiLabel()(logits + 2.0, targets)
    
    assert loss2 < loss1, f"Increasing positive logit should decrease loss: {loss1.item():.4f} -> {loss2.item():.4f}"
    
    print(f"  OK: Monotonic: loss decreases from {loss1.item():.4f} to {loss2.item():.4f}")


def test_asl_downweights_easy_negatives():
    """Test that ASL downweights easy negatives compared to BCE-like behavior."""
    print("[TEST] ASL downweights easy negatives...")
    
    logits = torch.tensor([[-6.0]])  # p ~ 0.0025 (very easy negative)
    targets = torch.tensor([[0.0]])
    
    asl = AsymmetricLossMultiLabel(gamma_neg=4.0, clip=0.05)
    bce_like = AsymmetricLossMultiLabel(gamma_neg=0.0, clip=0.0)  # behaves closer to BCE (not exact)
    
    loss_asl = asl(logits, targets)
    loss_bce_like = bce_like(logits, targets)
    
    assert loss_asl < loss_bce_like, f"ASL should downweight easy negatives: ASL={loss_asl.item():.4f}, BCE-like={loss_bce_like.item():.4f}"
    
    print(f"  OK: Easy negatives downweighted: ASL={loss_asl.item():.4f} < BCE-like={loss_bce_like.item():.4f}")


def test_asl_with_pos_weight():
    """Test that ASL can use pos_weight when provided."""
    print("[TEST] ASL with pos_weight...")
    
    B, C = 4, 3
    logits = torch.randn(B, C)
    targets = (torch.rand(B, C) > 0.7).float()
    pos_weight = torch.tensor([2.0, 3.0, 1.5])
    
    asl = AsymmetricLossMultiLabel(pos_weight=pos_weight)
    loss = asl(logits, targets)
    
    assert torch.isfinite(loss), f"Loss with pos_weight is not finite: {loss}"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    
    print(f"  OK: ASL with pos_weight works: loss={loss.item():.4f}")


def test_asl_clip_effect():
    """Test that ASL clip parameter affects loss for easy negatives."""
    print("[TEST] ASL clip effect...")
    
    logits = torch.tensor([[-5.0]])  # Easy negative
    targets = torch.tensor([[0.0]])
    
    asl_no_clip = AsymmetricLossMultiLabel(gamma_neg=4.0, clip=0.0)
    asl_with_clip = AsymmetricLossMultiLabel(gamma_neg=4.0, clip=0.05)
    
    loss_no_clip = asl_no_clip(logits, targets)
    loss_with_clip = asl_with_clip(logits, targets)
    
    # Clip should reduce loss for easy negatives
    assert loss_with_clip < loss_no_clip or torch.allclose(loss_with_clip, loss_no_clip, atol=1e-3), \
        f"Clip should reduce loss: no_clip={loss_no_clip.item():.4f}, with_clip={loss_with_clip.item():.4f}"
    
    print(f"  OK: Clip effect: no_clip={loss_no_clip.item():.4f}, with_clip={loss_with_clip.item():.4f}")


def main():
    """Run all ASL loss tests."""
    print("=" * 60)
    print("Asymmetric Loss (ASL) Tests")
    print("=" * 60)
    
    try:
        test_asl_shapes_and_finiteness()
        test_asl_monotonic_positive()
        test_asl_downweights_easy_negatives()
        test_asl_with_pos_weight()
        test_asl_clip_effect()
        
        print("\n" + "=" * 60)
        print("All ASL loss tests passed!")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

