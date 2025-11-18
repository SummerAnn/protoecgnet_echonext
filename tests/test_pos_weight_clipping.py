#!/usr/bin/env python3
"""Unit tests for pos_weight clipping and bounds."""

import sys
from pathlib import Path
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def compute_pos_weight(pos_counts, neg_counts, weight_clip_max=5.0, use_inverse_sqrt=False):
    """Compute pos_weight with clipping (matches echonext.py logic)."""
    pos = torch.tensor(pos_counts, dtype=torch.float32)
    neg = torch.tensor(neg_counts, dtype=torch.float32)
    
    # Check for zero-positive classes
    zero_pos_mask = pos == 0
    
    # Avoid division by zero
    pos_safe = torch.clamp(pos, min=1.0)
    neg_safe = torch.clamp(neg, min=1.0)
    
    # Compute ratio
    ratio = neg_safe / pos_safe
    
    # Clamp minimum to 1.0
    ratio = torch.clamp(ratio, min=1.0)
    
    # Optional sqrt
    if use_inverse_sqrt:
        ratio = torch.sqrt(ratio)
    
    # Clip maximum
    ratio = torch.clamp(ratio, max=weight_clip_max)
    
    # For zero-positive classes, set to clip_max
    if zero_pos_mask.any():
        ratio[zero_pos_mask] = weight_clip_max
    
    return ratio, zero_pos_mask


def test_pos_weight_bounds():
    """Test that pos_weight is finite and in bounds [1.0, clip_max]."""
    print("[TEST] Pos weight bounds...")
    
    # Include edge cases: zero-pos, very rare, common
    pos = torch.tensor([1., 10., 100., 0., 5.])  # include zero-pos edge
    neg = torch.tensor([999., 900., 800., 500., 995.])
    clip = 5.0
    
    # Test without sqrt
    w, zero_mask = compute_pos_weight(pos, neg, clip, use_inverse_sqrt=False)
    
    assert torch.isfinite(w).all(), f"Non-finite weights: {w[~torch.isfinite(w)]}"
    assert float(w.min()) >= 1.0, f"Min weight {w.min()} < 1.0"
    assert float(w.max()) <= clip, f"Max weight {w.max()} > {clip}"
    assert zero_mask.any(), "Should detect zero-pos class"
    assert float(w[zero_mask]) == clip, f"Zero-pos class should get clip_max, got {w[zero_mask]}"
    
    # Test with sqrt
    w_sqrt, _ = compute_pos_weight(pos, neg, clip, use_inverse_sqrt=True)
    
    assert torch.isfinite(w_sqrt).all(), f"Non-finite sqrt weights: {w_sqrt[~torch.isfinite(w_sqrt)]}"
    assert float(w_sqrt.min()) >= 1.0, f"Min sqrt weight {w_sqrt.min()} < 1.0"
    assert float(w_sqrt.max()) <= clip, f"Max sqrt weight {w_sqrt.max()} > {clip}"
    
    print(f"  OK: Bounds verified: weights in [1.0, {clip}], sqrt weights in [1.0, {clip}]")


def test_pos_weight_edge_cases():
    """Test edge cases: all zeros, all ones, extreme imbalance."""
    print("[TEST] Pos weight edge cases...")
    
    clip = 5.0
    
    # Case 1: All classes have positives
    pos1 = torch.tensor([1., 10., 100.])
    neg1 = torch.tensor([999., 990., 900.])
    w1, zero1 = compute_pos_weight(pos1, neg1, clip, False)
    assert not zero1.any(), "Should have no zero-pos classes"
    assert torch.isfinite(w1).all(), "All weights should be finite"
    
    # Case 2: One class has zero positives
    pos2 = torch.tensor([0., 10., 100.])
    neg2 = torch.tensor([1000., 990., 900.])
    w2, zero2 = compute_pos_weight(pos2, neg2, clip, False)
    assert zero2[0], "Should detect zero-pos class at index 0"
    assert float(w2[0]) == clip, f"Zero-pos should get clip_max, got {w2[0]}"
    
    # Case 3: Extreme imbalance (very rare class)
    pos3 = torch.tensor([1., 1000.])
    neg3 = torch.tensor([9999., 0.])
    w3, zero3 = compute_pos_weight(pos3, neg3, clip, False)
    assert float(w3[0]) == clip, f"Very rare class should be clipped, got {w3[0]}"
    assert float(w3[1]) == 1.0, f"Common class should get min weight, got {w3[1]}"
    
    print("  OK: Edge cases handled: zero-pos, extreme imbalance, all-positive")


def test_pos_weight_sqrt_vs_no_sqrt():
    """Test that sqrt produces gentler (lower) weights."""
    print("[TEST] Sqrt vs no-sqrt comparison...")
    
    pos = torch.tensor([1., 10., 100.])
    neg = torch.tensor([999., 900., 800.])
    clip = 5.0
    
    w_no_sqrt, _ = compute_pos_weight(pos, neg, clip, False)
    w_sqrt, _ = compute_pos_weight(pos, neg, clip, True)
    
    # Sqrt should produce lower or equal weights (gentler)
    assert (w_sqrt <= w_no_sqrt).all(), "Sqrt weights should be ≤ no-sqrt weights"
    assert float(w_sqrt.mean()) < float(w_no_sqrt.mean()), "Sqrt should have lower mean"
    
    print(f"  OK: Sqrt is gentler: mean {w_sqrt.mean():.3f} < {w_no_sqrt.mean():.3f}")


def main():
    """Run all pos_weight tests."""
    print("=" * 60)
    print("Pos Weight Clipping Tests")
    print("=" * 60)
    
    try:
        test_pos_weight_bounds()
        test_pos_weight_edge_cases()
        test_pos_weight_sqrt_vs_no_sqrt()
        
        print("\n" + "=" * 60)
        print("All pos_weight clipping tests passed!")
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

