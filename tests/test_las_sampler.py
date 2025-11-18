#!/usr/bin/env python3
"""Tests for Label-Aware Batch Sampler (PR5-B)."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from data.echonext import LabelAwareBatchSampler
except ImportError:
    # Try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("echonext", Path(__file__).parent.parent / "src" / "data" / "echonext.py")
    echonext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(echonext)
    LabelAwareBatchSampler = echonext.LabelAwareBatchSampler


def test_las_biases_rare_classes():
    """Test that LAS upweights rare classes in sampled batches."""
    print("[TEST] LAS biases rare classes...")
    
    # toy multi-label: C=4 with class 3 super rare
    N, C = 200, 4
    rng = np.random.default_rng(0)
    Y = rng.random((N, C)) < np.array([0.40, 0.20, 0.08, 0.02])[None, :]
    
    sampler = LabelAwareBatchSampler(
        labels_bool=Y,
        batch_size=32,
        iters_per_epoch=50,
        gamma=0.7,
        seed=1
    )
    
    counts = np.zeros(C, dtype=int)
    total = 0
    
    for batch in sampler:
        total += len(batch)
        for idx in batch:
            counts += Y[idx]
    
    # relative frequency in sampled batches should upweight rare class 3
    prop = counts / counts.sum() if counts.sum() > 0 else np.zeros(C)
    
    # assert: rare class gets at least half of common class proportion (very lenient, avoids flakiness)
    assert prop[3] > 0.5 * prop[2], f"Rare class 3 proportion {prop[3]:.4f} should be > 0.5 * class 2 {prop[2]:.4f}"
    assert prop[3] > 0.3 * prop[0], f"Rare class 3 proportion {prop[3]:.4f} should be > 0.3 * class 0 {prop[0]:.4f}"
    
    print(f"  OK: Rare class upweighted: prop={prop}, class 3 > 0.5*class 2 and > 0.3*class 0")


def test_batch_size_and_uniqueness():
    """Test that batches have correct size and unique indices."""
    print("[TEST] Batch size and uniqueness...")
    
    Y = np.zeros((50, 3), dtype=bool)
    Y[:10, 0] = True
    Y[10:20, 1] = True
    Y[20:25, 2] = True
    
    sampler = LabelAwareBatchSampler(
        labels_bool=Y,
        batch_size=16,
        iters_per_epoch=5,
        gamma=0.5,
        seed=7
    )
    
    for batch in sampler:
        assert len(batch) == 16, f"Batch size should be 16, got {len(batch)}"
        assert len(batch) == len(set(batch)), f"Batch should have unique indices, got duplicates"
        assert all(0 <= idx < 50 for idx in batch), f"Indices should be in [0, 50), got {batch}"
    
    print("  OK: Batch size and uniqueness verified")


def test_las_handles_empty_classes():
    """Test that LAS handles classes with zero positives."""
    print("[TEST] LAS handles empty classes...")
    
    Y = np.zeros((20, 3), dtype=bool)
    Y[:10, 0] = True  # class 0 has positives
    Y[10:15, 1] = True  # class 1 has positives
    # class 2 has no positives
    
    sampler = LabelAwareBatchSampler(
        labels_bool=Y,
        batch_size=8,
        iters_per_epoch=3,
        gamma=0.5,
        seed=42
    )
    
    # Should not crash
    batches = list(sampler)
    assert len(batches) == 3, f"Should have 3 batches, got {len(batches)}"
    
    # All indices should be valid
    for batch in batches:
        assert all(0 <= idx < 20 for idx in batch), f"Invalid indices in batch: {batch}"
    
    print("  OK: Empty classes handled correctly")


def main():
    """Run all LAS sampler tests."""
    print("=" * 60)
    print("Label-Aware Batch Sampler Tests")
    print("=" * 60)
    
    try:
        test_las_biases_rare_classes()
        test_batch_size_and_uniqueness()
        test_las_handles_empty_classes()
        
        print("\n" + "=" * 60)
        print("All LAS sampler tests passed!")
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

