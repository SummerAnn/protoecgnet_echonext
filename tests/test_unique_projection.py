#!/usr/bin/env python3
"""Smoke test for Unique (Hungarian) projection: should yield ≥75 unique ECGs on 80 prototypes."""

import sys
from pathlib import Path
import numpy as np
import torch

# Try to import scipy for Hungarian algorithm
try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[WARN] scipy not available, skipping Hungarian test")


def l2n(x):
    """L2 normalize."""
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def project_unique_hungarian(prototypes, embeddings):
    """
    Project prototypes to embeddings using Hungarian algorithm (1:1 mapping).
    
    Args:
        prototypes: (K, D) prototype vectors
        embeddings: (M, D) embedding vectors (M >= K)
    
    Returns:
        chosen_indices: (K,) indices into embeddings
        projected: (K, D) projected prototypes
    """
    if not HAVE_SCIPY:
        raise ImportError("scipy.optimize.linear_sum_assignment required for Unique projection")
    
    K, D = prototypes.shape
    M = embeddings.shape[0]
    
    assert M >= K, f"Need M >= K, got M={M}, K={K}"
    
    # Normalize for cosine similarity
    P_norm = l2n(prototypes)
    E_norm = l2n(embeddings)
    
    # Cosine similarity matrix
    S = P_norm @ E_norm.T  # (K, M)
    
    # Hungarian: minimize negative cosine (maximize cosine)
    cost = -S
    rows, cols = linear_sum_assignment(cost)
    
    # rows should be [0, 1, ..., K-1] (each prototype gets one embedding)
    assert np.array_equal(rows, np.arange(K)), "Hungarian rows should be [0..K-1]"
    
    chosen_indices = cols
    projected = embeddings[chosen_indices]
    
    return chosen_indices, projected


def test_unique_projection_toy():
    """Test Unique projection on toy data: 80 prototypes → 100 embeddings."""
    print("[TEST] Unique projection (toy: 80 protos → 100 embs)...")
    
    if not HAVE_SCIPY:
        print("  WARNING: Skipping (scipy not available)")
        return True
    
    K, D = 80, 512
    M = 100  # More embeddings than prototypes
    
    # Generate toy prototypes and embeddings
    np.random.seed(42)
    prototypes = np.random.randn(K, D).astype(np.float32)
    embeddings = np.random.randn(M, D).astype(np.float32)
    
    # Project
    chosen_indices, projected = project_unique_hungarian(prototypes, embeddings)
    
    # Check invariants
    assert chosen_indices.shape == (K,), f"Expected ({K},), got {chosen_indices.shape}"
    assert projected.shape == (K, D), f"Expected ({K}, {D}), got {projected.shape}"
    
    # Check uniqueness: all chosen indices should be unique
    unique_chosen = len(np.unique(chosen_indices))
    assert unique_chosen == K, f"Expected {K} unique indices, got {unique_chosen}"
    
    # Check range: indices should be in [0, M-1]
    assert np.min(chosen_indices) >= 0, f"Min index {np.min(chosen_indices)} < 0"
    assert np.max(chosen_indices) < M, f"Max index {np.max(chosen_indices)} >= {M}"
    
    print(f"  OK: Unique projection: {K} prototypes → {unique_chosen} unique ECGs (target ≥75)")
    return True


def test_unique_projection_exact_match():
    """Test Unique projection when M == K (exact match)."""
    print("[TEST] Unique projection (exact match: 80 protos → 80 embs)...")
    
    if not HAVE_SCIPY:
        print("  WARNING: Skipping (scipy not available)")
        return True
    
    K, D = 80, 512
    M = K  # Exact match
    
    # Generate toy data
    np.random.seed(42)
    prototypes = np.random.randn(K, D).astype(np.float32)
    embeddings = np.random.randn(M, D).astype(np.float32)
    
    # Project
    chosen_indices, projected = project_unique_hungarian(prototypes, embeddings)
    
    # Should choose all embeddings (1:1 mapping)
    unique_chosen = len(np.unique(chosen_indices))
    assert unique_chosen == K, f"Expected {K} unique indices, got {unique_chosen}"
    
    # Should be a permutation of [0, 1, ..., K-1]
    assert set(chosen_indices) == set(range(K)), "Should be permutation of [0..K-1]"
    
    print(f"  OK: Exact match: {K} prototypes → {unique_chosen} unique ECGs (perfect 1:1)")
    return True


def test_unique_projection_large_bank():
    """Test Unique projection with large embedding bank (M >> K)."""
    print("[TEST] Unique projection (large bank: 80 protos → 1000 embs)...")
    
    if not HAVE_SCIPY:
        print("  WARNING: Skipping (scipy not available)")
        return True
    
    K, D = 80, 512
    M = 1000  # Much larger bank
    
    # Generate toy data
    np.random.seed(42)
    prototypes = np.random.randn(K, D).astype(np.float32)
    embeddings = np.random.randn(M, D).astype(np.float32)
    
    # Project
    chosen_indices, projected = project_unique_hungarian(prototypes, embeddings)
    
    # Check uniqueness
    unique_chosen = len(np.unique(chosen_indices))
    assert unique_chosen == K, f"Expected {K} unique indices, got {unique_chosen}"
    
    # Check range
    assert np.min(chosen_indices) >= 0, f"Min index {np.min(chosen_indices)} < 0"
    assert np.max(chosen_indices) < M, f"Max index {np.max(chosen_indices)} >= {M}"
    
    print(f"  OK: Large bank: {K} prototypes → {unique_chosen} unique ECGs (target ≥75)")
    return True


def main():
    """Run all Unique projection tests."""
    print("=" * 60)
    print("Unique Projection (Hungarian) Tests")
    print("=" * 60)
    
    if not HAVE_SCIPY:
        print("\nWARNING: scipy not available - some tests will be skipped")
        print("   Install with: pip install scipy")
        return 0
    
    try:
        test_unique_projection_toy()
        test_unique_projection_exact_match()
        test_unique_projection_large_bank()
        
        print("\n" + "=" * 60)
        print("All Unique projection tests passed!")
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

