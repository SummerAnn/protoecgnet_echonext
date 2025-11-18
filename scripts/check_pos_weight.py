#!/usr/bin/env python3
"""Sanity-check pos_weight computation from class_freqs.json."""

import json
import sys
from pathlib import Path
import torch


def stats(x):
    """Compute statistics for a tensor."""
    x = torch.tensor(x).float()
    sorted_x = torch.sort(x)[0]
    n = len(x)
    return {
        "min": float(x.min()),
        "p50": float(sorted_x[n // 2]),
        "p90": float(sorted_x[int(0.9 * n)]) if n > 0 else float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/check_pos_weight.py <class_freqs.json>")
        sys.exit(1)
    
    freq_path = Path(sys.argv[1])
    if not freq_path.exists():
        print(f"Error: {freq_path} not found")
        sys.exit(1)
    
    meta = json.loads(freq_path.read_text())
    
    pos = torch.tensor(meta["pos_counts"]).float()
    neg = torch.tensor(meta["neg_counts"]).float()
    clip = meta.get("weight_clip_max", 5.0)
    use_sqrt = meta.get("use_inverse_sqrt", False)
    
    # Compute ratios
    pos_safe = torch.clamp(pos, min=1.0)
    ratio = neg / pos_safe
    ratio_sqrt = ratio.sqrt()
    
    print("=" * 60)
    print("Pos Weight Sanity Check")
    print("=" * 60)
    print(f"Classes: {len(pos)}")
    print(f"Clip max: {clip}")
    print(f"Use sqrt: {use_sqrt}")
    print(f"Zero-pos classes: {meta.get('zero_pos_classes', [])}")
    print()
    
    print("Raw ratio (neg/pos):")
    print(f"  {stats(ratio)}")
    print()
    
    print("Sqrt ratio:")
    print(f"  {stats(ratio_sqrt)}")
    print()
    
    print("Clipped (no sqrt):")
    clipped = torch.clamp(ratio, min=1.0, max=clip)
    print(f"  {stats(clipped)}")
    print()
    
    print("Clipped (with sqrt):")
    clipped_sqrt = torch.clamp(ratio_sqrt, min=1.0, max=clip)
    print(f"  {stats(clipped_sqrt)}")
    print()
    
    # Final weights from metadata
    final_weights = torch.tensor(meta["final_weights"]).float()
    print("Final weights (from metadata):")
    print(f"  {stats(final_weights)}")
    print()
    
    # Sanity checks
    assert torch.isfinite(final_weights).all(), "Non-finite weights detected!"
    assert (final_weights >= 1.0).all(), f"Weights below 1.0: {final_weights[final_weights < 1.0]}"
    assert (final_weights <= clip).all(), f"Weights above clip: {final_weights[final_weights > clip]}"
    
    print("All sanity checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

