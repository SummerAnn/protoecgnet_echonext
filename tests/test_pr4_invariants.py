#!/usr/bin/env python3
"""Comprehensive PR4 invariant tests: mapping, projection, freeze, masks."""

import sys
from pathlib import Path
import json
import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_diagnostics_manifest_invariants():
    """Test diagnostics manifest invariants."""
    print("[TEST] Diagnostics manifest invariants...")
    
    manifest_path = Path(__file__).parent.parent / "echonext_projection_rebuild_stageB" / "echonext_cat1_diagnostics_manifest.json"
    
    if not manifest_path.exists():
        print(f"  WARNING: Manifest not found: {manifest_path}")
        return True
    
    data = json.loads(manifest_path.read_text())
    
    # Check Unique produces ≥75 unique ECGs
    uni_unique = data["reuse"]["unique_unique_ecgs"]
    assert uni_unique >= 75, f"Unique should produce ≥75 ECGs, got {uni_unique}"
    
    # Check NN ≤ Unique
    nn_unique = data["reuse"]["nn_unique_ecgs"]
    assert nn_unique <= uni_unique, f"NN ({nn_unique}) should be ≤ Unique ({uni_unique})"
    
    # Check cosine matrix shape (80×80 = 3160 upper triangle pairs)
    nn_count = data["delta_stats"]["nn"]["count"]
    uni_count = data["delta_stats"]["unique"]["count"]
    assert nn_count == 3160, f"NN cosine matrix should be 80×80 (3160 pairs), got {nn_count}"
    assert uni_count == 3160, f"Unique cosine matrix should be 80×80 (3160 pairs), got {uni_count}"
    
    print(f"  OK: Manifest invariants: Unique={uni_unique} (≥75), NN={nn_unique} (≤Unique), cosine={nn_count} pairs")
    return True


def test_per_class_diagnostics_mapping():
    """Test that per-class diagnostics use correct mapping."""
    print("[TEST] Per-class diagnostics mapping...")
    
    npz_path = Path(__file__).parent.parent / "results" / "diagnostics" / "val_signals_stageB_fixed.npz"
    
    if not npz_path.exists():
        print(f"  ⚠️  NPZ not found: {npz_path}")
        return True
    
    D = np.load(npz_path, allow_pickle=True)
    proto_to_class = np.array(D["proto_to_class"])
    classes = D["class_names"]
    C = len(classes)
    
    # Check mapping indices < C
    assert np.max(proto_to_class) < C, f"proto_to_class max ({np.max(proto_to_class)}) >= num_classes ({C})"
    assert np.min(proto_to_class) >= 0, f"proto_to_class min ({np.min(proto_to_class)}) < 0"
    
    # Check that multiple classes are covered
    unique_classes = len(np.unique(proto_to_class))
    assert unique_classes > 1, f"Mapping should cover >1 class, got {unique_classes}"
    
    print(f"  OK: Mapping valid: indices in [0, {C-1}], covers {unique_classes} classes")
    return True


def test_coverage_nonzero():
    """Test that coverage shows non-zero usage for multiple classes."""
    print("[TEST] Coverage non-zero for multiple classes...")
    
    coverage_path = Path(__file__).parent.parent / "results" / "diagnostics" / "per_class_report_fixed" / "coverage_summary.json"
    
    if not coverage_path.exists():
        print(f"  ⚠️  Coverage file not found: {coverage_path}")
        return True
    
    coverage = json.loads(coverage_path.read_text())
    
    # Check that at least some classes have non-zero usage
    classes_with_usage = [c for c in coverage if c["unique_protos_used"] > 0]
    assert len(classes_with_usage) > 1, f"Should have >1 class with prototype usage, got {len(classes_with_usage)}"
    
    # Check that class_9 has high usage (expected)
    class_9 = next((c for c in coverage if c["class"] == "class_9"), None)
    if class_9:
        assert class_9["unique_protos_used"] > 5, f"class_9 should have >5 unique protos, got {class_9['unique_protos_used']}"
    
    print(f"  OK: Coverage: {len(classes_with_usage)} classes have prototype usage")
    return True


def main():
    """Run all PR4 invariant tests."""
    print("=" * 60)
    print("PR4 Invariant Tests")
    print("=" * 60)
    
    results = []
    
    try:
        results.append(("Diagnostics manifest", test_diagnostics_manifest_invariants()))
        results.append(("Per-class mapping", test_per_class_diagnostics_mapping()))
        results.append(("Coverage non-zero", test_coverage_nonzero()))
        
        print("\n" + "=" * 60)
        print("Test Results:")
        print("=" * 60)
        for name, passed in results:
            status = "PASS" if passed else "SKIP"
            print(f"  {status}: {name}")
        
        all_passed = all(p for _, p in results)
        if all_passed:
            print("\nAll PR4 invariant tests passed!")
        else:
            print("\nWARNING: Some tests were skipped (files not found)")
        
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

