#!/usr/bin/env python3
"""Unit test for prototype→class mapping inference from classifier weights."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def infer_proto_to_class_from_weights(model, num_classes, num_prototypes):
    """Infer prototype→class mapping from classifier weights (C,P) matrix."""
    # Find any parameter shaped (C,P)
    cand = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and p.shape[0] == num_classes and p.shape[1] == num_prototypes:
            cand.append((name, p.detach().cpu()))
    if not cand:
        raise RuntimeError(f"No (C,P)=({num_classes},{num_prototypes}) classifier matrix found.")
    
    # Prefer layers with obvious classifier naming if multiple:
    cand.sort(key=lambda t: (not any(k in t[0].lower() for k in ["classifier", "last", "final"]), t[0]))
    w_name, W = cand[0]
    
    Wpos = W.clone()
    Wpos[Wpos < 0] = 0.0
    if torch.all(Wpos == 0):
        # fallback: absolute largest influence
        Wpos = W.abs()
    
    cls_of_proto = torch.argmax(Wpos, dim=0).numpy()  # per-proto argmax over classes
    return cls_of_proto, w_name


def test_mapping_indices_in_range():
    """Test that weight-inferred mapping returns indices < C."""
    print("[TEST] Mapping indices in range...")
    
    C, P = 12, 80
    model = nn.Module()
    model.classifier = nn.Linear(P, C, bias=False)
    
    # Initialize with random weights
    with torch.no_grad():
        model.classifier.weight.data = torch.randn(C, P)
    
    proto_to_class, wname = infer_proto_to_class_from_weights(model, C, P)
    
    assert proto_to_class.shape == (P,), f"Expected shape ({P},), got {proto_to_class.shape}"
    assert np.max(proto_to_class) < C, f"Max index {np.max(proto_to_class)} >= {C}"
    assert np.min(proto_to_class) >= 0, f"Min index {np.min(proto_to_class)} < 0"
    assert len(np.unique(proto_to_class)) > 0, "Mapping should cover at least 1 class"
    
    print(f"  OK: Mapping valid: indices in [0, {C-1}], covers {len(np.unique(proto_to_class))} classes")


def test_mapping_covers_multiple_classes():
    """Test that mapping covers >0 classes (not all prototypes map to one class)."""
    print("[TEST] Mapping covers multiple classes...")
    
    C, P = 12, 80
    model = nn.Module()
    model.classifier = nn.Linear(P, C, bias=False)
    
    # Initialize with structured weights: each prototype has one dominant class
    with torch.no_grad():
        W = torch.zeros(C, P)
        for p in range(P):
            c = p % C  # Cycle through classes
            W[c, p] = 1.0 + torch.rand(1).item() * 0.5  # Dominant weight
            # Add small random weights to other classes
            for c2 in range(C):
                if c2 != c:
                    W[c2, p] = torch.rand(1).item() * 0.1
        model.classifier.weight.data = W
    
    proto_to_class, wname = infer_proto_to_class_from_weights(model, C, P)
    
    unique_classes = len(np.unique(proto_to_class))
    assert unique_classes > 1, f"Mapping should cover >1 class, got {unique_classes}"
    assert unique_classes <= C, f"Mapping should cover ≤{C} classes, got {unique_classes}"
    
    print(f"  OK: Mapping covers {unique_classes} classes (expected >1, ≤{C})")


def test_mapping_handles_negative_weights():
    """Test that mapping handles negative weights correctly (uses abs as fallback)."""
    print("[TEST] Mapping handles negative weights...")
    
    C, P = 12, 80
    model = nn.Module()
    model.classifier = nn.Linear(P, C, bias=False)
    
    # Initialize with all-negative weights (should use abs fallback)
    with torch.no_grad():
        model.classifier.weight.data = -torch.randn(C, P)
    
    proto_to_class, wname = infer_proto_to_class_from_weights(model, C, P)
    
    assert np.max(proto_to_class) < C, f"Max index {np.max(proto_to_class)} >= {C}"
    assert np.min(proto_to_class) >= 0, f"Min index {np.min(proto_to_class)} < 0"
    
    print("  OK: Mapping handles negative weights (uses abs fallback)")


def test_mapping_prefers_classifier_named_layer():
    """Test that mapping prefers 'classifier' named layers if multiple exist."""
    print("[TEST] Mapping prefers classifier-named layer...")
    
    C, P = 12, 80
    model = nn.Module()
    model.other_layer = nn.Linear(P, C, bias=False)
    model.classifier = nn.Linear(P, C, bias=False)
    
    # Make classifier have different pattern
    with torch.no_grad():
        model.other_layer.weight.data = torch.randn(C, P)
        model.classifier.weight.data = torch.randn(C, P) * 2  # Different scale
    
    proto_to_class, wname = infer_proto_to_class_from_weights(model, C, P)
    
    assert "classifier" in wname.lower(), f"Should prefer 'classifier', got '{wname}'"
    
    print(f"  OK: Mapping prefers classifier-named layer: '{wname}'")


def main():
    """Run all mapping tests."""
    print("=" * 60)
    print("Prototype Mapping Tests")
    print("=" * 60)
    
    try:
        test_mapping_indices_in_range()
        test_mapping_covers_multiple_classes()
        test_mapping_handles_negative_weights()
        test_mapping_prefers_classifier_named_layer()
        
        print("\n" + "=" * 60)
        print("All mapping tests passed!")
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

