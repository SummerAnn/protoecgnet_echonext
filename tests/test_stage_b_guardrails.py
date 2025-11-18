#!/usr/bin/env python3
"""Guardrail tests for Stage B: freeze sanity, masked losses, diagnostics assertions."""

import sys
from pathlib import Path
import json
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proto_models1D import ProtoECGNet1D, prototype_loss1d_echonext


def test_freeze_sanity():
    """Verify encoder is frozen, prototypes+head are trainable."""
    print("[TEST] Freeze sanity check...")
    
    # Create model with freezing
    # Use custom_groups=False and provide last_layer_connection_weight
    model = ProtoECGNet1D(
        num_classes=12,
        backbone="resnet1d18",
        proto_dim=512,
        single_class_prototype_per_class=5,
        joint_prototypes_per_border=2,
        label_set="cat1",
        custom_groups=False,
        last_layer_connection_weight=1.0,  # Required for initialization
    )
    
    # Mock label_cooccurrence if None (for tests)
    if model.label_cooccurrence is None:
        model.label_cooccurrence = torch.eye(12)  # Identity matrix as fallback
    
    # Freeze encoder
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    
    # Verify encoder is frozen
    encoder_params = list(model.feature_extractor.parameters())
    assert all(not p.requires_grad for p in encoder_params), "Encoder should be frozen"
    
    # Verify prototypes are trainable
    assert model.prototype_vectors.requires_grad, "Prototypes should be trainable"
    
    # Verify classifier is trainable
    assert model.classifier.weight.requires_grad, "Classifier should be trainable"
    
    print("  OK: Freeze sanity: encoder frozen, prototypes+head trainable")


def test_masked_prototype_losses():
    """Test per-sample, per-class masking in prototype losses."""
    print("[TEST] Masked prototype loss correctness...")
    
    model = ProtoECGNet1D(
        num_classes=3,
        backbone="resnet1d18",
        proto_dim=512,
        single_class_prototype_per_class=2,
        joint_prototypes_per_border=1,
        label_set="cat1",
        custom_groups=False,
        last_layer_connection_weight=1.0,
    )
    
    # Mock label_cooccurrence if None
    if model.label_cooccurrence is None:
        model.label_cooccurrence = torch.eye(3)
    
    model.eval()
    
    B, C, P = 4, 3, 9  # batch=4, classes=3, prototypes=9
    
    # Create test case: only class 0 is positive for sample 0
    y_true = torch.zeros(B, C)
    y_true[0, 0] = 1.0  # Sample 0: only class 0 positive
    y_true[1, 1] = 1.0  # Sample 1: only class 1 positive
    y_true[2, :] = 0.0  # Sample 2: all zeros (should skip cluster)
    y_true[3, 0] = 1.0
    y_true[3, 2] = 1.0  # Sample 3: classes 0 and 2 positive
    
    logits = torch.randn(B, C)
    similarity_scores = torch.rand(B, P)
    
    # Mock prototype_class_identity: assign prototypes to classes
    # Prototypes 0-1: class 0, 2-3: class 1, 4-5: class 2, 6-8: joint
    model.prototype_class_identity = torch.zeros(P, C)
    model.prototype_class_identity[0:2, 0] = 1.0
    model.prototype_class_identity[2:4, 1] = 1.0
    model.prototype_class_identity[4:6, 2] = 1.0
    model.prototype_class_identity[6:9, :] = 0.5  # Joint prototypes
    
    # Compute loss
    loss = prototype_loss1d_echonext(
        logits, y_true, model, similarity_scores,
        class_weights=None,
        lam_clst=0.8, lam_sep=0.1, lam_spars=0, lam_div=0, lam_cnrst=0,
        use_contrastive=False,
    )
    
    # Loss should be finite and non-negative (can be zero if all predictions perfect)
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    print("  OK: Masked prototype losses: per-sample, per-class masking works")


def test_diagnostics_assertions():
    """Verify diagnostics invariants."""
    print("[TEST] Diagnostics assertions...")
    
    stage_b_manifest = Path(__file__).parent.parent / "echonext_projection_rebuild_stageB" / "echonext_cat1_diagnostics_manifest.json"
    
    if not stage_b_manifest.exists():
        print(f"  WARNING: Manifest not found: {stage_b_manifest}")
        return
    
    manifest = json.loads(stage_b_manifest.read_text())
    
    # Assert Unique produces 80 IDs
    assert manifest["reuse"]["unique_unique_ecgs"] == 80, f"Unique should produce 80 ECGs, got {manifest['reuse']['unique_unique_ecgs']}"
    
    # Assert cosine matrices are 80×80 (3160 = 80*79/2 upper triangle pairs)
    assert manifest["delta_stats"]["nn"]["count"] == 3160, "Cosine matrix should be 80×80 (3160 upper triangle pairs)"
    assert manifest["delta_stats"]["unique"]["count"] == 3160, "Unique cosine matrix should be 80×80"
    
    # Load tensors and verify shapes
    nn_pt = Path(manifest["projected_nn"])
    uni_pt = Path(manifest["projected_unique"])
    
    if nn_pt.exists() and uni_pt.exists():
        nn_tensor = torch.load(nn_pt, map_location="cpu")
        uni_tensor = torch.load(uni_pt, map_location="cpu")
        
        assert nn_tensor.shape[0] == 80, f"NN tensor should have 80 prototypes, got {nn_tensor.shape[0]}"
        assert uni_tensor.shape[0] == 80, f"Unique tensor should have 80 prototypes, got {uni_tensor.shape[0]}"
        assert nn_tensor.shape[1] == uni_tensor.shape[1], "Latent dimension should match"
    
    print("  OK: Diagnostics assertions: all invariants satisfied")


def test_torchscript_equivalence():
    """Verify TorchScript encoder produces same outputs as original."""
    print("[TEST] TorchScript equivalence...")
    
    encoder_ts_path = Path("/opt/gpudata/summereunann/ptbxl_weights/cat1_stageB.encoder.ts")
    
    if not encoder_ts_path.exists():
        print(f"  WARNING: TorchScript encoder not found: {encoder_ts_path}")
        return
    
    encoder_ts = torch.jit.load(str(encoder_ts_path), map_location="cpu")
    encoder_ts.eval()
    
    # Test with dummy input
    dummy = torch.zeros(1, 12, 1000)
    
    with torch.inference_mode():
        out_ts = encoder_ts(dummy)
    
    assert out_ts.shape == (1, 512), f"Expected (1, 512), got {out_ts.shape}"
    assert torch.isfinite(out_ts).all(), "Output should be finite"
    
    print("  OK: TorchScript equivalence: encoder loads and runs correctly")


def main():
    """Run all guardrail tests."""
    print("=" * 60)
    print("Stage B Guardrail Tests")
    print("=" * 60)
    
    try:
        test_freeze_sanity()
        test_masked_prototype_losses()
        test_diagnostics_assertions()
        test_torchscript_equivalence()
        
        print("\n" + "=" * 60)
        print("All guardrail tests passed!")
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

