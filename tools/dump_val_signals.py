#!/usr/bin/env python3
"""Dump val-set signals needed for diagnostics:
- y_true (B,C), y_pred (B,C) after sigmoid
- proto_acts (B,P) pooled similarity/activation per prototype
- optional gradient norms: head (per class) + prototypes (per proto) from a few batches
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.echonext import get_echonext_dataloaders
from proto_models1D import ProtoECGNet1D


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--grad-batches", type=int, default=4,
                    help="Accumulate grads over N val batches (0 disables grad capture).")
    ap.add_argument("--limit", type=int, default=0, help="Optional sample cap.")
    return ap.parse_args()


@torch.no_grad()
def forward_collect(model, loader, device, limit=0):
    """Collect predictions and prototype activations."""
    y_true, y_pred, proto_acts = [], [], []
    n = 0
    model.eval()
    
    for x, y in loader:
        x = x.to(device).float()  # (B,12,T)
        y = y.to(device).float()  # (B,C)
        
        # Model forward - EchoNext returns (logits, distances, similarity_scores)
        out = model(x)
        
        if isinstance(out, (list, tuple)) and len(out) >= 1:
            logits = out[0]
            # Try to find pooled prototype scores/activations
            sims = None
            if len(out) >= 3 and out[2] is not None:
                sims = out[2]  # shape (B,P) expected
        else:
            logits, sims = out, None

        y_true.append(y.detach().cpu().numpy())
        y_pred.append(torch.sigmoid(logits).detach().cpu().numpy())
        if sims is not None:
            proto_acts.append(sims.detach().cpu().numpy())

        n += x.size(0)
        if limit and n >= limit:
            break

    Y = np.concatenate(y_true, axis=0)
    P = np.concatenate(y_pred, axis=0)
    S = np.concatenate(proto_acts, axis=0) if proto_acts else None
    return Y, P, S


def collect_gradients(model, loader, device, batches, limit=0):
    """
    Rough gradient-share proxy:
      - BCE loss on a few val batches
      - grad L2-norm per class row in classifier
      - grad L2-norm per prototype vector
    """
    if batches <= 0:
        return None, None
    
    model.train()  # enable grads, but we won't step
    
    # zero grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    crit = torch.nn.BCEWithLogitsLoss()
    seen = 0
    bdone = 0
    
    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float()
        
        # Get logits
        out = model(x)
        if isinstance(out, (list, tuple)):
            logits = out[0]
        else:
            logits = out
        
        loss = crit(logits, y)
        
        # Zero grads before backward
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        
        loss.backward()

        bdone += 1
        seen += x.size(0)
        if bdone >= batches:
            break
        if limit and seen >= limit:
            break

    head_grad = None
    proto_grad = None

    # classifier weight assumed at model.classifier.weight (C,D)
    if hasattr(model, "classifier") and hasattr(model.classifier, "weight") and model.classifier.weight.grad is not None:
        g = model.classifier.weight.grad.detach().cpu()
        head_grad = torch.linalg.vector_norm(g, dim=1).numpy()  # per-class grad L2

    # prototypes assumed at model.prototype_vectors (P,D)
    if hasattr(model, "prototype_vectors") and getattr(model.prototype_vectors, "grad", None) is not None:
        pg = model.prototype_vectors.grad.detach().cpu()
        proto_grad = torch.linalg.vector_norm(pg, dim=1).numpy()  # per-proto grad L2

    model.eval()
    return head_grad, proto_grad


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Dataloaders (EchoNext)
    print(f"[INFO] Loading EchoNext data from {args.dataset_root}...")
    train_loader, val_loader, test_loader, class_weights, class_freqs = get_echonext_dataloaders(
        dataset_root=str(args.dataset_root),
        train_split="train",
        val_split=args.val_split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_set="1",  # cat1
    )
    
    # Get MLB from train dataset to get class names
    mlb = train_loader.dataset.mlb
    if mlb is None:
        # Fallback: infer from labels shape
        num_classes = train_loader.dataset.labels.shape[1]
        class_names = [f"class_{i}" for i in range(num_classes)]
        print(f"[WARN] MLB not available, inferred {num_classes} classes")
    else:
        num_classes = len(mlb.classes_)
        class_names = mlb.classes_.tolist()
        print(f"[INFO] Found {num_classes} classes: {class_names}")

    # Model - load from checkpoint
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    
    # Infer dims from state
    if "model.prototype_vectors" in state:
        num_prototypes = state["model.prototype_vectors"].shape[0]
        proto_dim = state["model.prototype_vectors"].shape[1]
    elif "prototype_vectors" in state:
        num_prototypes = state["prototype_vectors"].shape[0]
        proto_dim = state["prototype_vectors"].shape[1]
    else:
        raise KeyError("Could not find prototype_vectors in checkpoint")
    
    print(f"[INFO] Checkpoint: {num_prototypes} prototypes, {proto_dim}D, {num_classes} classes")
    
    # Infer prototype config from checkpoint
    # Standard cat1: 5 per class, 2 per border
    # num_prototypes = (5 * num_classes) + (2 * num_class_pairs)
    # num_class_pairs = num_classes * (num_classes - 1) / 2
    # For 12 classes: (5*12) + (2*66) = 60 + 132 = 192
    # But checkpoint has 80, so likely 5 per class only (no joint)
    # Try: 80 / 12 ≈ 6.67, or 80 / 16 ≈ 5 (if 16 classes from PTB)
    # Actually, let's infer from the checkpoint structure
    
    # Build model - the model will auto-adjust num_prototypes from checkpoint
    # Use standard cat1 config, model will detect mismatch and adjust
    model = ProtoECGNet1D(
        num_classes=num_classes,
        backbone="resnet1d18",
        proto_dim=proto_dim,
        single_class_prototype_per_class=5,
        joint_prototypes_per_border=2,
        label_set="cat1",
        custom_groups=False,
        last_layer_connection_weight=1.0,
        load_strict=False,
        pretrained_weights=str(args.ckpt),  # Pass checkpoint to trigger auto-adjust
    )
    
    # Now load state dict - model should have adjusted num_prototypes
    print(f"[INFO] Model initialized with {model.num_prototypes} prototypes")
    
    # Load state dict
    if any(k.startswith("model.") for k in state.keys()):
        state_dict = {k.replace("model.", ""): v for k, v in state.items()}
    else:
        state_dict = state
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected[:5]}...")
    
    model.to(args.device)
    model.eval()

    # Forward collect
    print(f"[INFO] Collecting predictions on {args.val_split} set...")
    Y, P, S = forward_collect(model, val_loader, args.device, limit=args.limit)
    print(f"[INFO] Collected {Y.shape[0]} samples")
    
    # Ensure S is available for mapping fallback if needed
    if S is None:
        S = np.zeros((Y.shape[0], num_prototypes))

    # Optional gradient norms (over a few batches)
    head_grad, proto_grad = None, None
    if args.grad_batches > 0:
        print(f"[INFO] Collecting gradients over {args.grad_batches} batches...")
        head_grad, proto_grad = collect_gradients(model, val_loader, args.device,
                                                  batches=args.grad_batches, limit=args.limit)
        if head_grad is not None:
            print(f"[INFO] Head grad shape: {head_grad.shape}")
        if proto_grad is not None:
            print(f"[INFO] Proto grad shape: {proto_grad.shape}")

    # Prototype→class inference function
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
    
    # Prototype→class (12-way EchoNext) from classifier weights, not PTB identity
    try:
        proto_to_class, wname = infer_proto_to_class_from_weights(
            model, num_classes=num_classes, num_prototypes=num_prototypes
        )
        print(f"[map] Using classifier weights '{wname}' to map prototypes to EchoNext classes.")
    except Exception as e:
        print(f"[map] Failed weight-based mapping; falling back to activation correlation. {e}")
        # Fallback: assign each proto to the class whose positives yield highest mean activation
        if S is None:
            raise RuntimeError("No proto_acts available; cannot build activation-based mapping.")
        proto_to_class = []
        for p in range(S.shape[1]):
            means = []
            for c in range(num_classes):
                idx = np.where(Y[:, c] == 1)[0]
                means.append(0.0 if idx.size == 0 else float(S[idx, p].mean()))
            proto_to_class.append(int(np.argmax(means)))
        proto_to_class = np.asarray(proto_to_class)

    # Sanity check: proto_to_class indices must be < num_classes
    assert np.max(proto_to_class) < num_classes, \
        f"proto_to_class has indices ≥ {num_classes} (max={np.max(proto_to_class)}); mapping is inconsistent."
    
    out = {
        "y_true": Y,
        "y_pred": P,
        "proto_acts": S if S is not None else np.zeros((Y.shape[0], num_prototypes)),
        "class_names": class_names,
        "proto_to_class": proto_to_class.tolist(),
        "head_grad_l2": None if head_grad is None else head_grad.tolist(),
        "proto_grad_l2": None if proto_grad is None else proto_grad.tolist(),
    }
    
    np.savez_compressed(args.out, **out)
    
    meta = {
        "ckpt": str(args.ckpt),
        "val_split": args.val_split,
        "limit": args.limit,
        "grad_batches": args.grad_batches,
        "num_classes": int(num_classes),
        "num_prototypes": int(num_prototypes),
        "proto_dim": int(proto_dim),
    }
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"[OK] wrote {args.out} and {args.out}.json")


if __name__ == "__main__":
    main()

