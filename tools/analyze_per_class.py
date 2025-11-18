#!/usr/bin/env python3
"""Reads the NPZ from dump_val_signals.py and prints/saves:
- per-class pos/neg, AUROC, PR-AUC (+ low-support flag)
- prototype coverage per class (unique protos for positives, top-5 overuse)
- gradient-share proxy (norms over classifier rows and per-class proto grads)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--coverage-topk", type=int, default=3, help="Protos counted per positive (top-k by activation).")
    ap.add_argument("--lowpos-thresh", type=int, default=20, help="Flag classes with < this many positives.")
    return ap.parse_args()


def safe_auc(y, p):
    """Skip if trivially positive/negative or too few positives."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos < 2 or neg < 2:
        return None
    try:
        return roc_auc_score(y, p)
    except Exception:
        return None


def safe_ap(y, p):
    """Skip if no positives."""
    pos = int((y == 1).sum())
    if pos < 1:
        return None
    try:
        return average_precision_score(y, p)
    except Exception:
        return None


def gini_from_counts(counts: np.ndarray):
    """Gini on nonnegative counts (0 = uniform use, 1 = maximally skewed)."""
    x = counts.astype(np.float64)
    if x.sum() <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    cum = np.cumsum(x)
    gini = (n + 1 - 2 * (cum / x.sum()).sum() / n)
    return float(gini)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Loading {args.npz}...")
    D = np.load(args.npz, allow_pickle=True)
    
    Y = D["y_true"]  # (N,C)
    P = D["y_pred"]  # (N,C)
    classes = D["class_names"].tolist()
    proto_to_class = np.array(D["proto_to_class"])
    proto_acts = D["proto_acts"] if "proto_acts" in D.files else None
    head_grad = D["head_grad_l2"] if "head_grad_l2" in D.files else None
    proto_grad = D["proto_grad_l2"] if "proto_grad_l2" in D.files else None

    N, C = Y.shape
    
    # Guardrail: proto_to_class indices must be < num_classes
    assert np.max(proto_to_class) < C, \
        f"proto_to_class contains indices ≥ {C} (max={np.max(proto_to_class)}); mapping is inconsistent with current class space."
    per_class = []
    macro_auc_vals, macro_ap_vals = [], []

    # Coverage counters
    Pcount = proto_to_class.shape[0]
    per_class_proto_counts = np.zeros((C, Pcount), dtype=np.int64)

    print(f"[INFO] Analyzing {N} samples, {C} classes, {Pcount} prototypes...")
    
    for c in range(C):
        y_c = Y[:, c]
        p_c = P[:, c]
        pos = int((y_c == 1).sum())
        neg = int((y_c == 0).sum())

        auc = safe_auc(y_c, p_c)
        ap = safe_ap(y_c, p_c)
        if auc is not None:
            macro_auc_vals.append(auc)
        if ap is not None:
            macro_ap_vals.append(ap)

        # Prototype coverage: count top-k prototypes among positives (if proto_acts provided)
        topk = []
        if proto_acts is not None and pos > 0:
            # for each positive sample for class c, take top-k activated prototypes
            pos_idx = np.where(y_c == 1)[0]
            acts_pos = proto_acts[pos_idx]  # (n_pos, P)
            # choose top-k per sample
            k = min(args.coverage_topk, acts_pos.shape[1])
            top_idx = np.argpartition(-acts_pos, kth=k - 1, axis=1)[:, :k]
            # flatten and count
            flat = top_idx.reshape(-1)
            # restrict to prototypes that belong to class c
            belong = np.where(proto_to_class == c)[0]
            # count only those
            sel = flat[np.isin(flat, belong)]
            for j in sel:
                per_class_proto_counts[c, j] += 1
            topk = sel

        per_class.append({
            "class": classes[c],
            "idx": c,
            "pos": pos,
            "neg": neg,
            "auc": None if auc is None else float(auc),
            "pr_auc": None if ap is None else float(ap),
            "low_support": pos < args.lowpos_thresh
        })

    # Macro metrics (ignore None)
    macro_auc = float(np.mean(macro_auc_vals)) if macro_auc_vals else None
    macro_ap = float(np.mean(macro_ap_vals)) if macro_ap_vals else None
    micro_auc = safe_auc(Y.ravel(), P.ravel())
    micro_ap = safe_ap(Y.ravel(), P.ravel())

    # Coverage summaries
    coverage_summary = []
    for c in range(C):
        counts = per_class_proto_counts[c]  # (P,)
        belong = np.where(proto_to_class == c)[0]
        if belong.size == 0:
            print(f"[warn] No prototypes assigned to class {classes[c]}; "
                  "check mapping or weight matrix shape.")
        used = counts[belong] if belong.size > 0 else np.array([], dtype=np.int64)
        unique_protos = int((used > 0).sum())
        total_hits = int(used.sum())
        # Top-5 overuse table
        if used.size > 0:
            top5_idx = belong[np.argsort(-used)[:5]]
            top5_counts = used[np.argsort(-used)[:5]]
        else:
            top5_idx = np.array([], dtype=int)
            top5_counts = np.array([], dtype=int)
        gini = gini_from_counts(used) if used.size > 0 else 0.0
        coverage_summary.append({
            "class": classes[c],
            "unique_protos_used": unique_protos,
            "total_hits_on_class_protos": total_hits,
            "gini_overuse": gini,
            "top5_proto_indices": top5_idx.tolist(),
            "top5_counts": top5_counts.astype(int).tolist(),
        })

    # Gradient-share proxy
    grad_summary = {}
    if head_grad is not None:
        # Check if it's actually None (stored as object array)
        if isinstance(head_grad, np.ndarray) and head_grad.dtype == object:
            if head_grad.item() is None:
                head_grad = None
        if head_grad is not None:
            head_grad_arr = np.asarray(head_grad, dtype=np.float64)
            # Handle case where head_grad might be None or empty
            if head_grad_arr.size > 0 and np.isfinite(head_grad_arr).any():
                grad_summary["head_grad_l2_per_class"] = head_grad_arr.tolist()
                # normalize share
                grad_sum = float(head_grad_arr.sum())
                s = grad_sum if grad_sum > 0 else 1.0
                grad_summary["head_grad_share_per_class"] = (head_grad_arr / s).tolist()

    if proto_grad is not None:
        # Check if it's actually None (stored as object array)
        if isinstance(proto_grad, np.ndarray) and proto_grad.dtype == object:
            if proto_grad.item() is None:
                proto_grad = None
        if proto_grad is not None:
            proto_grad_arr = np.asarray(proto_grad, dtype=np.float64)
            if proto_grad_arr.size > 0:
                per_class_proto = []
                for c in range(C):
                    belong = np.where(proto_to_class == c)[0]
                    if belong.size > 0 and belong.size <= proto_grad_arr.size:
                        g = proto_grad_arr[belong]
                        total = float(g.sum())
                        per_class_proto.append({
                            "class": classes[c],
                            "proto_grad_sum": total,
                            "proto_grad_mean": float(g.mean()) if g.size > 0 else 0.0
                        })
                    else:
                        per_class_proto.append({
                            "class": classes[c],
                            "proto_grad_sum": 0.0,
                            "proto_grad_mean": 0.0
                        })
                grad_summary["proto_grad_by_class"] = per_class_proto

    # Write reports
    Path(args.outdir, "per_class_table.json").write_text(json.dumps(per_class, indent=2))
    Path(args.outdir, "coverage_summary.json").write_text(json.dumps(coverage_summary, indent=2))
    Path(args.outdir, "grad_summary.json").write_text(json.dumps(grad_summary, indent=2))
    Path(args.outdir, "macro_micro.json").write_text(json.dumps({
        "macro_auc": macro_auc, "macro_pr_auc": macro_ap,
        "micro_auc": micro_auc, "micro_pr_auc": micro_ap
    }, indent=2))

    # Pretty print quick view
    print("\n" + "=" * 80)
    print("=== Per-class (pos, AUROC, PR-AUC) ===")
    print("=" * 80)
    for r in per_class:
        flag = " [LOW_SUPPORT]" if r["low_support"] else ""
        auc_str = f"{r['auc']:.4f}" if r['auc'] is not None else "None"
        pr_str = f"{r['pr_auc']:.4f}" if r['pr_auc'] is not None else "None"
        print(f"{r['idx']:2d} {r['class'][:30]:30s}  pos={r['pos']:5d}  AUC={auc_str:>8}  PR={pr_str:>8}{flag}")

    print("\n" + "=" * 80)
    print("=== Macro/Micro Summary ===")
    print("=" * 80)
    print(f"Macro AUC:  {macro_auc:.4f}" if macro_auc else "Macro AUC:  None")
    print(f"Macro PR-AUC: {macro_ap:.4f}" if macro_ap else "Macro PR-AUC: None")
    print(f"Micro AUC:  {micro_auc:.4f}" if micro_auc else "Micro AUC:  None")
    print(f"Micro PR-AUC: {micro_ap:.4f}" if micro_ap else "Micro PR-AUC: None")

    print("\n" + "=" * 80)
    print("=== Coverage (unique protos used | Gini overuse | top5) ===")
    print("=" * 80)
    for r in coverage_summary:
        top5_str = str(list(zip(r['top5_proto_indices'], r['top5_counts'])))[:60]
        print(f"{r['class'][:30]:30s}  uniq={r['unique_protos_used']:2d}  gini={r['gini_overuse']:.3f}  top5={top5_str}")

    if grad_summary:
        print("\n" + "=" * 80)
        print("=== Gradient share (head rows) — higher means class drives more updates ===")
        print("=" * 80)
        if "head_grad_share_per_class" in grad_summary:
            shares = grad_summary["head_grad_share_per_class"]
            for i, cls in enumerate(classes):
                print(f"{i:2d} {cls[:30]:30s}  share={shares[i]:.4f}")

    print(f"\n[OK] wrote reports to {args.outdir}")


if __name__ == "__main__":
    main()

