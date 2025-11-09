#!/usr/bin/env python3
"""
Compute ProtoECGNet evaluation metrics (AUROC/AUPRC) from a saved CSV.

The CSV is expected to contain pairs of columns with prefixes like
`Label_*` and `Prob_*`. The script will derive per-label and aggregate
metrics using the helper routines defined in `scripts/metrics.py`.
"""
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports from the repository
# ---------------------------------------------------------------------------
try:
    from metrics import compute_metrics_dict
except ImportError:  # pragma: no cover - defensive
    import sys

    SCRIPT_DIR = Path(__file__).resolve().parent
    candidates = [
        SCRIPT_DIR,
        SCRIPT_DIR.parent,
        SCRIPT_DIR.parent.parent,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        candidate_str = str(candidate)
        if candidate_str and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    from metrics import compute_metrics_dict  # type: ignore


def _extract_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return label/prob arrays (num_examples x num_labels) and label names."""
    label_cols = [c for c in df.columns if c.lower().startswith("label_")]
    prob_cols = [c for c in df.columns if c.lower().startswith("prob_")]

    if label_cols and prob_cols and len(label_cols) == len(prob_cols):
        label_cols = sorted(label_cols)
        prob_cols = sorted(prob_cols)
        y_true = df[label_cols].to_numpy(dtype=float)
        y_prob = df[prob_cols].to_numpy(dtype=float)
        names = [col[len("Label_") :] for col in label_cols]
        return y_true, y_prob, names

    # Fallback to generic naming (binary-style outputs)
    for truth_key in ("y_true", "label", "target", "gt"):
        if truth_key in df.columns:
            y_true = df[truth_key].to_numpy(dtype=float).reshape(-1, 1)
            break
    else:
        raise ValueError("Could not identify label columns in CSV.")

    for prob_key in ("y_prob", "prob", "pred_prob", "prediction", "score", "logit"):
        if prob_key in df.columns:
            y_prob = df[prob_key].to_numpy(dtype=float).reshape(-1, 1)
            break
    else:
        raise ValueError("Could not identify probability columns in CSV.")

    return y_true, y_prob, ["output"]


def _infer_mode(requested_mode: str, num_labels: int) -> str:
    if requested_mode:
        return requested_mode.lower()
    return "binary" if num_labels == 1 else "multilabel"


def _build_output_dict(mode: str, metrics_dict: dict, label_names: List[str]) -> dict:
    """Assemble a JSON-serialisable dictionary for output."""
    result = {
        "mode": mode,
        "macro_auroc": metrics_dict.get("macro_auroc"),
        "macro_auprc": metrics_dict.get("macro_auprc"),
        "micro_auroc": metrics_dict.get("micro_auroc"),
        "micro_auprc": metrics_dict.get("micro_auprc"),
    }

    if mode == "binary":
        # binary_metrics schema
        result.update(
            {
                "auroc": metrics_dict.get("auroc"),
                "auprc": metrics_dict.get("auprc"),
                "best_f1": metrics_dict.get("best_f1"),
                "best_threshold": metrics_dict.get("best_thr"),
            }
        )
        per_label = [
            {
                "label": label_names[0],
                "auroc": metrics_dict.get("auroc"),
                "auprc": metrics_dict.get("auprc"),
            }
        ]
    else:
        per_label_stats = metrics_dict.get("per_label", [])
        per_label = []
        for name, stats in zip(label_names, per_label_stats):
            per_label.append(
                {
                    "label": name,
                    "auroc": stats.get("auroc") if stats else None,
                    "auprc": stats.get("auprc") if stats else None,
                }
            )
    result["per_label"] = per_label
    return result


def _write_per_label_csv(per_label: List[dict], destination: Path) -> None:
    if not per_label:
        return
    df = pd.DataFrame(per_label)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AUROC/AUPRC metrics from a test CSV.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to CSV file with test predictions.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["binary", "multilabel", "multitask"],
        default=None,
        help="Override task mode; defaults to 'binary' if one label else 'multilabel'.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the aggregate metrics JSON.",
    )
    parser.add_argument(
        "--per-label-csv",
        type=Path,
        default=None,
        help="Optional CSV path for per-label AUROC/AUPRC.",
    )
    parser.add_argument(
        "--label-name-prefix",
        type=str,
        default="class_",
        help="Prefix for generated label names when none are present.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    y_true, y_prob, inferred_names = _extract_arrays(df)

    if not inferred_names:
        inferred_names = [f"{args.label_name_prefix}{idx}" for idx in range(y_true.shape[1])]
    mode = _infer_mode(args.mode, y_true.shape[1])
    metrics_dict = compute_metrics_dict(mode, y_true, y_prob)

    output = _build_output_dict(mode, metrics_dict, inferred_names)
    output["num_examples"] = int(y_true.shape[0])
    output["num_labels"] = int(y_true.shape[1])
    output["csv_path"] = str(args.csv)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)

    if args.per_label_csv:
        _write_per_label_csv(output["per_label"], args.per_label_csv)


if __name__ == "__main__":
    main()

