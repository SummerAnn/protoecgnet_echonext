#!/usr/bin/env python3
"""Inspect a ProtoECGNet PTB-XL cat1 checkpoint and export prototype→class mapping.

Example:
    python tools/inspect_ptbxl_checkpoint.py \
        --ckpt /opt/gpudata/summereunann/ptbxl_weights/cat1.ckpt \
        --output-json artifacts/ptbxl_cat1_prototype_index.json \
        --output-markdown artifacts/ptbxl_cat1_readme.md

This script is read-only; it does not modify the checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch


def _format_proto_mapping(class_identity: torch.Tensor) -> List[Dict[str, object]]:
    mapping = []
    for idx in range(class_identity.shape[0]):
        active = torch.nonzero(class_identity[idx] > 0.5, as_tuple=False).flatten().tolist()
        mapping.append({"prototype_index": idx, "active_classes": active})
    return mapping


def inspect_checkpoint(ckpt_path: Path) -> Dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    proto_vectors = state_dict["model.prototype_vectors"]
    proto_class_identity = state_dict["model.prototype_class_identity"].float()

    summary: Dict[str, object] = {
        "checkpoint_path": str(ckpt_path),
        "num_prototypes": int(proto_vectors.shape[0]),
        "num_classes": int(proto_class_identity.shape[1]),
        "prototype_class_identity": _format_proto_mapping(proto_class_identity),
    }

    hyper_params = ckpt.get("hyper_parameters")
    if hyper_params is not None:
        args = hyper_params.get("args")
        if args is not None:
            summary["training_args"] = {
                "label_set": getattr(args, "label_set", None),
                "custom_groups": getattr(args, "custom_groups", None),
                "single_class_prototype_per_class": getattr(args, "single_class_prototype_per_class", None),
            }
    return summary


def write_markdown(summary: Dict[str, object], output_path: Path) -> None:
    lines = [
        "# ProtoECGNet PTB-XL cat1 checkpoint",
        "",
        f"*Checkpoint path*: `{summary['checkpoint_path']}`",
        "",
        f"*Number of prototypes*: {summary['num_prototypes']}",
        f"*Number of classes*: {summary['num_classes']}",
        "",
        "| Prototype index | Active class IDs |",
        "| --- | --- |",
    ]
    for proto in summary["prototype_class_identity"]:
        class_list = ", ".join(map(str, proto["active_classes"])) or "(none)"
        lines.append(f"| {proto['prototype_index']} | {class_list} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to ProtoECGNet PTB-XL cat1 checkpoint (.ckpt)")
    parser.add_argument("--output-json", type=Path, required=True, help="Where to write the prototype mapping JSON")
    parser.add_argument("--output-markdown", type=Path, required=True, help="Where to write the Markdown summary")
    args = parser.parse_args()

    summary = inspect_checkpoint(args.ckpt)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    write_markdown(summary, args.output_markdown)

    print(f"[OK] Wrote JSON to {args.output_json}")
    print(f"[OK] Wrote Markdown to {args.output_markdown}")


if __name__ == "__main__":
    main()
