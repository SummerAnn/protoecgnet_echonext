#!/usr/bin/env python3
"""Rebuild EchoNext projections (NN / Unique) for ProtoECGNet cat1."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

try:
    from scipy.optimize import linear_sum_assignment

    SCIPY_AVAILABLE = True
except Exception:  # pragma: no cover - SciPy optional
    SCIPY_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild EchoNext projection for ProtoECGNet cat1.")
    parser.add_argument("--ckpt", type=Path, required=True, help="ProtoECGNet cat1 checkpoint (Lightning).")
    parser.add_argument("--dataset-root", type=Path, required=True, help="EchoNext dataset root (v1.0.0).")
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train", help="EchoNext split to use (default: train)."
    )
    parser.add_argument(
        "--encoder-ts",
        type=Path,
        required=True,
        help="TorchScript encoder exported from cat1 (e.g., ptbxl_weights/cat1.encoder.ts).",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=Path,
        default=None,
        help="Optional .npz cache containing {'embeddings','ecg_ids'} to skip recompute.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for projection artifacts.")
    parser.add_argument("--label-set", default="cat1", choices=["cat1"], help="Prototype label set (fixed).")
    parser.add_argument("--mode", choices=["nn", "unique"], default="nn", help="Projection strategy.")
    parser.add_argument("--limit", type=int, default=0, help="Cap EchoNext samples (0 = all).")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prototypes(ckpt_path: Path) -> Tuple[Tensor, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if "model.prototype_vectors" not in state:
        raise KeyError("Checkpoint missing 'model.prototype_vectors'.")
    prototypes = state["model.prototype_vectors"].detach().float().cpu()

    proto_labels = None
    if "model.prototype_class_identity" in state:
        pci = state["model.prototype_class_identity"].float()
        proto_labels = torch.argmax(pci, dim=1).cpu().numpy()
    elif "prototype_class_identity" in ckpt:
        pci = ckpt["prototype_class_identity"]
        if isinstance(pci, torch.Tensor):
            proto_labels = torch.argmax(pci, dim=1).cpu().numpy()

    if proto_labels is None:
        proto_labels = np.arange(prototypes.shape[0]) // 5
    return prototypes, proto_labels


def load_metadata_ids(dataset_root: Path, split: str) -> List[str]:
    meta_path = dataset_root / "EchoNext_metadata_100k.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata CSV at {meta_path}")
    df = pd.read_csv(meta_path, usecols=["ecg_key", "split"])
    split_df = df[df["split"] == split].reset_index(drop=True)
    return split_df["ecg_key"].astype(str).tolist()


def waveforms_path(dataset_root: Path, split: str) -> Path:
    path = dataset_root / f"EchoNext_{split}_waveforms.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing EchoNext waveforms file at {path}")
    return path


def build_embeddings(
    wave_path: Path,
    encoder_ts: Path,
    ids: List[str],
    batch_size: int,
    device: str,
    limit: int,
) -> np.ndarray:
    waves = np.load(wave_path, mmap_mode="r")
    total = waves.shape[0]
    if total != len(ids):
        raise ValueError(f"Waveforms ({total}) and metadata IDs ({len(ids)}) mismatch.")
    if limit > 0:
        total = min(total, limit)
    encoder = torch.jit.load(encoder_ts, map_location=device)
    encoder.eval()

    embeddings = []
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            chunk = waves[start:end]  # (B,1,T,12)
            if chunk.ndim != 4 or chunk.shape[1] != 1:
                raise RuntimeError(f"Unexpected waveform shape {chunk.shape}")
            chunk = np.transpose(chunk[:, 0, :, :], (0, 2, 1))  # -> (B,12,T)
            tensor = torch.from_numpy(np.ascontiguousarray(chunk)).float().to(device)
            emb = encoder(tensor)
            if emb.ndim != 2:
                raise RuntimeError(f"Encoder output must be (B,D); got {emb.shape}")
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def load_or_build_embeddings(args: argparse.Namespace, ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    cache_path = args.embeddings_cache
    if cache_path and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        embeddings = cached["embeddings"]
        cached_ids = cached["ecg_ids"].astype(str).tolist()
        if len(cached_ids) != len(ids):
            print("[WARN] Cached ecg_ids length differs from metadata; using cached order.", file=sys.stderr)
            ids = cached_ids
        if args.limit > 0:
            embeddings = embeddings[: args.limit]
            ids = ids[: args.limit]
        return embeddings, ids

    wav_path = waveforms_path(args.dataset_root, args.split)
    embeddings = build_embeddings(
        wav_path,
        args.encoder_ts,
        ids,
        args.batch_size,
        args.device,
        args.limit,
    )
    if args.limit > 0:
        ids = ids[: embeddings.shape[0]]
    if cache_path:
        np.savez(cache_path, embeddings=embeddings, ecg_ids=np.asarray(ids, dtype=str))
    return embeddings, ids


def normalize(vecs: Tensor) -> Tensor:
    return torch.nn.functional.normalize(vecs, dim=1)


def select_nn(protos: Tensor, pool: Tensor) -> Tuple[np.ndarray, np.ndarray]:
    sims = normalize(protos) @ normalize(pool).T
    sims_np = sims.cpu().numpy()
    idx = sims_np.argmax(axis=1)
    scores = sims_np[np.arange(sims_np.shape[0]), idx]
    return idx, scores


def select_unique(protos: Tensor, pool: Tensor) -> Tuple[np.ndarray, np.ndarray, str]:
    sims = normalize(protos) @ normalize(pool).T
    sims_np = sims.cpu().numpy()
    cost = 1.0 - sims_np
    if SCIPY_AVAILABLE and sims_np.shape[0] <= sims_np.shape[1]:
        r, c = linear_sum_assignment(cost)
        order = np.argsort(r)
        idx = c[order]
        method = "hungarian"
    else:
        method = "greedy_fallback"
        n_proto = sims_np.shape[0]
        idx = -np.ones(n_proto, dtype=np.int64)
        used = set()
        proto_order = np.argsort(-sims_np.max(axis=1))
        for proto_idx in proto_order:
            ranked = np.argsort(-sims_np[proto_idx])
            chosen = None
            for candidate in ranked:
                if candidate not in used:
                    chosen = candidate
                    break
            if chosen is None:
                chosen = int(ranked[0])
            idx[proto_idx] = chosen
            used.add(chosen)
    scores = sims_np[np.arange(sims_np.shape[0]), idx]
    return idx, scores, method


def write_proto_labels_csv(path: Path, class_ids: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prototype_index", "class_id"])
        for proto_idx, class_id in enumerate(class_ids):
            writer.writerow([proto_idx, int(class_id)])


def save_projection(
    out_dir: Path,
    mode: str,
    projected: Tensor,
    selection_idx: np.ndarray,
    scores: np.ndarray,
    ecg_ids: List[str],
    class_ids: np.ndarray,
    selection_method: str,
) -> None:
    torch.save(projected.cpu(), out_dir / f"echonext_{mode}_projected_prototypes.pt")

    meta = []
    for proto_idx, pool_idx in enumerate(selection_idx):
        meta.append(
            {
                "prototype_index": proto_idx,
                "class_id": int(class_ids[proto_idx]),
                "ecg_id": ecg_ids[int(pool_idx)],
                "pool_index": int(pool_idx),
                "cosine": float(scores[proto_idx]),
                "selection_method": selection_method,
            }
        )
    (out_dir / f"echonext_{mode}_prototype_metadata.json").write_text(json.dumps(meta, indent=2))

    labels_csv = out_dir / (
        "echonext_nn_proto_labels.csv" if mode == "nn" else "echonext_unique_proto_labels.csv"
    )
    write_proto_labels_csv(labels_csv, class_ids)


def append_manifest(manifest_path: Path, entry: Dict[str, object]) -> None:
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {}
    selection_modes = set(manifest.get("selection_modes", []))
    selection_modes.add(entry["mode"])
    manifest["selection_modes"] = sorted(selection_modes)
    manifest.update(entry)
    manifest_path.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prototypes, proto_labels = load_prototypes(args.ckpt)
    ecg_ids = load_metadata_ids(args.dataset_root, args.split)
    embeddings, embedding_ids = load_or_build_embeddings(args, ecg_ids)
    if len(embedding_ids) != embeddings.shape[0]:
        raise ValueError("Embedding count mismatch with ecg_ids.")

    pool = torch.from_numpy(embeddings).float()
    if args.mode == "nn":
        choice, scores = select_nn(prototypes, pool)
        selection_method = "nearest_neighbor"
    else:
        choice, scores, selection_method = select_unique(prototypes, pool)
    projected = pool[choice]

    save_projection(args.out_dir, args.mode, projected, choice, scores, embedding_ids, proto_labels, selection_method)

    manifest_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ckpt": str(args.ckpt.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "label_set": args.label_set,
        "mode": args.mode,
        "selection_method": selection_method,
        "limit": args.limit,
        "seed": args.seed,
        "device": args.device,
        "embeddings_count": int(embeddings.shape[0]),
        "scipy_available": SCIPY_AVAILABLE,
        "paths": {
            "projected_pt": str((args.out_dir / f"echonext_{args.mode}_projected_prototypes.pt").resolve()),
            "metadata": str((args.out_dir / f"echonext_{args.mode}_prototype_metadata.json").resolve()),
        },
    }
    append_manifest(args.out_dir / "echonext_cat1_rebuild_manifest.json", manifest_entry)
    print(f"[OK] EchoNext projection ({args.mode}) complete. Method={selection_method}.")


if __name__ == "__main__":
    main()
