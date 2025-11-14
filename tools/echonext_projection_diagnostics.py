#!/usr/bin/env python3
"""Diagnostics for ProtoECGNet cat1 projection onto EchoNext (NN vs Unique)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoNext projection diagnostics for ProtoECGNet cat1.")
    parser.add_argument("--ptb-ckpt", type=Path, required=True, help="ProtoECGNet cat1 checkpoint.")
    parser.add_argument("--projected-pt-nn", type=Path, required=True, help="NN projected prototypes tensor (.pt).")
    parser.add_argument("--projected-pt-uni", type=Path, required=True, help="Unique projected prototypes tensor (.pt).")
    parser.add_argument("--proto-labels-csv", type=Path, required=True, help="Proto→class CSV (prototype_index,class_id).")
    parser.add_argument("--metadata-nn", type=Path, required=True, help="NN metadata JSON.")
    parser.add_argument("--metadata-uni", type=Path, required=True, help="Unique metadata JSON.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--commit-sha", default="UNKNOWN")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for quick smoke tests.")
    return parser.parse_args()


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_proto_labels(csv_path: Path) -> np.ndarray:
    labels = np.zeros(80, dtype=int)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[int(row["prototype_index"])] = int(row["class_id"])
    return labels


def load_unprojected(ckpt_path: Path) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    return state["model.prototype_vectors"].detach().float().cpu()


def cosine_matrix(x: torch.Tensor) -> np.ndarray:
    norm = torch.nn.functional.normalize(x, dim=1)
    return (norm @ norm.T).cpu().numpy()


def plot_heatmap(matrix: np.ndarray, labels: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="viridis", vmin=-1.0, vmax=1.0)
    for tick in range(0, matrix.shape[0] + 1, 5):
        ax.axhline(tick - 0.5, color="white", alpha=0.15, linewidth=0.5)
        ax.axvline(tick - 0.5, color="white", alpha=0.15, linewidth=0.5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pca(tensor: torch.Tensor, labels: np.ndarray, out_path: Path, title: str) -> None:
    data = tensor.cpu().numpy()
    data = data - data.mean(axis=0, keepdims=True)
    comp = PCA(n_components=2, random_state=42).fit_transform(data)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    scatter = ax.scatter(comp[:, 0], comp[:, 1], c=labels, s=30, cmap="tab20")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cosine_delta_stats(unproj: torch.Tensor, proj: torch.Tensor, out_path: Path, title: str) -> Dict[str, float]:
    cu = cosine_matrix(unproj)
    cp = cosine_matrix(proj)
    iu = np.triu_indices_from(cu, k=1)
    delta = np.abs(cp[iu] - cu[iu])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(delta, bins=40, color="#2c7fb8")
    ax.set_title(title)
    ax.set_xlabel("|Δ cosine| (upper triangle)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return {"max": float(delta.max()), "mean": float(delta.mean()), "count": int(delta.size), "gt_0.1": int(np.sum(delta > 0.1))}


def reuse_counts(meta_path: Path) -> Dict[str, int]:
    data = json.loads(meta_path.read_text())
    counts: Dict[str, int] = {}
    for entry in data:
        ecg_id = entry["ecg_id"]
        counts[ecg_id] = counts.get(ecg_id, 0) + 1
    return counts


def main() -> None:
    args = parse_args()
    ensure_out_dir(args.out_dir)

    labels = load_proto_labels(args.proto_labels_csv)
    unprojected = load_unprojected(args.ptb_ckpt)
    proj_nn = torch.load(args.projected_pt_nn, map_location="cpu").float()
    proj_uni = torch.load(args.projected_pt_uni, map_location="cpu").float()

    plot_heatmap(cosine_matrix(unprojected), labels, args.out_dir / "echonext_cat1_unprojected_cosine_heatmap.png", "Unprojected (cat1.ckpt)")
    plot_heatmap(cosine_matrix(proj_nn), labels, args.out_dir / "echonext_cat1_projected_nn_cosine_heatmap.png", "EchoNext projected (NN)")
    plot_heatmap(cosine_matrix(proj_uni), labels, args.out_dir / "echonext_cat1_projected_unique_cosine_heatmap.png", "EchoNext projected (Unique)")

    plot_pca(unprojected, labels, args.out_dir / "echonext_cat1_unprojected_pca.png", "PCA — Unprojected")
    plot_pca(proj_nn, labels, args.out_dir / "echonext_cat1_projected_nn_pca.png", "PCA — Projected (NN)")
    plot_pca(proj_uni, labels, args.out_dir / "echonext_cat1_projected_unique_pca.png", "PCA — Projected (Unique)")

    stats_nn = cosine_delta_stats(unprojected, proj_nn, args.out_dir / "echonext_cat1_cosine_delta_histogram_nn.png", "|Δcos| Unprojected → NN")
    stats_uni = cosine_delta_stats(unprojected, proj_uni, args.out_dir / "echonext_cat1_cosine_delta_histogram_unique.png", "|Δcos| Unprojected → Unique")

    reuse_nn = reuse_counts(args.metadata_nn)
    reuse_uni = reuse_counts(args.metadata_uni)
    (args.out_dir / "echonext_cat1_nn_reuse.json").write_text(json.dumps(reuse_nn, indent=2))
    (args.out_dir / "echonext_cat1_unique_reuse.json").write_text(json.dumps(reuse_uni, indent=2))

    manifest = {
        "ckpt": str(args.ptb_ckpt.resolve()),
        "projected_nn": str(args.projected_pt_nn.resolve()),
        "projected_unique": str(args.projected_pt_uni.resolve()),
        "proto_labels_csv": str(args.proto_labels_csv.resolve()),
        "metadata_nn": str(args.metadata_nn.resolve()),
        "metadata_unique": str(args.metadata_uni.resolve()),
        "commit_sha": args.commit_sha,
        "limit": args.limit,
        "delta_stats": {"nn": stats_nn, "unique": stats_uni},
        "reuse": {
            "nn_unique_ecgs": len(reuse_nn),
            "unique_unique_ecgs": len(reuse_uni),
        },
    }
    (args.out_dir / "echonext_cat1_diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("[OK] EchoNext diagnostics generated (heatmaps, PCA, Δcos, reuse).")


if __name__ == "__main__":
    main()
