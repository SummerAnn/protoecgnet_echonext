#!/usr/bin/env python3
"""
Launch grid sweeps of ProtoECGNet configurations on the EchoNext dataset.

The script iterates over modes, backbones, prototype counts, and embedding
dimensions, running `src/main.py` for each combination. After each run it
invokes `scripts/compute_metrics_from_csv.py` to materialise AUROC/AUPRC
summaries.
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_MODES = ["binary", "multilabel", "multitask"]
DEFAULT_BACKBONES = [
    "simple_cnn",
    "cnn_gru",
    "cnn_lstm",
    "cnn_transformer",
    "inception_cnn",
    "deep_cnn",
    "wide_cnn",
    "resnet1d18",
    "resnet1d34",
    "resnet1d50",
    "resnet1d101",
    "resnet1d152",
]
DEFAULT_PROTOTYPES_PER_CLASS = [5, 10, 15, 20, 30]
DEFAULT_PROTO_DIMS = [256, 512]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoNext sweep runner")
    parser.add_argument(
        "--modes", nargs="+", default=DEFAULT_MODES,
        help="Task modes to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--backbones", nargs="+", default=DEFAULT_BACKBONES,
        help="Model backbones to evaluate.",
    )
    parser.add_argument(
        "--prototypes-per-class", nargs="+", type=int,
        default=DEFAULT_PROTOTYPES_PER_CLASS,
        help="Prototype counts per class to sweep.",
    )
    parser.add_argument(
        "--proto-dims", nargs="+", type=int,
        default=DEFAULT_PROTO_DIMS,
        help="Prototype embedding dimensions to sweep.",
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Number of epochs for each joint run.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate to use for training.",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="L2 weight decay term.",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=500,
        help="Sampling rate of the EchoNext waveforms.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--label-set", type=str, default="all",
        help="Label set to use from ecg_utils.load_label_mappings.",
    )
    parser.add_argument(
        "--processed-dir", type=Path,
        default=Path("/opt/gpudata/summereunann/echonext_suite/processed"),
        help="Directory containing processed EchoNext arrays.",
    )
    parser.add_argument(
        "--labels-dir", type=Path,
        default=Path("/opt/gpudata/summereunann/echonext_suite/echonext_suite/data"),
        help="Directory containing label numpy files (for binary/multitask).",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("/opt/gpudata/summereunann/protoecgnet_results/checkpoints"),
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=Path("/opt/gpudata/summereunann/protoecgnet_results/logs"),
    )
    parser.add_argument(
        "--test-dir", type=Path,
        default=Path("/opt/gpudata/summereunann/protoecgnet_results/test_results"),
    )
    parser.add_argument(
        "--results-root", type=Path,
        default=Path("/opt/gpudata/summereunann/protoecgnet_results/test_results"),
        help="Base directory where CSV/JSON metrics are stored.",
    )
    parser.add_argument(
        "--multitask-columns", type=str, default="",
        help="Comma-separated column indices to keep for multitask mode.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip combinations that already have a metrics JSON.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional limit on the number of combinations to execute.",
    )
    parser.add_argument(
        "--python-executable", type=str, default=sys.executable,
        help="Python interpreter to use when launching training jobs.",
    )
    parser.add_argument(
        "--training-stage", type=str, default="joint",
        help="Training stage to run (default: joint).",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--save-weights", action="store_true",
        help="Persist best checkpoint weights for each run.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device string to pass through (cuda or cpu).",
    )
    parser.add_argument(
        "--proto-time-len", type=int, default=3,
        help="Length of prototype temporal dimension.",
    )
    return parser.parse_args()


def generate_combinations(
    modes: Sequence[str],
    backbones: Sequence[str],
    prototypes: Sequence[int],
    proto_dims: Sequence[int],
) -> Iterable[Tuple[str, str, int, int]]:
    return itertools.product(modes, backbones, prototypes, proto_dims)


def build_job_name(mode: str, backbone: str, ppc: int, proto_dim: int) -> str:
    return f"echonext_{mode}_{backbone}_ppc{ppc}_d{proto_dim}"


def metrics_exist(results_root: Path, job_name: str) -> bool:
    metrics_path = results_root / job_name / f"{job_name}_metrics.json"
    return metrics_path.exists()


def locate_latest_csv(job_dir: Path, job_name: str) -> Path | None:
    if not job_dir.exists():
        return None
    candidates = sorted(job_dir.glob(f"{job_name}_test_results_v*.csv"))
    return candidates[-1] if candidates else None


def run_subprocess(cmd: List[str], env: dict, dry_run: bool) -> int:
    if dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return 0
    print("[RUN]", " ".join(cmd))
    start = time.time()
    completed = subprocess.run(cmd, env=env)
    duration = time.time() - start
    print(f"[DONE] returncode={completed.returncode} duration={duration:.1f}s")
    return completed.returncode


def main() -> None:
    args = parse_args()

    combinations = list(generate_combinations(
        args.modes,
        args.backbones,
        args.prototypes_per_class,
        args.proto_dims,
    ))

    if args.limit is not None:
        combinations = combinations[: args.limit]

    multitask_columns = args.multitask_columns.replace(" ", "")
    env_base = os.environ.copy()
    env_base["ECHONEXT_PROCESSED_DIR"] = str(args.processed_dir)
    env_base["ECHONEXT_LABELS_DIR"] = str(args.labels_dir)
    env_base["ECHONEXT_SAMPLING_RATE"] = str(args.sampling_rate)

    results_root = args.results_root
    results_root.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.test_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(combinations)
    print(f"Planning {total_runs} combinations.")

    for idx, (mode, backbone, ppc, proto_dim) in enumerate(combinations, start=1):
        job_name = build_job_name(mode, backbone, ppc, proto_dim)
        print(f"\n[{idx}/{total_runs}] Job {job_name}")

        if args.skip_existing and metrics_exist(results_root, job_name):
            print("  Metrics already exist; skipping.")
            continue

        env = env_base.copy()
        env["ECHONEXT_TASK_MODE"] = mode
        if mode == "multitask" and multitask_columns:
            env["ECHONEXT_MULTITASK_COLUMNS"] = multitask_columns
        else:
            env.pop("ECHONEXT_MULTITASK_COLUMNS", None)

        train_cmd = [
            args.python_executable,
            "src/main.py",
            "--job_name", job_name,
            "--training_stage", args.training_stage,
            "--dimension", "1D",
            "--sampling_rate", str(args.sampling_rate),
            "--backbone", backbone,
            "--single_class_prototype_per_class", str(ppc),
            "--proto_dim", str(proto_dim),
            "--proto_time_len", str(args.proto_time_len),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--lr", str(args.learning_rate),
            "--l2", str(args.weight_decay),
            "--label_set", args.label_set,
            "--num_workers", str(args.num_workers),
            "--checkpoint_dir", str(args.checkpoint_dir),
            "--log_dir", str(args.log_dir),
            "--test_dir", str(args.test_dir),
            "--device", args.device,
            "--patience", str(args.patience),
        ]
        train_cmd.extend(["--save_weights", "True" if args.save_weights else "False"])

        returncode = run_subprocess(train_cmd, env, args.dry_run)
        if returncode != 0:
            print(f"[WARN] Training failed for {job_name} (exit {returncode})")
            continue

        job_test_dir = args.test_dir / job_name
        csv_path = locate_latest_csv(job_test_dir, job_name)
        if csv_path is None:
            print(f"[WARN] No CSV found for {job_name} under {job_test_dir}")
            continue

        metrics_json = results_root / job_name / f"{job_name}_metrics.json"
        per_label_csv = results_root / job_name / f"{job_name}_metrics_per_label.csv"
        metrics_json.parent.mkdir(parents=True, exist_ok=True)

        metrics_cmd = [
            args.python_executable,
            "scripts/compute_metrics_from_csv.py",
            "--csv", str(csv_path),
            "--mode", mode,
            "--output-json", str(metrics_json),
            "--per-label-csv", str(per_label_csv),
        ]
        returncode = run_subprocess(metrics_cmd, env, args.dry_run)
        if returncode != 0:
            print(f"[WARN] Metrics computation failed for {job_name}")
        else:
            print(f"[OK] Metrics stored at {metrics_json}")


if __name__ == "__main__":
    main()

