#!/usr/bin/env python3
"""
Thin wrapper around src/tune.py that configures the EchoNext processed data paths
and launches Optuna Hyperband tuning runs.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EchoNext Hyperband tuning wrapper")
    parser.add_argument("--job-name", required=True, help="Unique name for this tuning job.")
    parser.add_argument("--mode", choices=["binary", "multilabel", "multitask"], default="binary")
    parser.add_argument("--backbone", default="resnet1d18", help="Backbone to tune.")
    parser.add_argument("--training-stage", default="joint", help="Training stage (default: joint).")
    parser.add_argument("--dimension", default="1D", help="Model dimension (default: 1D).")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per trial.")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials.")
    parser.add_argument("--batch-size", type=int, default=128, help="Base batch size (subject to tuning).")
    parser.add_argument("--sampling-rate", type=int, default=500, help="Waveform sampling rate.")
    parser.add_argument("--label-set", type=str, default="all", help="Label set identifier.")
    parser.add_argument("--processed-dir", type=Path,
                        default=Path("/opt/gpudata/summereunann/echonext_suite/processed"))
    parser.add_argument("--labels-dir", type=Path,
                        default=Path("/opt/gpudata/summereunann/echonext_suite/echonext_suite/data"))
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=Path("/opt/gpudata/summereunann/protoecgnet_results/checkpoints"))
    parser.add_argument("--log-dir", type=Path,
                        default=Path("/opt/gpudata/summereunann/protoecgnet_results/logs"))
    parser.add_argument("--test-dir", type=Path,
                        default=Path("/opt/gpudata/summereunann/protoecgnet_results/test_results"))
    parser.add_argument("--study-dir", type=Path,
                        default=Path("/opt/gpudata/summereunann/protoecgnet_results/optuna_studies"))
    parser.add_argument("--num-workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing.")
    parser.add_argument("--multitask-columns", type=str, default="",
                        help="Comma-separated indices to retain for multitask mode.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--proto-dim", type=int, default=512)
    parser.add_argument("--proto-time-len", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = os.environ.copy()
    env["ECHONEXT_PROCESSED_DIR"] = str(args.processed_dir)
    env["ECHONEXT_LABELS_DIR"] = str(args.labels_dir)
    env["ECHONEXT_TASK_MODE"] = args.mode
    env["ECHONEXT_SAMPLING_RATE"] = str(args.sampling_rate)
    if args.mode == "multitask" and args.multitask_columns:
        env["ECHONEXT_MULTITASK_COLUMNS"] = args.multitask_columns.replace(" ", "")
    elif "ECHONEXT_MULTITASK_COLUMNS" in env:
        env.pop("ECHONEXT_MULTITASK_COLUMNS")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.test_dir.mkdir(parents=True, exist_ok=True)
    args.study_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_executable,
        "src/tune.py",
        "--job_name", args.job_name,
        "--training_stage", args.training_stage,
        "--dimension", args.dimension,
        "--backbone", args.backbone,
        "--epochs", str(args.epochs),
        "--n_trials", str(args.n_trials),
        "--batch_size", str(args.batch_size),
        "--sampling_rate", str(args.sampling_rate),
        "--label_set", args.label_set,
        "--num_workers", str(args.num_workers),
        "--checkpoint_dir", str(args.checkpoint_dir),
        "--log_dir", str(args.log_dir),
        "--test_dir", str(args.test_dir),
        "--study_dir", str(args.study_dir),
        "--device", args.device,
        "--proto_dim", str(args.proto_dim),
        "--proto_time_len", str(args.proto_time_len),
    ]

    if args.dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return

    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()

