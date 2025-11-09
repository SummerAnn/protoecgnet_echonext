#!/usr/bin/env python3
"""
Run a quick sanity check training loop on the EchoNext data using the author's main.py.

The script keeps everything minimal:
    * sets environment variables so ecg_utils.py picks up the EchoNext numpy arrays
    * limits the sample count to keep the run light enough for verification
    * writes outputs one directory above the repo under protoecgnet_results/
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path("/opt/gpudata/summereunann")
PROCESSED_DIR = PROJECT_ROOT / "echonext_suite" / "processed"
LABELS_DIR = PROJECT_ROOT / "echonext_suite" / "echonext_suite" / "data"
RESULT_ROOT = PROJECT_ROOT / "protoecgnet_results"

JOB_NAME = "echonext_sanity_joint"


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "checkpoints").mkdir(exist_ok=True)
    (RESULT_ROOT / "logs").mkdir(exist_ok=True)
    (RESULT_ROOT / "test_results").mkdir(exist_ok=True)

    env = os.environ.copy()
    env.setdefault("ECHONEXT_PROCESSED_DIR", str(PROCESSED_DIR))
    env.setdefault("ECHONEXT_LABELS_DIR", str(LABELS_DIR))
    env.setdefault("ECHONEXT_TASK_MODE", "multilabel")
    env.setdefault("ECHONEXT_LIMIT_SAMPLES", "512")
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")

    cmd = [
        sys.executable,
        "src/main.py",
        "--job_name", JOB_NAME,
        "--training_stage", "joint",
        "--dimension", "1D",
        "--sampling_rate", "500",
        "--backbone", "resnet1d18",
        "--single_class_prototype_per_class", "5",
        "--proto_dim", "512",
        "--epochs", "1",
        "--batch_size", "16",
        "--lr", "0.0003",
        "--l2", "0.0",
        "--label_set", "all",
        "--num_workers", "0",
        "--checkpoint_dir", str(RESULT_ROOT / "checkpoints"),
        "--log_dir", str(RESULT_ROOT / "logs"),
        "--test_dir", str(RESULT_ROOT / "test_results"),
        "--save_weights", "False",
        "--save_top_k", "1",
        "--patience", "2",
    ]

    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)

    job_dir = (RESULT_ROOT / "test_results" / JOB_NAME).resolve()
    csv_candidates = sorted(job_dir.glob(f"{JOB_NAME}_test_results_v*.csv"))
    if not csv_candidates:
        print(f"[metrics] no CSV found under {job_dir}, skipping metric computation.")
        return

    latest_csv = csv_candidates[-1]
    metrics_json = job_dir / f"{JOB_NAME}_metrics.json"
    per_label_csv = job_dir / f"{JOB_NAME}_metrics_per_label.csv"

    metrics_cmd = [
        sys.executable,
        "scripts/compute_metrics_from_csv.py",
        "--csv", str(latest_csv),
        "--mode", env.get("ECHONEXT_TASK_MODE", "multilabel"),
        "--output-json", str(metrics_json),
        "--per-label-csv", str(per_label_csv),
    ]
    print(f"[metrics] computing metrics for {latest_csv.name}")
    subprocess.run(metrics_cmd, cwd=REPO_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()

