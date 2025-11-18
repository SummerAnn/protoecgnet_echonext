#!/bin/bash
# Hyperparameter tuning script for EchoNext Stage B (30 epochs, full dataset, GPU 2)

cd /opt/gpudata/summereunann/protoecgnet

# Configuration
GPU_ID=2
CKPT="/opt/gpudata/summereunann/ptbxl_weights/cat1.ckpt"
BASE_OUT_DIR="results/hyperparameter_tuning_30ep"
MODE="grid"  # or "random" for random search
N_TRIALS=20  # Only used for random search mode

echo "=========================================="
echo "Hyperparameter Tuning: EchoNext Stage B"
echo "=========================================="
echo "GPU: $GPU_ID"
echo "Epochs: 30 (full run)"
echo "Mode: $MODE"
echo "Output: $BASE_OUT_DIR"
echo "=========================================="
echo ""

# Dry run first to show what will be executed
echo "Dry run (showing first 5 configurations)..."
python3 tuning.py \
  --mode "$MODE" \
  --n_trials "$N_TRIALS" \
  --gpu_id "$GPU_ID" \
  --ckpt "$CKPT" \
  --base_out_dir "$BASE_OUT_DIR" \
  --dry_run \
  --start_trial 0 | head -100

echo ""
echo "=========================================="
echo "Ready to run? This will execute all trials."
echo "Total configurations in grid mode: 128 (4 lr × 4 lambda_cluster × 4 lambda_separation × 2 batch_size × 2 scheduler)"
echo "=========================================="
echo ""
read -p "Continue with full tuning? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "Starting hyperparameter tuning..."
  python3 tuning.py \
    --mode "$MODE" \
    --n_trials "$N_TRIALS" \
    --gpu_id "$GPU_ID" \
    --ckpt "$CKPT" \
    --base_out_dir "$BASE_OUT_DIR" \
    2>&1 | tee "$BASE_OUT_DIR/tuning.log"
  
  echo ""
  echo "=========================================="
  echo "Tuning complete! Results saved to:"
  echo "$BASE_OUT_DIR/results.json"
  echo "=========================================="
else
  echo "Cancelled."
fi

