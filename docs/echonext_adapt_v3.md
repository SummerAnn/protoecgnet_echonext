# EchoNext Adaptation Baseline → Reprojection

## Scope

This PR adds EchoNext adaptation baseline and reprojection diagnostics using minimal switches to the training code.

- **Does:**
  - Head-only baseline (encoder + prototypes frozen, train classifier on EchoNext labels)
  - Joint fine-tuning (prototypes + head trainable, encoder frozen)
  - Reprojection and diagnostics after fine-tuning
  - No PTB label fallback (EchoNext labels only)
  
- **Does not:**
  - Modify PTB-XL code paths
  - Include lead occlusion
  - Include pruning
  - Fine-tune encoder (optional follow-up only)

## How to Run

### A) Head-only Baseline (Diagnostic)

```bash
export CKPT=/opt/gpudata/summereunann/ptbxl_weights/cat1.ckpt
export ECHONEXT=/opt/gpudata/ecg/echonext-v1.0.0
export OUT=results/head_only

python -m src.main \
  --job_name echonext_head_only \
  --training_stage echonext_adapt \
  --dataset_root $ECHONEXT \
  --train_split train --val_split val \
  --ckpt $CKPT \
  --dimension 1D --backbone resnet1d18 \
  --label_set cat1 --custom_groups true \
  --single_class_prototype_per_class 5 --proto_dim 512 \
  --train_head_only true \
  --freeze_encoder true \
  --freeze_prototypes true \
  --lambda_cluster 0.0 --lambda_separation 0.0 \
  --batch_size 128 --epochs 5 --lr 3e-4 --seed 42 \
  --export_encoder_ts $OUT/cat1.encoder.ts \
  --out_dir $OUT \
  --checkpoint_dir $OUT/checkpoints \
  --log_dir $OUT/logs \
  --test_dir $OUT/test_results
```

### B) Joint Fine-tune (Prototypes + Head)

```bash
export CKPT_FT_IN=$CKPT  # Start from cat1.ckpt
export OUT=results/joint_ft

python -m src.main \
  --job_name echonext_joint_ft \
  --training_stage echonext_adapt \
  --dataset_root $ECHONEXT \
  --train_split train --val_split val \
  --ckpt $CKPT_FT_IN \
  --dimension 1D --backbone resnet1d18 \
  --label_set cat1 --custom_groups true \
  --single_class_prototype_per_class 5 --proto_dim 512 \
  --train_head_only false \
  --freeze_encoder true \
  --freeze_prototypes false \
  --lambda_cluster 0.8 --lambda_separation 0.1 \
  --topk_push 5 \
  --batch_size 64 --epochs 10 --lr 1e-4 --seed 42 \
  --early_stop_metric auroc_micro --early_stop_patience 3 \
  --export_encoder_ts $OUT/cat1.encoder.ts \
  --out_dir $OUT \
  --checkpoint_dir $OUT/checkpoints \
  --log_dir $OUT/logs \
  --test_dir $OUT/test_results
```

**Losses:**
- BCE over EchoNext multi-label targets (no PTB fallback)
- Cluster loss only for present classes in batch (multi-label: if sample has {c1, c2}, push toward those class prototypes)
- Separation applied to negatives
- If a prototype's class has no positives in batch → no cluster term for it in that batch

### C) Re-project + Re-diagnose (Required)

```bash
# Rebuild projections (NN + Unique) with the *fine-tuned* encoder/prototypes
export ENCODER=$OUT/cat1.encoder.ts
export REPROJ=$OUT/reprojection

python tools/rebuild_echonext_projection.py \
  --ckpt $OUT/checkpoints/best.ckpt \
  --dataset-root $ECHONEXT --split train \
  --encoder-ts $ENCODER --out-dir $REPROJ --mode nn --seed 42

python tools/rebuild_echonext_projection.py \
  --ckpt $OUT/checkpoints/best.ckpt \
  --dataset-root $ECHONEXT --split train \
  --encoder-ts $ENCODER --out-dir $REPROJ --mode unique --seed 42

# Diagnostics (reuse PR3 tool)
python tools/echonext_projection_diagnostics.py \
  --ptb-ckpt $OUT/checkpoints/best.ckpt \
  --projected-pt-nn  $REPROJ/echonext_nn_projected_prototypes.pt \
  --projected-pt-uni $REPROJ/echonext_unique_projected_prototypes.pt \
  --proto-labels-csv $REPROJ/echonext_nn_proto_labels.csv \
  --metadata-nn  $REPROJ/echonext_nn_prototype_metadata.json \
  --metadata-uni $REPROJ/echonext_unique_prototype_metadata.json \
  --out-dir $REPROJ/diagnostics --commit-sha $(git rev-parse --short HEAD)
```

## Before → After Comparison

**Metrics:**
- EchoNext val AUROC micro/macro from head-only vs joint-FT
- PR-AUC for both stages
- Unique ECG counts (NN & Unique modes)
- Mean/max |Δcos| histograms
- Heatmaps/PCA before and after fine-tuning

**Expected Improvements:**
- Joint FT should improve val AUROC vs head-only (or at least be numerically sane)
- Reprojection should show:
  - Unique mode: ~80 distinct ECGs (one-to-one assignment)
  - NN mode: may show collapse (fewer unique ECGs)
  - |Δcos| should decrease after fine-tuning (prototypes move closer to EchoNext embeddings)

## Files Changed

- `src/main.py`: Added flags (`--freeze_encoder`, `--freeze_prototypes`, `--train_head_only`, `--lambda_cluster`, `--lambda_separation`, `--topk_push`, `--export_encoder_ts`, `--early_stop_metric`, `--early_stop_patience`, `--dataset_root`, `--train_split`, `--val_split`, `--out_dir`, `--ckpt`)
- `src/data/echonext.py` (new): EchoNext dataset + loader (reads metadata CSV + waveforms NPY; multi-label vector)
- `src/model/export.py` (new): Export encoder to TorchScript utility
- `src/training_functions.py`: Updated `configure_optimizers` to respect freezing flags, added AUROC micro/macro and PR-AUC metrics, early stopping support
- `configs/echonext_head_only.yaml`: Head-only baseline config
- `configs/echonext_joint_ft.yaml`: Joint fine-tuning config
- `docs/echonext_adapt_v3.md`: This file

## Verification

- Head-only EchoNext baseline trains and logs metrics (no encoder/prototype updates)
- Joint FT updates prototypes (not encoder), improves val AUROC vs head-only or at least is numerically sane
- Reprojection/diagnostics after FT complete and produce before/after tables:
  - `unique_ecgs` (NN/Unique), `mean|Δcos|`, `max|Δcos|`, AUROC/PR-AUC
- No PTB code path changes; PR is self-contained to EchoNext adaptation + flags

## Troubleshooting

- **Missing EchoNext dataset assets** → Check `$ECHONEXT` path and ensure `EchoNext_metadata_100k.csv` and `EchoNext_{split}_waveforms.npy` exist
- **CUDA mismatch** → Re-run with `--device cpu --limit 200` for smoke tests
- **Memory-limited** → Use `--limit 200` in dataloader for smoke runs, then rerun with full data
- **Different EchoNext split** → Pass `--train_split val` or `--train_split test` to probe other cohorts
- **Hungarian unavailable** → Tool falls back to greedy; manifest records `selection_method`
- **Class imbalance** → Per-class BCE weights are computed automatically (effective-number weighting)

## Acceptance Criteria

- Head-only EchoNext baseline trains and logs metrics (no encoder/prototype updates)
- Joint FT updates prototypes (not encoder), improves val AUROC vs head-only or at least is numerically sane
- Reprojection/diagnostics after FT complete and produce before/after tables:
  - `unique_ecgs` (NN/Unique), `mean|Δcos|`, `max|Δcos|`, AUROC/PR-AUC
- No PTB code path changes; PR is self-contained to EchoNext adaptation + flags

## Design Requirements & Rationale

### Core Principle

**Freeze the encoder, fine-tune prototypes + head on EchoNext using EchoNext labels only, then re-project onto EchoNext and compare NN vs Unique to show collapse reduction. No PTB label fallback, no prototype relabeling from nearest samples.**

### Key Decisions

1. **No PTB Label Fallback:** Uses EchoNext labels directly. Proto→class inferred from classifier weights (12×80), not PTB identities.
   - Implementation: `src/proto_models1D.py:prototype_loss1d_echonext()`, `tools/dump_val_signals.py:infer_proto_to_class_from_weights()`

2. **Per-Sample, Per-Class Masking:** Multi-label masking ensures cluster loss on `y==1` only, separation on `y==0` only.
   - Implementation: `src/proto_models1D.py:prototype_loss1d_echonext()` (lines 48-101)

3. **Encoder Frozen:** PTB-XL pretrained features preserved throughout.
   - Implementation: `src/training_functions.py:configure_optimizers()` (lines 483-533)

4. **NN vs Unique Projection:** NN shows natural collapse, Unique shows upper bound (1:1 mapping).
   - Implementation: `tools/rebuild_echonext_projection.py`

## Checklist

- [ ] Flags are minimal and clearly documented in `src/main.py`
- [ ] EchoNext loaders return correct shapes/labels (spot-check 5 samples)
- [ ] Prototype grads are non-zero in joint-FT, zero in head-only
- [ ] Diagnostics show NN collapse vs Unique spread before FT, and the after view from the FT'd checkpoint





## Results: Stage B Baseline + Extended Training

### Stage B Baseline (10 epochs)

**Configuration:**
- **Encoder**: FROZEN (PTB-XL pretrained features)
- **Prototypes**: TRAINABLE (fine-tuned on EchoNext)
- **Head**: TRAINABLE (classifier fine-tuned on EchoNext labels)
- **Loss**: BCE + Cluster (λ=0.8) + Separation (λ=0.1)
- **Epochs**: 10
- **Batch size**: 64
- **LR**: 1e-4
- **Early stopping**: `val_auroc_micro` (patience=3)

**Validation Metrics:**

| Metric | Stage A (Head-Only) | Stage B (Joint FT) | Improvement |
|--------|---------------------|-------------------|-------------|
| **val_auroc_micro** | 0.740 (74.0%) | **0.754 (75.4%)** | **+1.4%** |
| **val_auroc_macro** | 0.569 (56.9%) | **0.599 (59.9%)** | **+3.0%** |
| **val_loss** | 0.231 | 0.230 | - |

### Stage B Long Run (30 epochs, ReduceLROnPlateau)

**Configuration:**
- **Encoder**: FROZEN
- **Prototypes**: TRAINABLE
- **Head**: TRAINABLE
- **Loss**: BCE + Cluster (λ=0.8) + Separation (λ=0.1)
- **Epochs**: 30
- **Batch size**: 64
- **LR**: 1e-4 (initial), ReduceLROnPlateau (factor=0.5, patience=2, min_lr=1e-6)
- **Early stopping**: `val_auroc_micro` (patience=5)

**Validation Metrics (Best Epoch):**

| Metric | Value |
|--------|-------|
| **val_auroc_macro** | **0.609** (60.9%) |
| **val_auroc_micro** | **0.787** (78.7%) |
| **Best epoch** | 10 (early stopped) |

**Improvement vs Stage B Baseline:**
- Macro AUROC: 0.599 → 0.609 (+1.0%)
- Micro AUROC: 0.754 → 0.787 (+3.3%)

### Hyperparameter Tuning Results (30 epochs)
Used tuning.py

Top 10 configurations from 20-trial hyperparameter search:

| Rank | Trial | Macro AUROC | Micro AUROC | LR | λ_cluster | λ_sep | Batch Size | Scheduler | Epochs | Best Epoch |
|------|-------|-------------|-------------|-----|-----------|-------|------------|-----------|--------|------------|
| 1 | tune016 | **0.643** | 0.788 | 5e-4 | 0.8 | 0.15 | 64 | CosineAnnealingLR | 17 | 10 |
| 2 | tune010 | 0.632 | 0.788 | 5e-4 | 1.0 | 0.2 | 128 | CosineAnnealingLR | 12 | 5 |
| 3 | tune006 | 0.631 | 0.787 | 5e-4 | 0.6 | 0.15 | 128 | ReduceLROnPlateau | 12 | 12 |
| 4 | tune008 | 0.623 | 0.788 | 1e-4 | 1.0 | 0.1 | 64 | CosineAnnealingLR | 19 | 16 |
| 5 | tune013 | 0.623 | 0.788 | 1e-4 | 0.8 | 0.15 | 64 | CosineAnnealingLR | 19 | 16 |
| 6 | tune019 | 0.620 | 0.788 | 5e-5 | 0.8 | 0.2 | 64 | ReduceLROnPlateau | 26 | 19 |
| 7 | tune005 | 0.611 | 0.789 | 5e-5 | 0.8 | 0.15 | 64 | CosineAnnealingLR | 29 | 27 |
| 8 | tune009 | 0.611 | 0.789 | 5e-5 | 1.0 | 0.05 | 64 | CosineAnnealingLR | 29 | 27 |
| 9 | tune011 | 0.611 | 0.788 | 1e-4 | 1.0 | 0.1 | 128 | CosineAnnealingLR | 29 | 29 |
| 10 | tune017 | 0.611 | 0.788 | 1e-4 | 1.2 | 0.2 | 128 | CosineAnnealingLR | 29 | 29 |

**Key Findings:**
- **Best configuration (tune016)**: LR=5e-4, λ_cluster=0.8, λ_sep=0.15, BS=64, CosineAnnealingLR
- **Best macro AUROC**: 0.643 (+4.4% vs Stage B baseline 0.599)
- **Best micro AUROC**: 0.789 (+3.5% vs Stage B baseline 0.754)
- Higher learning rates (5e-4) with cosine annealing performed best
- Optimal λ_cluster appears to be 0.8, λ_sep around 0.15

**Reproduction Command for Best Configuration (tune016):**

```bash
python3 src/main.py \
  --job_name echonext_tune016_best \
  --training_stage echonext_adapt \
  --dataset_root /opt/gpudata/ecg/echonext-v1.0.0 \
  --train_split train --val_split val \
  --ckpt /opt/gpudata/summereunann/ptbxl_weights/cat1.ckpt \
  --dimension 1D --backbone resnet1d18 \
  --label_set 1 --custom_groups true \
  --single_class_prototype_per_class 5 --proto_dim 512 \
  --freeze_encoder true --freeze_prototypes false \
  --lambda_cluster 0.8 --lambda_separation 0.15 \
  --batch_size 64 --epochs 30 --lr 5e-4 --seed 42 \
  --sampling_rate 100 \
  --loss bce \
  --use_class_weights true \
  --scheduler cosine \
  --scheduler_eta_min 1e-6 \
  --early_stop_metric auroc_micro --early_stop_patience 5 \
  --out_dir results/tune016_best
```

### Post-FT Reprojection Diagnostics

#### NN Projection (Nearest Neighbor)

| Metric | Stage A (Before FT) | Stage B (After FT) | Change |
|--------|---------------------|-------------------|--------|
| **Unique ECGs** | 3 | **29** | **+26** (less collapse) |
| **\|Δcos\| mean** | 0.668 | **0.537** | **-0.131** (closer to unprojected) |
| **\|Δcos\| max** | 0.992 | **0.957** | **-0.035** |
| **\|Δcos\| > 0.1** | 2975 / 3160 | 2900 / 3160 | -75 |

#### Unique Projection (Hungarian 1:1)

| Metric | Stage A | Stage B | Expected |
|--------|---------|---------|----------|
| **Unique ECGs** | 80 | 80 | ~80 |

**Interpretation**: Unique projection maintains 1:1 mapping (80 prototypes → 80 unique ECGs) both before and after fine-tuning, as expected.

## Key Observations

1. **Performance Improvement**: Joint fine-tuning improved both micro and macro AUROC, with macro improving more (+3.0%), suggesting better performance on rare classes.

2. **Reduced NN Collapse**: NN projection reuse increased from 3 to 29 unique ECGs, indicating that fine-tuning reduced prototype collapse (fewer prototypes mapping to the same ECG).

3. **Better Prototype Preservation**: |Δcos| mean decreased from 0.668 to 0.537, meaning prototypes are closer to their unprojected versions after fine-tuning, suggesting the fine-tuning preserved prototype geometry while adapting to EchoNext.

4. **Unique Projection Stability**: Hungarian assignment maintains 80 unique ECGs both before and after, confirming the 1:1 mapping is preserved.

## Implementation Details

### Loss Function

The `prototype_loss1d_echonext` function implements per-sample, per-class masking:

- **Cluster Loss** (λ=0.8): Applied only to positive classes (y==1), skipped for all-zero samples
- **Separation Loss** (λ=0.1): Applied only to negative classes (y==0)
- **Contrastive Loss**: Disabled for EchoNext

### Freezing Verification

- Encoder parameters: `requires_grad=False`
- Prototype parameters: `requires_grad=True`
- Classifier parameters: `requires_grad=True`

## Reproduction Steps

### 1. Train Stage B

```bash
cd /opt/gpudata/summereunann/protoecgnet/src
python3 main.py \
  --job_name echonext_joint_ft_stage_b \
  --training_stage echonext_adapt \
  --dataset_root /opt/gpudata/ecg/echonext-v1.0.0 \
  --train_split train --val_split val \
  --ckpt /opt/gpudata/summereunann/ptbxl_weights/cat1.ckpt \
  --dimension 1D --backbone resnet1d18 \
  --label_set 1 --custom_groups true \
  --single_class_prototype_per_class 5 --proto_dim 512 \
  --train_head_only false \
  --freeze_encoder true --freeze_prototypes false \
  --lambda_cluster 0.8 --lambda_separation 0.1 \
  --lam_clst 0.8 --lam_sep 0.1 \
  --batch_size 64 --epochs 10 --lr 1e-4 \
  --seed 42 --sampling_rate 100 \
  --export_encoder_ts ../results/joint_ft/cat1.encoder.ts \
  --out_dir ../results/joint_ft \
  --checkpoint_dir ../results/joint_ft/checkpoints \
  --log_dir ../results/joint_ft/logs \
  --early_stop_metric auroc_micro --early_stop_patience 3 \
  --use_class_weights true
```

### 2. Export Encoder TS

```bash
# Extract encoder directly from checkpoint
python3 -c "
import sys; sys.path.insert(0, 'src')
import torch
from backbones import resnet1d18

ckpt = torch.load('/opt/gpudata/summereunann/ptbxl_weights/cat1_echonext_stageB.ckpt', map_location='cpu', weights_only=False)
state = ckpt.get('state_dict', ckpt)

encoder = resnet1d18(num_classes=512, dropout=0)
encoder.fc = torch.nn.Identity()

encoder_state = {}
for k, v in state.items():
    if k.startswith('model.feature_extractor.'):
        encoder_state[k.replace('model.feature_extractor.', '')] = v

encoder.load_state_dict(encoder_state, strict=False)
encoder.eval()

dummy = torch.zeros(1, 12, 1000)
traced = torch.jit.trace(encoder, dummy)
traced.save('/opt/gpudata/summereunann/ptbxl_weights/cat1_stageB.encoder.ts')
print('[OK] Encoder exported')
"
```

### 3. Re-project (NN + Unique)

```bash
export ECHONEXT=/opt/gpudata/ecg/echonext-v1.0.0
export OUT=echonext_projection_rebuild_stageB

# NN projection
python3 tools/rebuild_echonext_projection.py \
  --ckpt /opt/gpudata/summereunann/ptbxl_weights/cat1_echonext_stageB.ckpt \
  --dataset-root $ECHONEXT --split train \
  --encoder-ts /opt/gpudata/summereunann/ptbxl_weights/cat1_stageB.encoder.ts \
  --out-dir $OUT --mode nn --seed 42

# Unique projection
python3 tools/rebuild_echonext_projection.py \
  --ckpt /opt/gpudata/summereunann/ptbxl_weights/cat1_echonext_stageB.ckpt \
  --dataset-root $ECHONEXT --split train \
  --encoder-ts /opt/gpudata/summereunann/ptbxl_weights/cat1_stageB.encoder.ts \
  --out-dir $OUT --mode unique --seed 42
```

### 4. Run Diagnostics

```bash
python3 tools/echonext_projection_diagnostics.py \
  --ptb-ckpt /opt/gpudata/summereunann/ptbxl_weights/cat1_echonext_stageB.ckpt \
  --projected-pt-nn $OUT/echonext_nn_projected_prototypes.pt \
  --projected-pt-uni $OUT/echonext_unique_projected_prototypes.pt \
  --proto-labels-csv $OUT/echonext_nn_proto_labels.csv \
  --metadata-nn $OUT/echonext_nn_prototype_metadata.json \
  --metadata-uni $OUT/echonext_unique_prototype_metadata.json \
  --out-dir $OUT \
  --commit-sha $(git rev-parse --short HEAD)
```

### 5. Compare Before/After

```bash
# Before (Stage A)
jq '.reuse.nn_unique_ecgs, .delta_stats.nn.mean, .delta_stats.nn.max' \
  echonext_projection_rebuild_stageA/echonext_cat1_diagnostics_manifest.json

# After (Stage B)
jq '.reuse.nn_unique_ecgs, .delta_stats.nn.mean, .delta_stats.nn.max' \
  echonext_projection_rebuild_stageB/echonext_cat1_diagnostics_manifest.json
```

## Artifacts

- **Checkpoint**: `/opt/gpudata/summereunann/ptbxl_weights/cat1_echonext_stageB.ckpt`
- **Encoder TS**: `/opt/gpudata/summereunann/ptbxl_weights/cat1_stageB.encoder.ts`
- **Projections**: `echonext_projection_rebuild_stageB/`
- **Diagnostics**: `echonext_projection_rebuild_stageB/echonext_cat1_diagnostics_manifest.json`

## Acceptance Criteria

- Val AUROC (micro) improved: 0.740 → 0.754
- Val AUROC (macro) improved: 0.569 → 0.599
- NN reuse increased: 3 → 29 unique ECGs (less collapse)
- |Δcos| mean decreased: 0.668 → 0.537 (better preservation)
- Unique projection maintains 80 unique ECGs
- Encoder frozen, prototypes+head trainable
- Per-sample, per-class masking verified


