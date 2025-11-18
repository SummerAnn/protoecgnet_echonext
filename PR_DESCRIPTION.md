# EchoNext Adaptation Baseline with Hyperparameter Tuning

## Overview

This PR implements EchoNext dataset adaptation for ProtoECGNet, enabling fine-tuning of prototypes and classifier head on EchoNext multi-label ECG data while keeping the encoder frozen.

## Key Features

- **Stage A (Head-only)**: Baseline with encoder + prototypes frozen, only classifier head trained
- **Stage B (Joint Fine-tuning)**: Prototypes + head trainable, encoder frozen
- **Hyperparameter Tuning**: 20-trial search with grid/random strategies
- **Post-FT Reprojection**: Diagnostics comparing NN vs Unique projection methods
- **Comprehensive Tools**: Validation signal dumper, per-class analyzer, encoder export

## Results

### Baseline (10 epochs)
- Macro AUROC: **0.599** (59.9%)
- Micro AUROC: **0.754** (75.4%)

### Long Run (30 epochs, ReduceLROnPlateau)
- Macro AUROC: **0.609** (60.9%)
- Micro AUROC: **0.787** (78.7%)

### Best Hyperparameter Tuned (tune016)
- Macro AUROC: **0.643** (64.3%) - **+4.4% vs baseline**
- Micro AUROC: **0.788** (78.8%) - **+3.4% vs baseline**
- Configuration: LR=5e-4, λ_cluster=0.8, λ_sep=0.15, BS=64, CosineAnnealingLR

## Implementation Highlights

1. **Multi-label Support**: EchoNext dataloader with `MultiLabelBinarizer` for consistent label encoding
2. **Per-Class Masking**: Prototype losses applied only to relevant classes per sample
3. **Class Imbalance Handling**: Inverse-frequency class weights with optional clipping/smoothing
4. **Ablation Studies**: ASL loss and Label-Aware Batch Sampler (documented but not in winning config)
5. **Diagnostics**: Tools for analyzing per-class performance, prototype coverage, and gradient attribution

## Files Changed

- `src/data/echonext.py`: EchoNext dataset loader
- `src/model/export.py`: Encoder TorchScript export utility
- `src/losses/asl.py`: Asymmetric Loss implementation
- `src/proto_models1D.py`: EchoNext-specific prototype loss
- `src/training_functions.py`: Training logic with freezing and metrics
- `src/main.py`: CLI flags for EchoNext adaptation
- `tools/`: Diagnostics and analysis tools
- `tests/`: Comprehensive unit and integration tests
- `docs/echonext_adapt_v3.md`: Complete documentation with results

## Testing

- Unit tests for prototype mapping, unique projection, loss functions
- Integration tests for Stage B guardrails and diagnostics invariants
- All tests passing

## Documentation

See `docs/echonext_adapt_v3.md` for:
- Complete workflow (Stage A, B, reprojection)
- Reproduction commands
- Results tables and interpretation
- Design rationale
