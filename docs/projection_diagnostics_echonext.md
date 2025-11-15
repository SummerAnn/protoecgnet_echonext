# ProtoECGNet cat1 diagnostics – EchoNext projection

## Scope

- **Does:** Adds EchoNext projection diagnostics for ProtoECGNet `cat1.ckpt`, comparing nearest-neighbor (NN) vs unique (Hungarian) selection.
- **Does not:** Modify training, touch PTB-XL artifacts, or include lead occlusion.

## Reproduce

```bash
export CKPT=/opt/gpudata/summereunann/ptbxl_weights/cat1.ckpt
export ECHONEXT=/opt/gpudata/ecg/echonext-v1.0.0
export ENCODER=/opt/gpudata/summereunann/ptbxl_weights/cat1.encoder.ts
export OUT=echonext_projection_rebuild

# Rebuild projections
python tools/rebuild_echonext_projection.py --ckpt $CKPT --dataset-root $ECHONEXT \
  --split train --encoder-ts $ENCODER --label-set cat1 --out-dir $OUT --mode nn --seed 42
python tools/rebuild_echonext_projection.py --ckpt $CKPT --dataset-root $ECHONEXT \
  --split train --encoder-ts $ENCODER --label-set cat1 --out-dir $OUT --mode unique --seed 42

# Diagnostics (compares unprojected vs NN vs Unique)
python tools/echonext_projection_diagnostics.py --ptb-ckpt $CKPT \
  --projected-pt-nn  $OUT/echonext_nn_projected_prototypes.pt \
  --projected-pt-uni $OUT/echonext_unique_projected_prototypes.pt \
  --proto-labels-csv $OUT/echonext_nn_proto_labels.csv \
  --metadata-nn  $OUT/echonext_nn_prototype_metadata.json \
  --metadata-uni $OUT/echonext_unique_prototype_metadata.json \
  --out-dir $OUT
```

> All vectors (unprojected + EchoNext projections) live in the same 512-D latent space produced by the cat1 encoder (`--encoder-ts`), so cosine comparisons are well-defined.

## Outputs

| File | Description |
| --- | --- |
| `echonext_cat1_unprojected_cosine_heatmap.png` | Base checkpoint pairwise cosine (viridis) |
| `echonext_cat1_projected_nn_cosine_heatmap.png` | EchoNext NN projection cosine matrix |
| `echonext_cat1_projected_unique_cosine_heatmap.png` | EchoNext unique projection cosine matrix |
| `echonext_cat1_unprojected_pca.png` | PCA of checkpoint prototypes |
| `echonext_cat1_projected_nn_pca.png` | PCA of NN projection |
| `echonext_cat1_projected_unique_pca.png` | PCA of unique projection |
| `echonext_cat1_cosine_delta_histogram_nn.png` | |Δcos| histogram (NN vs unprojected) |
| `echonext_cat1_cosine_delta_histogram_unique.png` | |Δcos| histogram (unique vs unprojected) |
| `echonext_cat1_nn_reuse.json` | EchoNext `ecg_id → count` (how many prototypes picked each id; NN allows reuse) |
| `echonext_cat1_unique_reuse.json` | EchoNext `ecg_id → 1` (1 proto per ECG; expect ≈80 unique ids) |
| `echonext_cat1_diagnostics_manifest.json` | Provenance + Δcos stats + reuse counts |
| `echonext_cat1_rebuild_manifest.json` | Projection build provenance (paths, selection, versions) |

## Interpretation checklist

1. NN allows prototype reuse; unique enforces one-to-one via Hungarian.
2. Compare class block stability in heatmaps vs PTB-XL baseline.
3. Use reuse JSONs to quantify “80 → K” EchoNext samples under NN vs unique.

### EchoNext collapse vs unique selection (example from 2025-11-14 run)

Numbers below are from a single diagnostic run (manifest: `echonext_cat1_diagnostics_manifest.json`). Reproduce or recompute with the snippet if needed.

| Mode | Unique ECGs | Mean \|Δcos\| | Max \|Δcos\| | Reuse note |
| --- | ---: | ---: | ---: | --- |
| NN (unconstrained) | 3 | 0.668 | 0.998 | 70→`ecg_id=50364`, 10→`43071`, 5→`16486` (collapse) |
| Unique (Hungarian) | 80 | 0.624 | 0.992 | 1 proto → 1 EchoNext ECG (no reuse) |

Compute stats yourself:

```bash
OUT=echonext_projection_rebuild
NN_UNIQ=$(jq 'length' "$OUT/echonext_cat1_nn_reuse.json")
UNI_UNIQ=$(jq 'length' "$OUT/echonext_cat1_unique_reuse.json")
MEAN_NN=$(jq -r '.delta_stats.delta_nn.mean'  "$OUT/echonext_cat1_diagnostics_manifest.json")
MAX_NN=$(jq -r  '.delta_stats.delta_nn.max'   "$OUT/echonext_cat1_diagnostics_manifest.json")
MEAN_U=$(jq -r  '.delta_stats.delta_unique.mean' "$OUT/echonext_cat1_diagnostics_manifest.json")
MAX_U=$(jq -r   '.delta_stats.delta_unique.max'  "$OUT/echonext_cat1_diagnostics_manifest.json")
printf "Mode\tUnique ECGs\tMean|Δcos|\tMax|Δcos|\n"
printf "NN\t%s\t%s\t%s\n" "$NN_UNIQ" "$MEAN_NN" "$MAX_NN"
printf "Unique\t%s\t%s\t%s\n" "$UNI_UNIQ" "$MEAN_U" "$MAX_U"
```

- Unprojected cosine matrix retains the 16×5 block structure; projecting onto EchoNext with NN collapses most prototypes onto a handful of “central” embeddings, yielding an almost all-ones heatmap and a single blob in PCA.
- Unique assignment forces 80 distinct anchors, restoring block structure and PCA spread, though the Δcos remains large, highlighting the PTB→EchoNext domain gap.
- Showing both modes answers the reviewer ask: NN (unconstrained, “80 → 3”) vs Unique (upper bound, “80 → 80”).

## Verification

- 80 prototypes (16 × 5) for unprojected and both projections.
- Unprojected/NN/Unique tensors share the same `[80, D]` shape (identical `D`).
- Rebuild/diagnostics manifests capture checkpoint path, dataset root, selection mode, seed, and versions; manifests list selection method (`nearest_neighbor` vs `hungarian|greedy_fallback`).
- Reuse counts: `len(nn_reuse) ≤ len(unique_reuse)`; unique ≥ 75 (target 80). Lower values usually mean the EchoNext pool was limited (`--limit` or `split` filtering).

## Troubleshooting

- Missing EchoNext dataset assets → check `$ECHONEXT` path.
- Hungarian fallback triggered → install `scipy>=1.10` or document greedy fallback.
- CUDA mismatch → re-run with `--device cpu --limit 5` for smoke tests.
- No SciPy? Unique mode automatically falls back to greedy 1:1 and records `selection_method` in the manifest.
- Memory-limited? Use `--limit 200` for smoke runs, then rerun with full data when ready.
- Different EchoNext split? Pass `--split val` or `--split test` to probe other cohorts.
