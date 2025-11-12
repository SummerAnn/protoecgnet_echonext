# ProtoECGNet PTB-XL Projection Diagnostics

Run the diagnostic script with the PTB-XL ProtoECGNet checkpoint and the
projection tensor produced by the original push step:

```bash
python tools/diagnostics_prototypes.py \
  --ptb_ckpt /path/to/ptbxl_weights/cat1.ckpt \
  --projected_pt /path/to/ptbxl_original_projection_protos.pt \
  --proto_labels_csv /path/to/ptb_cat1.PROJECTED.proto_labels.csv \
  --out_dir /path/to/out_dir \
  --projected_label "Projected→PTB-XL (orig)"
```

Outputs are written to the directory given by `--out_dir` and include:

- `ptb_(unprojected)_cosine_heatmap.png`
- `projected→ptb-xl_cosine_heatmap.png`
- `projected→ptb-xl_dup_clusters.json`

The same script can be reused for other projection tensors by changing the
inputs.
