#!/usr/bin/env python3
"""CLI for exporting encoder to TorchScript."""

import argparse
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proto_models1D import ProtoECGNet1D
from model.export import export_encoder_to_torchscript


def parse_args():
    parser = argparse.ArgumentParser(description="Export encoder to TorchScript")
    parser.add_argument("--ckpt", type=Path, required=True, help="ProtoECGNet checkpoint path")
    parser.add_argument("--out", type=Path, required=True, help="Output TorchScript path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing file")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for export")
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.out.exists() and not args.force:
        print(f"[ERROR] Output file {args.out} exists. Use --force to overwrite.")
        sys.exit(1)
    
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    # Try to load via Lightning first
    try:
        # Extract model config from checkpoint
        hparams = ckpt.get("hyper_parameters", {})
        
        # Infer num_classes from state_dict
        state = ckpt.get("state_dict", ckpt)
        if "model.classifier.weight" in state:
            num_classes = state["model.classifier.weight"].shape[0]
        elif "classifier.weight" in state:
            num_classes = state["classifier.weight"].shape[0]
        else:
            num_classes = 12  # Default for EchoNext
        
        print(f"[INFO] Inferred num_classes={num_classes} from checkpoint")
        
        # Create model (minimal config for export)
        # Use standard cat1 config: 5 prototypes per class, 2 per border
        # Set custom_groups=False to avoid loading co-occurrence matrix
        model = ProtoECGNet1D(
            num_classes=num_classes,
            backbone="resnet1d18",
            proto_dim=512,
            single_class_prototype_per_class=5,
            joint_prototypes_per_border=2,
            label_set="cat1",
            custom_groups=False,  # Avoid loading co-occurrence matrix
            load_strict=False,
        )
        
        # Load state dict
        state_dict = ckpt.get("state_dict", ckpt)
        # Remove 'model.' prefix if present
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    print(f"[INFO] Exporting encoder to {args.out}")
    export_encoder_to_torchscript(
        model=model,
        output_path=args.out,
        device=args.device,
        input_len=1000,
        leads=12,
    )
    print(f"[OK] Encoder exported successfully")


if __name__ == "__main__":
    main()

