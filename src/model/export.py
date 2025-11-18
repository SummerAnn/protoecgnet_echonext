"""Export encoder to TorchScript for consistent diagnostics."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Union


def _to_512(x: torch.Tensor) -> torch.Tensor:
    """Ensure output is 512-D."""
    if x.ndim == 1:
        if x.shape[0] != 512:
            raise ValueError(f"Expected 512-D vector, got {x.shape[0]}-D")
        return x
    elif x.ndim == 2:
        if x.shape[1] != 512:
            raise ValueError(f"Expected (B, 512), got {x.shape}")
        return x
    else:
        raise ValueError(f"Unexpected tensor shape: {x.shape}")


class WrapEncoder(nn.Module):
    """Wrapper to ensure encoder outputs 512-D vectors."""
    
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.enc = encoder
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: (B, 12, T) -> (B, 512)."""
        out = self.enc(x)
        return _to_512(out)


def export_encoder_to_torchscript(
    model: nn.Module,
    output_path: Union[str, Path],
    device: str = "cpu",
    input_len: int = 1000,
    leads: int = 12,
) -> None:
    """Export encoder from ProtoECGNet model to TorchScript.
    
    Args:
        model: ProtoECGNet1D or ProtoECGNet2D model
        output_path: Path to save TorchScript encoder
        device: Device to use for export ('cpu' or 'cuda')
        input_len: Expected input length in samples
        leads: Number of ECG leads (default 12)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract encoder (feature extractor)
    if hasattr(model, "feature_extractor"):
        encoder = model.feature_extractor
    else:
        raise ValueError(f"Model {type(model)} does not have 'feature_extractor' attribute")
    
    # Set to eval mode
    encoder.eval()
    
    # Move to device
    device_obj = torch.device(device)
    encoder = encoder.to(device_obj)
    
    # Wrap encoder to ensure 512-D output
    wrapped = WrapEncoder(encoder).eval().to(device_obj)
    
    # Create dummy input: (1, 12, T)
    dummy = torch.zeros(1, leads, input_len, device=device_obj)
    
    # Sanity check: forward pass
    with torch.inference_mode():
        z = wrapped(dummy)
        assert z.ndim == 2 and z.size(1) == 512, f"Expected (1, 512), got {z.shape}"
    
    # Trace and save
    try:
        scripted = torch.jit.trace(wrapped, (dummy,))
        scripted.save(str(output_path))
        print(f"[OK] TorchScript encoder saved to {output_path}")
    except Exception as e:
        # Fallback to script if tracing fails
        print(f"[WARNING] Tracing failed ({e}), trying scripting...")
        scripted = torch.jit.script(wrapped)
        scripted.save(str(output_path))
        print(f"[OK] TorchScript encoder (scripted) saved to {output_path}")

