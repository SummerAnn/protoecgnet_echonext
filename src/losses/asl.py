# src/losses/asl.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    - p_t = sigmoid(logits) for positives, (1 - sigmoid(logits)) for negatives
    - For negatives only, down-weight easy negatives with (1 - p)^gamma_neg and clamp p with clip.
    - For positives, optionally use gamma_pos (often 0).
    
    References: ASL (Ridnik et al., 2021) behavior.
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        pos_weight: torch.Tensor | None = None,   # optional class-wise positive weights
    ):
        super().__init__()
        assert reduction in {"none", "mean", "sum"}
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip) if clip is not None else None
        self.eps = float(eps)
        self.reduction = reduction
        self.register_buffer("pos_weight_buf", None, persistent=False)
        if pos_weight is not None:
            self.set_pos_weight(pos_weight)
    
    def set_pos_weight(self, pw: torch.Tensor | None):
        if pw is None:
            self.pos_weight_buf = None
        else:
            self.pos_weight_buf = pw.detach().float().view(1, -1)  # (1, C)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, C) raw scores
        targets: (B, C) in {0,1}
        """
        x = logits
        y = targets
        
        # Probabilities
        xs_pos = torch.sigmoid(x)
        xs_neg = 1.0 - xs_pos
        
        # Optional probability clip for negatives (reduces extreme easy negatives)
        if self.clip is not None and self.clip > 0.0:
            xs_neg = torch.clamp(xs_neg + self.clip, max=1.0)
        
        # Asymmetric focusing
        # For positives, weight by (1 - p)^gamma_pos; for negatives, weight by p^gamma_neg on (1-p)
        pos_loss = - y * torch.log(xs_pos.clamp_min(self.eps))
        neg_loss = - (1.0 - y) * torch.log(xs_neg.clamp_min(self.eps))
        
        if self.gamma_pos > 0.0:
            pos_focus = torch.pow(1.0 - xs_pos, self.gamma_pos)
            pos_loss = pos_loss * pos_focus
        
        if self.gamma_neg > 0.0:
            neg_focus = torch.pow(xs_pos, self.gamma_neg)   # xs_pos small → easy negative → down-weighted
            neg_loss = neg_loss * neg_focus
        
        loss = pos_loss + neg_loss  # (B, C)
        
        # Optional class-wise positive weighting (kept gentle; many ASL setups omit this)
        if self.pos_weight_buf is not None:
            loss = loss * torch.where(y > 0.5, self.pos_weight_buf, 1.0)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

