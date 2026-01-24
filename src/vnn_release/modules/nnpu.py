"""Non-Negative PU (nnPU) loss and simple training utilities (PyTorch).

This module lets you train any logit-producing model with nnPU loss using
separate positive (P) and unlabeled (U) mini-batches. Optionally you can pass
per-sample weights for a subset of U marked as hard negatives (HN).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class NNPUConfig:
    pi: float  # class prior P(Y=1)
    reduction: str = "mean"  # 'mean' or 'sum'
    hn_weight: float = 1.0    # scale for hard negatives in U


class NNPUCriterion(nn.Module):
    """Kiryo et al., 'Positive-Unlabeled Learning with Non-Negative Risk' (2017).

    Uses logistic (BCE-with-logits) partial risks:
      Rp+  = E_p[ BCEWithLogits(logit_p, 1) ]
      Rp-  = E_p[ BCEWithLogits(logit_p, 0) ]
      Ru-  = E_u[ BCEWithLogits(logit_u, 0) ]

    Overall risk:  pi * Rp+  +  max(0, Ru- - pi * Rp-)
    """

    def __init__(self, cfg: NNPUConfig):
        super().__init__()
        self.cfg = cfg
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        logit_p: torch.Tensor,
        logit_u: torch.Tensor,
        *,
        u_weights: Optional[torch.Tensor] = None,  # e.g., HN upweighting
    ) -> torch.Tensor:
        pi = self.cfg.pi
        # positive partial risks
        rp_pos = self.bce(logit_p, torch.ones_like(logit_p))  # y=1
        rp_neg = self.bce(logit_p, torch.zeros_like(logit_p)) # y=0

        # unlabeled negative partial risk (optionally weighted)
        ru_neg = self.bce(logit_u, torch.zeros_like(logit_u))
        if u_weights is not None:
            ru_neg = ru_neg * u_weights

        if self.cfg.reduction == "mean":
            rp_pos = rp_pos.mean()
            rp_neg = rp_neg.mean()
            ru_neg = ru_neg.mean()
        else:
            rp_pos = rp_pos.sum()
            rp_neg = rp_neg.sum()
            ru_neg = ru_neg.sum()

        risk = pi * rp_pos + torch.clamp(ru_neg - pi * rp_neg, min=0.0)
        return risk


def train_nnpu(
    model: nn.Module,
    *,
    loader_p,
    loader_u,
    cfg: NNPUConfig,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda",
    weight_decay: float = 0.0,
    u_hn_getter=None,  # callable(batch_idx, x_u, y_u, meta_u) -> weights tensor
    clip_grad: Optional[float] = 5.0,
    verbose: bool = True,
):
    """Train a model with nnPU loss using two loaders (P and U).

    Each loader must yield either (x, y) or (x, y, meta). Only x is used here.
    If `u_hn_getter` is provided, it must return a tensor of weights per U
    sample to upweight hard negatives (e.g., cfg.hn_weight for HN, 1.0 else).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    crit = NNPUCriterion(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for ep in range(1, epochs + 1):
        model.train(); run_loss = 0.0; n_batches = 0
        it_p = iter(loader_p)
        it_u = iter(loader_u)
        steps = min(len(loader_p), len(loader_u))
        for _ in range(steps):
            # fetch P batch
            batch_p = next(it_p)
            if len(batch_p) == 3:
                x_p, _, _ = batch_p
            else:
                x_p, _ = batch_p
            # fetch U batch
            batch_u = next(it_u)
            if len(batch_u) == 3:
                x_u, _, meta_u = batch_u
            else:
                x_u, _ = batch_u
                meta_u = None

            x_p = x_p.to(device)
            x_u = x_u.to(device)
            logit_p = model(x_p).squeeze(-1)
            logit_u = model(x_u).squeeze(-1)

            u_weights = None
            if u_hn_getter is not None:
                u_weights = u_hn_getter(_, x_u, None, meta_u)
                if u_weights is not None:
                    u_weights = u_weights.to(device)

            loss = crit(logit_p, logit_u, u_weights=u_weights)
            opt.zero_grad(); loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()

            run_loss += loss.item(); n_batches += 1
        if verbose:
            print(f"[nnPU] Epoch {ep:03d} | loss {run_loss/max(1,n_batches):.4f}")

    return model

