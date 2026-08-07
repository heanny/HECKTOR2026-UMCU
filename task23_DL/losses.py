"""
losses.py — Loss functions for HECKTOR 2026 Task 2 and Task 3.

Task 2: OrdinalCrossEntropyLoss  (T-stage / N-stage ordinal classification)
Task 3: NegativeLogLikelihood    (Breslow-approximation Cox partial likelihood)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Task 3: Cox partial likelihood ──────────────────────────────────────────

class NegativeLogLikelihood(nn.Module):
    """
    Breslow-approximation negative Cox partial log-likelihood.

    Adapted from TransRP (github.com/baoqiangma96/TransRP).

    Call signature
    --------------
    loss, nll, l2 = criterion(risk_pred, times, events)
    loss, nll, l2 = criterion(risk_pred, times, events, model)  # + L2 reg

    Parameters
    ----------
    l2_weight : weight for L2 parameter regularisation (0 = disabled).
    """

    def __init__(self, l2_reg: float = 5e-5):
        super().__init__()
        self.l2_reg = l2_reg

    def forward(
        self,
        risk_pred: torch.Tensor,     # [B]  log-risk scores (higher = worse)
        times:     torch.Tensor,     # [B]  follow-up / event times
        events:    torch.Tensor,     # [B]  1 = event occurred, 0 = censored
        model:     nn.Module = None,
    ):
        # Sort descending by time so cumsum gives the risk set at each event
        order = torch.argsort(times, descending=True)
        risk  = risk_pred[order]
        E     = events[order]

        # Breslow: log Σ exp(risk_j) for all j at risk at time i
        log_cumsum = torch.logcumsumexp(risk, dim=0)

        # Partial log-likelihood for event cases only
        nll = -(risk - log_cumsum) * E
        nll = nll.sum() / E.sum().clamp(min=1.0)

        # Optional weight-norm L2 regularisation
        l2 = risk_pred.new_zeros(1).squeeze()
        if model is not None and self.l2_reg > 0:
            for name, param in model.named_parameters():
                if "weight" in name:
                    l2 = l2 + torch.norm(param, p=2)
            l2 = l2 * self.l2_reg

        return nll + l2, nll, l2


# ─── Task 2: Ordinal cross-entropy ────────────────────────────────────────────

class OrdinalCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with an ordinal-distance penalty.

    Standard CE is re-weighted per sample by
        w = 1 + |E[predicted class] − true class|
    so errors far from the true ordinal class are penalised more strongly.

    Parameters
    ----------
    n_classes : number of ordinal stages (4 for T1-T4 or N0-N3).
    """

    def __init__(self, n_classes: int = 4):
        super().__init__()
        self.n_classes = n_classes
        self.register_buffer(
            "class_idx",
            torch.arange(n_classes, dtype=torch.float32),
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : [B, n_classes]
        targets : [B]  integer labels
        """
        ce = F.cross_entropy(logits, targets, reduction="none")   # [B]

        with torch.no_grad():
            probs    = torch.softmax(logits, dim=-1)
            pred_exp = (probs * self.class_idx).sum(dim=-1)        # expected class
            w        = 1.0 + torch.abs(pred_exp - targets.float())

        return (ce * w).mean()


class Task2Loss(nn.Module):
    """
    Combined T-stage + N-stage ordinal CE loss.

    total = t_weight * loss_T + n_weight * loss_N

    Returns (total_loss, t_loss, n_loss).
    """

    def __init__(
        self,
        n_classes: int   = 4,
        t_weight:  float = 1.0,
        n_weight:  float = 1.0,
    ):
        super().__init__()
        self.t_loss   = OrdinalCrossEntropyLoss(n_classes)
        self.n_loss   = OrdinalCrossEntropyLoss(n_classes)
        self.t_weight = t_weight
        self.n_weight = n_weight

    def forward(self, t_logits, n_logits, t_targets, n_targets):
        lt = self.t_loss(t_logits, t_targets)
        ln = self.n_loss(n_logits, n_targets)
        return self.t_weight * lt + self.n_weight * ln, lt, ln
