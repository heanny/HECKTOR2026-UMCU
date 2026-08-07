"""
models.py — DenseNet121-based architectures for HECKTOR 2026.

All three model classes support optional clinical and/or imaging inputs via
use_clinical and use_image constructor flags:

  use_image=True,  use_clinical=True  : full model (default)
  use_image=True,  use_clinical=False : imaging only
  use_image=False, use_clinical=True  : clinical only

The fusion dimension is always FIXED regardless of which inputs are active:
  img branch  : 1024 dims  (zeros when use_image=False)
  clc branch  :   64 dims  (zeros when use_clinical=False)
  fusion      : 1088 dims  (Task 2 / Task 3)
              : 1096 dims  (Multi-task RFS head — includes 8-dim T/N soft preds)

This means the downstream FC layers are identical in all modes; you can swap
modes without changing any layer size.

Clinical feature inputs per model
───────────────────────────────────
  DenseNet121Task2      staging clinical only
                        (Age, Gender, HPV Status, HPV_missing)  → dim 4

  DenseNet121Task3      RFS clinical only
                        (Age, Gender, HPV Status, Treatment, HPV_missing) → dim 5

  DenseNet121MultiTask  two separate clinical encoders:
    • staging branch  → staging clinical (dim 4)
    • RFS branch      → RFS clinical (dim 5)  +  T/N soft predictions (dim 8)

Multi-task forward data flow
─────────────────────────────
  image [B,C,H,W,D] ──► DenseNet backbone ──► img_feat [B,1024]
                                                    │
  clinical_staging [B,4] ──► stg_enc ──► stg_clc [B,64]
                                              │
                              cat(img_feat, stg_clc) = [B,1088]
                                              │
                                       staging_trunk
                                         [B, 256]
                                       ┌────┴────┐
                                   t_head      n_head
                                   [B, 4]      [B, 4]
                                       └────┬────┘
                        softmax (detached) → tn_soft [B, 8]

  clinical_rfs [B,5] ──► rfs_enc ──► rfs_clc [B,64]
                                          │
                  cat(img_feat, rfs_clc, tn_soft) = [B, 1096]
                                          │
                                      rfs_head
                                       risk [B]
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

from config import (
    N_T_CLASSES, N_N_CLASSES,
    N_STAGING_CLINICAL, N_RFS_CLINICAL,
)


# ─── Shared building blocks ───────────────────────────────────────────────────

class _DenseNetBackbone(nn.Module):
    """3-D DenseNet121 without its classifier head → 1024-dim feature vector."""

    FEATURE_DIM = 1024

    def __init__(self, in_channels: int = 4):
        super().__init__()
        _full = DenseNet121(
            spatial_dims=3, in_channels=in_channels,
            out_channels=2, pretrained=False,
        )
        self.features = _full.features
        self.pool     = nn.AdaptiveAvgPool3d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W, D]  →  [B, 1024]"""
        x = self.features(x)
        x = torch.relu(x)
        return self.pool(x).flatten(1)


def _clinical_encoder(n_in: int, n_out: int = 64,
                       dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(n_in, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(128, n_out), nn.BatchNorm1d(n_out), nn.ReLU(),
    )


def _fc_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim),
        nn.ReLU(), nn.Dropout(dropout),
    )


# ─── Task 2 — staging only ────────────────────────────────────────────────────

class DenseNet121Task2(nn.Module):
    """
    T-stage and N-stage classifier.

    Forward inputs
    --------------
    image            : [B, C, H, W, D]  (ignored when use_image=False)
    clinical_staging : [B, N_STAGING_CLINICAL]  (ignored when use_clinical=False)

    Returns
    -------
    t_logits : [B, N_T_CLASSES]
    n_logits : [B, N_N_CLASSES]
    """

    def __init__(
        self,
        in_channels:  int   = 4,
        n_clinical:   int   = N_STAGING_CLINICAL,   # 4
        use_image:    bool  = True,
        use_clinical: bool  = True,
        clc_dim:      int   = 64,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.use_image    = use_image
        self.use_clinical = use_clinical
        self.clc_dim      = clc_dim

        if use_image:
            self.backbone = _DenseNetBackbone(in_channels)

        if use_clinical:
            self.clc_encoder = _clinical_encoder(n_clinical, clc_dim, dropout)

        fusion_dim = _DenseNetBackbone.FEATURE_DIM + clc_dim   # 1088 always fixed
        self.trunk  = nn.Sequential(
            _fc_block(fusion_dim, 512, dropout),
            _fc_block(512, 256, dropout),
        )
        self.t_head = nn.Linear(256, N_T_CLASSES)
        self.n_head = nn.Linear(256, N_N_CLASSES)

    def forward(self, image, clinical_staging):
        B, dev = image.shape[0], image.device

        img_feat = (self.backbone(image)
                    if self.use_image
                    else torch.zeros(B, _DenseNetBackbone.FEATURE_DIM, device=dev))

        clc_feat = (self.clc_encoder(clinical_staging)
                    if self.use_clinical
                    else torch.zeros(B, self.clc_dim, device=dev))

        x = self.trunk(torch.cat([img_feat, clc_feat], dim=1))
        return self.t_head(x), self.n_head(x)


# ─── Task 3 — RFS only ───────────────────────────────────────────────────────

class DenseNet121Task3(nn.Module):
    """
    Cox log-risk score predictor for RFS.

    Forward inputs
    --------------
    image        : [B, C, H, W, D]  (ignored when use_image=False)
    clinical_rfs : [B, N_RFS_CLINICAL]  (ignored when use_clinical=False)

    Returns
    -------
    risk : [B]   log-risk score (higher → worse prognosis)
    """

    def __init__(
        self,
        in_channels:  int   = 4,
        n_clinical:   int   = N_RFS_CLINICAL,   # 5
        use_image:    bool  = True,
        use_clinical: bool  = True,
        clc_dim:      int   = 64,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.use_image    = use_image
        self.use_clinical = use_clinical
        self.clc_dim      = clc_dim

        if use_image:
            self.backbone = _DenseNetBackbone(in_channels)

        if use_clinical:
            self.clc_encoder = _clinical_encoder(n_clinical, clc_dim, dropout)

        fusion_dim = _DenseNetBackbone.FEATURE_DIM + clc_dim   # 1088 always fixed
        self.head = nn.Sequential(
            _fc_block(fusion_dim, 512, dropout),
            _fc_block(512, 256, dropout),
            nn.Linear(256, 1),
        )

    def forward(self, image, clinical_rfs):
        B, dev = image.shape[0], image.device

        img_feat = (self.backbone(image)
                    if self.use_image
                    else torch.zeros(B, _DenseNetBackbone.FEATURE_DIM, device=dev))

        clc_feat = (self.clc_encoder(clinical_rfs)
                    if self.use_clinical
                    else torch.zeros(B, self.clc_dim, device=dev))

        return self.head(torch.cat([img_feat, clc_feat], dim=1)).squeeze(-1)


# ─── Multi-task — staging + RFS ───────────────────────────────────────────────

class DenseNet121MultiTask(nn.Module):
    """
    Joint model for Task 2 (T/N staging) and Task 3 (RFS).

    Separate clinical encoders per branch:
      • Staging branch  : clinical_staging [B, N_STAGING_CLINICAL]
      • RFS branch      : clinical_rfs     [B, N_RFS_CLINICAL]
                          + T/N soft predictions [B, 8]  (detached)

    The T/N→RFS path uses detach() so the Cox loss cannot distort staging heads.

    Training strategy (two-phase)
    ------------------------------
    Phase 1  epochs 1..warmup  : loss = L_T + L_N         (rfs_weight = 0)
    Phase 2  epochs warmup+1.. : loss = L_T + L_N + w·Cox (rfs_weight = w)

    Forward inputs
    --------------
    image            : [B, C, H, W, D]  (ignored when use_image=False)
    clinical_staging : [B, N_STAGING_CLINICAL]  (ignored when use_clinical=False)
    clinical_rfs     : [B, N_RFS_CLINICAL]      (ignored when use_clinical=False)

    Returns
    -------
    t_logits : [B, N_T_CLASSES]
    n_logits : [B, N_N_CLASSES]
    risk     : [B]
    """

    _TN_SOFT_DIM = N_T_CLASSES + N_N_CLASSES   # 8

    def __init__(
        self,
        in_channels:  int   = 4,
        n_stg_clc:    int   = N_STAGING_CLINICAL,   # 4
        n_rfs_clc:    int   = N_RFS_CLINICAL,        # 5
        use_image:    bool  = True,
        use_clinical: bool  = True,
        clc_dim:      int   = 64,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.use_image    = use_image
        self.use_clinical = use_clinical
        self.clc_dim      = clc_dim
        img_dim           = _DenseNetBackbone.FEATURE_DIM   # 1024

        # ── Shared backbone ──────────────────────────────────────────────────
        if use_image:
            self.backbone = _DenseNetBackbone(in_channels)

        # ── Staging branch ───────────────────────────────────────────────────
        if use_clinical:
            self.stg_clc_enc = _clinical_encoder(n_stg_clc, clc_dim, dropout)

        stg_fusion_dim = img_dim + clc_dim                   # 1088 always fixed
        self.staging_trunk = nn.Sequential(
            _fc_block(stg_fusion_dim, 512, dropout),
            _fc_block(512, 256, dropout),
        )
        self.t_head = nn.Linear(256, N_T_CLASSES)
        self.n_head = nn.Linear(256, N_N_CLASSES)

        # ── RFS branch ───────────────────────────────────────────────────────
        # img_feat(1024) + rfs_clc(64) + tn_soft(8) = 1096 always fixed
        if use_clinical:
            self.rfs_clc_enc = _clinical_encoder(n_rfs_clc, clc_dim, dropout)

        rfs_in_dim = img_dim + clc_dim + self._TN_SOFT_DIM   # 1096 always fixed
        self.rfs_head = nn.Sequential(
            _fc_block(rfs_in_dim, 512, dropout),
            _fc_block(512, 256, dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        image:            torch.Tensor,   # [B, C, H, W, D]
        clinical_staging: torch.Tensor,   # [B, N_STAGING_CLINICAL]
        clinical_rfs:     torch.Tensor,   # [B, N_RFS_CLINICAL]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns  t_logits [B,4],  n_logits [B,4],  risk [B]
        """
        B, dev = image.shape[0], image.device

        img_feat = (self.backbone(image)
                    if self.use_image
                    else torch.zeros(B, _DenseNetBackbone.FEATURE_DIM, device=dev))

        # ── Staging ──────────────────────────────────────────────────────────
        stg_clc = (self.stg_clc_enc(clinical_staging)
                   if self.use_clinical
                   else torch.zeros(B, self.clc_dim, device=dev))

        s        = self.staging_trunk(torch.cat([img_feat, stg_clc], dim=1))
        t_logits = self.t_head(s)
        n_logits = self.n_head(s)

        # ── RFS ──────────────────────────────────────────────────────────────
        # Detach T/N softmax: Cox gradient must not distort staging heads
        tn_soft = torch.cat([
            torch.softmax(t_logits.detach(), dim=1),       # [B, 4]
            torch.softmax(n_logits.detach(), dim=1),       # [B, 4]
        ], dim=1)                                           # [B, 8]

        rfs_clc = (self.rfs_clc_enc(clinical_rfs)
                   if self.use_clinical
                   else torch.zeros(B, self.clc_dim, device=dev))

        risk = self.rfs_head(
            torch.cat([img_feat, rfs_clc, tn_soft], dim=1)
        ).squeeze(-1)

        return t_logits, n_logits, risk
