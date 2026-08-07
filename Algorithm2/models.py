"""
models.py — DenseNet121-based multitask architecture for HECKTOR 2026.

Copied verbatim (DenseNet121MultiTask + shared blocks) from the training repo
task2_3_prediction/models.py. Only the `config` import is repointed to the
path-free `constants` module so it is safe to import inside the container.
"""

import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

from constants import (
    N_T_CLASSES, N_N_CLASSES,
    N_STAGING_CLINICAL, N_RFS_CLINICAL,
)


class _DenseNetBackbone(nn.Module):
    """3-D DenseNet121 without its classifier head -> 1024-dim feature vector."""

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


class DenseNet121MultiTask(nn.Module):
    """Joint model for Task 2 (T/N staging) and Task 3 (RFS)."""

    _TN_SOFT_DIM = N_T_CLASSES + N_N_CLASSES   # 8

    def __init__(
        self,
        in_channels:  int   = 4,
        n_stg_clc:    int   = N_STAGING_CLINICAL,
        n_rfs_clc:    int   = N_RFS_CLINICAL,
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

        if use_image:
            self.backbone = _DenseNetBackbone(in_channels)

        if use_clinical:
            self.stg_clc_enc = _clinical_encoder(n_stg_clc, clc_dim, dropout)

        stg_fusion_dim = img_dim + clc_dim                   # 1088
        self.staging_trunk = nn.Sequential(
            _fc_block(stg_fusion_dim, 512, dropout),
            _fc_block(512, 256, dropout),
        )
        self.t_head = nn.Linear(256, N_T_CLASSES)
        self.n_head = nn.Linear(256, N_N_CLASSES)

        if use_clinical:
            self.rfs_clc_enc = _clinical_encoder(n_rfs_clc, clc_dim, dropout)

        rfs_in_dim = img_dim + clc_dim + self._TN_SOFT_DIM   # 1096
        self.rfs_head = nn.Sequential(
            _fc_block(rfs_in_dim, 512, dropout),
            _fc_block(512, 256, dropout),
            nn.Linear(256, 1),
        )

    def forward(self, image, clinical_staging, clinical_rfs):
        B, dev = image.shape[0], image.device

        img_feat = (self.backbone(image)
                    if self.use_image
                    else torch.zeros(B, _DenseNetBackbone.FEATURE_DIM, device=dev))

        stg_clc = (self.stg_clc_enc(clinical_staging)
                   if self.use_clinical
                   else torch.zeros(B, self.clc_dim, device=dev))

        s        = self.staging_trunk(torch.cat([img_feat, stg_clc], dim=1))
        t_logits = self.t_head(s)
        n_logits = self.n_head(s)

        tn_soft = torch.cat([
            torch.softmax(t_logits.detach(), dim=1),
            torch.softmax(n_logits.detach(), dim=1),
        ], dim=1)

        rfs_clc = (self.rfs_clc_enc(clinical_rfs)
                   if self.use_clinical
                   else torch.zeros(B, self.clc_dim, device=dev))

        risk = self.rfs_head(
            torch.cat([img_feat, rfs_clc, tn_soft], dim=1)
        ).squeeze(-1)

        return t_logits, n_logits, risk
