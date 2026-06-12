"""
training package

This package intentionally avoids importing heavy submodules (trainer, losses, distillation, etc.)
at import-time to prevent circular-import issues such as:

    ImportError: cannot import name 'Trainer' from partially initialized module 'training.trainer'

Best practice in this codebase:
    - Import submodules explicitly, e.g.:
        from training.trainer import Trainer, ModeCollapseDetector
        from training.losses import MultiTaskLoss, LossScheduler
"""

from __future__ import annotations

import importlib
from typing import Any

# Public symbols (lazy-loaded via __getattr__)
__all__ = [
    # trainer
    "Trainer",
    "ModeCollapseDetector",

    # dataset helpers (compat)
    "AudioDataset",
    "collate_fn",

    # losses
    "MultiTaskLoss",
    "LossScheduler",
    "MetricsTracker",

    # distillation (optional)
    "DistillationLoss",
    "EMATeacher",
    "ProjectionLayer",
    "WhisperFeatureExtractor",
    "MHubertFeatureExtractor",

    # contrastive / consistency (optional)
    "ContrastiveLoss",
    "SimCLRLoss",
    "NTXentLoss",
    "ConsistencyLoss",
    "ConsistencyWrapper",
]

_LAZY = {
    # trainer
    "Trainer": ("training.trainer", "Trainer"),
    "ModeCollapseDetector": ("training.trainer", "ModeCollapseDetector"),

    # dataset
    "AudioDataset": ("training.dataset", "AudioDataset"),
    "collate_fn": ("training.dataset", "collate_fn"),

    # losses
    "MultiTaskLoss": ("training.losses", "MultiTaskLoss"),
    "LossScheduler": ("training.losses", "LossScheduler"),
    "MetricsTracker": ("training.losses", "MetricsTracker"),

    # distillation
    "DistillationLoss": ("training.distillation", "DistillationLoss"),
    "EMATeacher": ("training.distillation", "EMATeacher"),
    "ProjectionLayer": ("training.distillation", "ProjectionLayer"),
    "WhisperFeatureExtractor": ("training.distillation", "WhisperFeatureExtractor"),
    "MHubertFeatureExtractor": ("training.distillation", "MHubertFeatureExtractor"),

    # contrastive / consistency
    "ContrastiveLoss": ("training.contrastive", "ContrastiveLoss"),
    "SimCLRLoss": ("training.contrastive", "SimCLRLoss"),
    "NTXentLoss": ("training.contrastive", "NTXentLoss"),
    "ConsistencyLoss": ("training.consistency", "ConsistencyLoss"),
    "ConsistencyWrapper": ("training.consistency", "ConsistencyWrapper"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    raise AttributeError(f"module 'training' has no attribute {name!r}")
