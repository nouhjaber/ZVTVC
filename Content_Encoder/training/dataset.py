"""
Dataset compatibility shim for Content Encoder.

The real dataset is ContentEncoderDataset from Shard_dataset_unified.py.
This file provides AudioDataset and collate_fn as aliases so that
training/trainer.py (parent Trainer class) can still import them
without breaking.
"""

import sys
from pathlib import Path

# Add ZVTVC root to path so we can import the shard dataset
zvtvc_root = Path(__file__).parent.parent.parent
if str(zvtvc_root) not in sys.path:
    sys.path.insert(0, str(zvtvc_root))

from Shard_dataset_unified import ContentEncoderDataset as AudioDataset

# Re-export collate_fn from the shard dataset
collate_fn = AudioDataset.collate_fn

__all__ = ['AudioDataset', 'collate_fn']
