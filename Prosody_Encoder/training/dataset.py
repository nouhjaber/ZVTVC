"""
Dataset compatibility shim for Prosody Encoder.

The real dataset is ProsodyEncoderDataset from Shard_dataset_unified.py.
This file provides gather_audio_files and collate_fn for backward
compatibility with scripts that import from here.
"""

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

# Re-export collate_fn matching Prosody format: (features, features_aug)
def collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]):
    """Collate function for Prosody Encoder DataLoader."""
    features = torch.stack([item[0] for item in batch])
    if batch[0][1] is not None:
        features_aug = torch.stack([item[1] for item in batch])
    else:
        features_aug = None
    return features, features_aug


def gather_audio_files(data_dir: str, extensions: List[str] = None) -> List[str]:
    """Recursively gather audio files from directory."""
    if extensions is None:
        extensions = [".wav", ".flac", ".mp3"]
    audio_paths = []
    for ext in extensions:
        audio_paths.extend(Path(data_dir).rglob(f"*{ext}"))
    return [str(p) for p in audio_paths]


__all__ = ['collate_fn', 'gather_audio_files']
