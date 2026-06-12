"""
Dataset and DataLoader for Generator Training — FIXED VERSION
Supports TWO modes via --use_precomputed flag:

MODE 1: Precomputed (FAST — use this)
    Loads mel, content, prosody, timbre from pre-computed .npz shards.
    num_workers > 0 is safe (no GPU ops in __getitem__).
    ~3x faster than on-the-fly.

MODE 2: On-the-fly (SLOW — original behavior)
    Loads audio from raw shards, computes mel on CPU, runs frozen encoders on GPU.
    num_workers MUST be 0 (CUDA tensors can't cross process boundaries).

Returns per sample:
    mel:     [80, T]   - target mel-spectrogram (raw log-mel, NO per-utterance normalization)
    content: [512, T]  - from Content Encoder (fed log(clamp) mel, no normalization)
    prosody: [32, T]   - from Prosody Encoder
    timbre:  [256]     - from Timbre Encoder (fed per-utterance normalized mel)
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import json
import random
import logging

log = logging.getLogger(__name__)


# ============================================
# Base Shard Dataset (shared by both modes)
# ============================================

class BaseShardDataset(Dataset):
    """Base class for loading .npz shards."""

    def __init__(self, shard_dir: str, split: str = 'train', lang: Optional[str] = None):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.lang = lang

        self.shard_files = self._find_shards()
        if not self.shard_files:
            raise ValueError(f"No shards for {split}/{lang} in {shard_dir}")

        self.index = self._build_index()
        self._cache: Dict[int, Dict] = {}
        self._max_cache = 8  # keep more shards cached to avoid reload thrashing with workers

    def _find_shards(self) -> List[Path]:
        shards = []
        for f in sorted(self.shard_dir.glob("*.npz")):
            parts = f.stem.split('_')
            if parts[0] != self.split:
                continue
            if self.lang and len(parts) >= 2 and parts[1] != self.lang:
                continue
            shards.append(f)
        return shards

    def _build_index(self) -> List[Tuple[int, int]]:
        index = []
        self._shard_ranges = {}
        for shard_idx, path in enumerate(self.shard_files):
            with np.load(path, allow_pickle=True) as data:
                n = len(data['mel'] if 'mel' in data else data['audio'])
            start = len(index)
            for i in range(n):
                index.append((shard_idx, i))
            self._shard_ranges[shard_idx] = (start, len(index))
        return index

    def _load_shard(self, idx: int) -> Dict:
        if idx in self._cache:
            return self._cache[idx]
        if len(self._cache) >= self._max_cache:
            # Evict oldest
            del self._cache[min(self._cache.keys())]

        # Retry logic: forked DataLoader workers can get transient
        # zlib/read errors when multiple workers hit the same npz file
        import time as _time
        for attempt in range(3):
            try:
                data = dict(np.load(self.shard_files[idx], allow_pickle=True))
                self._cache[idx] = data
                return data
            except Exception as e:
                if attempt < 2:
                    _time.sleep(0.1 * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Failed to load shard {self.shard_files[idx]} after 3 attempts: {e}"
                    ) from e

    def __len__(self) -> int:
        return len(self.index)


class ShardAwareSampler(Sampler):
    """Groups indices by shard to minimize I/O. Shuffles shard order + within-shard."""

    def __init__(self, dataset: BaseShardDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self._shard_ranges = dataset._shard_ranges
        self._num_shards = len(self._shard_ranges)

    def __iter__(self) -> Iterator[int]:
        shard_order = list(range(self._num_shards))
        if self.shuffle:
            random.shuffle(shard_order)

        indices = []
        for shard_idx in shard_order:
            start, end = self._shard_ranges[shard_idx]
            shard_indices = list(range(start, end))
            if self.shuffle:
                random.shuffle(shard_indices)
            indices.extend(shard_indices)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


# ============================================
# MODE 1: Precomputed Dataset (FAST)
# ============================================

class PrecomputedGeneratorDataset(BaseShardDataset):
    """
    Loads pre-computed encoder outputs from .npz shards.
    No GPU ops — safe for num_workers > 0.

    Shard format (from precompute_encoder_outputs.py):
        mel:      [N] object array of float32 [80, T]
        content:  [N] object array of float32 [512, T]
        prosody:  [N] object array of float32 [32, T]
        timbre:   [N] object array of float32 [256]
        speaker_ids, paths
    """

    def __init__(
        self,
        shard_dir: str,
        split: str = 'train',
        lang: Optional[str] = None,
        sequence_length: int = 400,
    ):
        super().__init__(shard_dir, split, lang)
        self.sequence_length = sequence_length

        log.info(f"[PrecomputedGeneratorDataset] {split}: {len(self)} samples, "
                 f"{len(self.shard_files)} shards, seq_len={sequence_length}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, sample_idx = self.index[idx]
        shard = self._load_shard(shard_idx)

        # FIX: np.asarray with explicit dtype handles object arrays
        mel = torch.from_numpy(np.asarray(shard['mel'][sample_idx], dtype=np.float32))
        content = torch.from_numpy(np.asarray(shard['content'][sample_idx], dtype=np.float32))
        prosody = torch.from_numpy(np.asarray(shard['prosody'][sample_idx], dtype=np.float32))
        timbre = torch.from_numpy(np.asarray(shard['timbre'][sample_idx], dtype=np.float32))

        # Align temporal lengths (should already match, but be safe)
        T = mel.shape[1]
        content = _align_temporal(content, T)
        prosody = _align_temporal(prosody, T)

        # Crop/pad to sequence_length
        mel, content, prosody = _crop_sequence(mel, content, prosody, self.sequence_length)

        return {
            'mel': mel,         # [80, seq_len]
            'content': content, # [512, seq_len]
            'prosody': prosody, # [32, seq_len]
            'timbre': timbre,   # [256]
        }


# ============================================
# MODE 2: On-the-fly Dataset (SLOW, original)
# ============================================

class OnTheFlyGeneratorDataset(BaseShardDataset):
    """
    Original behavior: loads audio from raw shards, computes mel,
    runs frozen encoders on GPU in __getitem__.
    num_workers MUST be 0.
    """

    def __init__(
        self,
        shard_dir: str,
        split: str = 'train',
        lang: Optional[str] = None,
        content_encoder: Optional[nn.Module] = None,
        prosody_encoder: Optional[nn.Module] = None,
        timbre_encoder: Optional[nn.Module] = None,
        sequence_length: int = 400,
        sample_rate: int = 16000,
        device: str = 'cuda',
    ):
        super().__init__(shard_dir, split, lang)
        self.content_encoder = content_encoder
        self.prosody_encoder = prosody_encoder
        self.timbre_encoder = timbre_encoder
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.device = device

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=320,
            win_length=1024, n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
        )

        for enc in [content_encoder, prosody_encoder, timbre_encoder]:
            if enc is not None:
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad = False

        log.info(f"[OnTheFlyGeneratorDataset] {split}: {len(self)} samples, "
                 f"{len(self.shard_files)} shards")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, sample_idx = self.index[idx]
        shard = self._load_shard(shard_idx)

        # FIX: np.asarray with explicit dtype to avoid object array crash
        audio = np.asarray(shard['audio'][sample_idx], dtype=np.float32)
        f0 = np.asarray(shard['f0'][sample_idx], dtype=np.float32)
        energy = np.asarray(shard['energy'][sample_idx], dtype=np.float32)
        voicing = np.asarray(shard['voicing'][sample_idx], dtype=np.float32)
        rhythm = np.asarray(shard['rhythm'][sample_idx], dtype=np.float32)

        # Compute mel on CPU — three variants for three different consumers
        waveform = torch.from_numpy(audio)
        mel_raw = self.mel_transform(waveform)   # [80, T] linear power mel

        # Generator target: raw log-mel, no normalization (vocoder-ready scale)
        mel_target = torch.log(mel_raw + 1e-6)

        # Content Encoder input: log(clamp), no per-utterance normalization
        # Matches Content_Encoder/train.py:adapt_batch_for_content_encoder exactly
        mel_for_ce = torch.log(mel_raw.clamp(min=1e-5))

        # Timbre Encoder input: log + per-utterance normalization
        # Matches Shard_dataset_unified.py:TimbreEncoderDataset (Timbre Encoder training)
        mel_for_timbre = torch.log(mel_raw + 1e-6)
        mel_for_timbre = (mel_for_timbre - mel_for_timbre.mean()) / (mel_for_timbre.std() + 1e-6)

        T = mel_target.shape[1]

        with torch.no_grad():
            # Content encoder — receives log(clamp) mel, no normalization
            if self.content_encoder is not None:
                content = self.content_encoder(mel_for_ce.unsqueeze(0).to(self.device))
                if isinstance(content, tuple):
                    content = content[0]
                content = content.squeeze(0).cpu()
            else:
                content = torch.randn(512, T)

            # Prosody encoder
            if self.prosody_encoder is not None:
                prosody_features = torch.stack([
                    torch.from_numpy(f0), torch.from_numpy(energy),
                    torch.from_numpy(voicing), torch.from_numpy(rhythm),
                ], dim=0).unsqueeze(0).to(self.device)
                prosody = self.prosody_encoder(explicit_features=prosody_features)
                if isinstance(prosody, tuple):
                    prosody = prosody[0]
                prosody = prosody.squeeze(0).cpu()
            else:
                prosody = torch.randn(32, T)

            # Timbre encoder — receives per-utterance normalized mel
            if self.timbre_encoder is not None:
                timbre = self.timbre_encoder(mel_for_timbre.unsqueeze(0).to(self.device))
                if isinstance(timbre, tuple):
                    timbre = timbre[0]
                timbre = timbre.squeeze(0).cpu()
            else:
                timbre = torch.randn(256)

        content = _align_temporal(content, T)
        prosody = _align_temporal(prosody, T)

        mel, content, prosody = _crop_sequence(mel_target, content, prosody, self.sequence_length)

        return {
            'mel': mel,
            'content': content,
            'prosody': prosody,
            'timbre': timbre,
        }


# ============================================
# Helpers
# ============================================

def _align_temporal(x: torch.Tensor, target_T: int) -> torch.Tensor:
    T = x.shape[-1]
    if T == target_T:
        return x
    elif T < target_T:
        return torch.nn.functional.pad(x, (0, target_T - T))
    else:
        return x[..., :target_T]


def _crop_sequence(
    mel: torch.Tensor,
    content: torch.Tensor,
    prosody: torch.Tensor,
    sequence_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = mel.shape[1]
    if T <= sequence_length:
        pad = sequence_length - T
        mel = torch.nn.functional.pad(mel, (0, pad))
        content = torch.nn.functional.pad(content, (0, pad))
        prosody = torch.nn.functional.pad(prosody, (0, pad))
    else:
        start = random.randint(0, T - sequence_length)
        mel = mel[:, start:start + sequence_length]
        content = content[:, start:start + sequence_length]
        prosody = prosody[:, start:start + sequence_length]
    return mel, content, prosody


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack batch into tensors."""
    return {
        'mel': torch.stack([b['mel'] for b in batch]),
        'content': torch.stack([b['content'] for b in batch]),
        'prosody': torch.stack([b['prosody'] for b in batch]),
        'timbre': torch.stack([b['timbre'] for b in batch]),
    }


# ============================================
# DataLoader Factory
# ============================================

def get_dataloader(
    dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    use_shard_sampler: bool = True,
    **kwargs,
) -> DataLoader:
    sampler = None
    if shuffle and use_shard_sampler:
        sampler = ShardAwareSampler(dataset, shuffle=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        **kwargs,
    )


# ============================================
# Encoder Loading (only needed for on-the-fly mode)
# ============================================

def _import_from_sibling(module_folder: str, import_path: str, class_name: str):
    """Import a class from a sibling module folder."""
    import sys
    import importlib
    import importlib.util

    module_folder = Path(module_folder)
    old_path = sys.path.copy()
    saved_modules = {}
    for key in list(sys.modules.keys()):
        if key == 'model' or key.startswith('model.') or key == 'utils' or key.startswith('utils.'):
            saved_modules[key] = sys.modules.pop(key)

    try:
        sys.path.insert(0, str(module_folder))
        model_dir = module_folder / import_path.split('.')[0]
        has_init = (model_dir / '__init__.py').exists()

        if has_init:
            mod = importlib.import_module(import_path)
            cls = getattr(mod, class_name)
            return cls
        else:
            parts = import_path.split('.')
            package_name = parts[0]
            module_name = parts[1]

            package_spec = importlib.util.spec_from_file_location(
                package_name, str(model_dir),
                submodule_search_locations=[str(model_dir)]
            )
            if package_spec is None:
                import types
                pkg = types.ModuleType(package_name)
                pkg.__path__ = [str(model_dir)]
                pkg.__package__ = package_name
                sys.modules[package_name] = pkg
            else:
                pkg = importlib.util.module_from_spec(package_spec)
                pkg.__path__ = [str(model_dir)]
                sys.modules[package_name] = pkg

            file_path = model_dir / f'{module_name}.py'
            spec = importlib.util.spec_from_file_location(
                import_path, str(file_path),
                submodule_search_locations=[str(model_dir)]
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[import_path] = mod
            spec.loader.exec_module(mod)
            cls = getattr(mod, class_name)
            return cls
    finally:
        sys.path = old_path
        for key in list(sys.modules.keys()):
            if key == 'model' or key.startswith('model.') or key == 'utils' or key.startswith('utils.'):
                del sys.modules[key]
        sys.modules.update(saved_modules)


# Keep the encoder loading functions for backward compatibility (on-the-fly mode)

class ContentEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, mel):
        out = self.encoder(mel)
        return out[0] if isinstance(out, tuple) else out


class ProsodyEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, explicit_features):
        out = self.encoder(explicit_features=explicit_features)
        return out[0] if isinstance(out, tuple) else out


class TimbreEncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, mel):
        out = self.encoder(mel)
        return out[0] if isinstance(out, tuple) else out


class DummyEncoderWrapper(nn.Module):
    def __init__(self, output_dim: int, output_type: str = 'frame'):
        super().__init__()
        self.output_dim = output_dim
        self.output_type = output_type

    def forward(self, *args, **kwargs):
        ref = None
        for a in args:
            if isinstance(a, torch.Tensor):
                ref = a
                break
        if ref is None:
            for v in kwargs.values():
                if isinstance(v, torch.Tensor):
                    ref = v
                    break
        if ref is None:
            raise ValueError("DummyEncoderWrapper needs at least one tensor input")
        B = ref.shape[0]
        device = ref.device
        if self.output_type == 'frame':
            T = ref.shape[-1]
            return torch.randn(B, self.output_dim, T, device=device)
        else:
            return torch.randn(B, self.output_dim, device=device)


def load_content_encoder(checkpoint_path: str, device: str = 'cuda') -> ContentEncoderWrapper:
    generator_dir = Path(__file__).resolve().parent.parent
    project_root = generator_dir.parent
    content_encoder_dir = project_root / 'Content Encoder'
    if not content_encoder_dir.exists():
        content_encoder_dir = project_root / 'Content_Encoder'

    ContentEncoder = _import_from_sibling(
        module_folder=str(content_encoder_dir),
        import_path='model.content_encoder', class_name='ContentEncoder',
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = ContentEncoder()

    if 'encoder' in ckpt:
        state_dict = ckpt['encoder']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    model_keys = set(encoder.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys != ckpt_keys and 'preprocess_conv.conv.weight' in ckpt_keys:
        remapped = {}
        for old_key, value in state_dict.items():
            new_key = old_key
            new_key = new_key.replace('preprocess_conv.', 'preprocessing.conv.')
            new_key = new_key.replace('preprocess_norm.', 'preprocessing.layer_norm.')
            new_key = new_key.replace('backbone.', 'multi_scale.')
            new_key = new_key.replace('output_proj.', 'output_projection.conv.')
            new_key = new_key.replace('output_norm.', 'output_projection.norm.')
            remapped[new_key] = value
        state_dict = remapped

    incompatible = encoder.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"  [WARN CE] {len(incompatible.missing_keys)} missing keys "
              f"(first 5: {incompatible.missing_keys[:5]})")
    if incompatible.unexpected_keys:
        print(f"  [WARN CE] {len(incompatible.unexpected_keys)} unexpected keys "
              f"(first 5: {incompatible.unexpected_keys[:5]})")
    encoder = encoder.to(device)
    return ContentEncoderWrapper(encoder)


def load_prosody_encoder(checkpoint_path: str, device: str = 'cuda') -> ProsodyEncoderWrapper:
    generator_dir = Path(__file__).resolve().parent.parent
    project_root = generator_dir.parent
    prosody_encoder_dir = project_root / 'Prosody Encoder'
    if not prosody_encoder_dir.exists():
        prosody_encoder_dir = project_root / 'Prosody_Encoder'

    ProsodyEncoder = _import_from_sibling(
        module_folder=str(prosody_encoder_dir),
        import_path='model.prosody_encoder', class_name='ProsodyEncoder',
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # use_reconstruction_heads=True matches the training default in Prosody_Encoder/train.py
    encoder = ProsodyEncoder(
        sample_rate=16000, hop_length=320,
        explicit_dim=4, refined_dim=32,
        use_refinement=True, use_residual=True,
        use_reconstruction_heads=True, output_format='refined',
    )

    if 'model_state_dict' in ckpt:
        incompatible = encoder.load_state_dict(ckpt['model_state_dict'], strict=False)
    elif 'encoder' in ckpt:
        incompatible = encoder.load_state_dict(ckpt['encoder'], strict=False)
    else:
        incompatible = encoder.load_state_dict(ckpt, strict=False)

    if incompatible.missing_keys:
        print(f"  [WARN PE] {len(incompatible.missing_keys)} missing keys "
              f"(first 5: {incompatible.missing_keys[:5]})")
    if incompatible.unexpected_keys:
        print(f"  [WARN PE] {len(incompatible.unexpected_keys)} unexpected keys "
              f"(first 5: {incompatible.unexpected_keys[:5]})")
    encoder = encoder.to(device)
    return ProsodyEncoderWrapper(encoder)


def load_timbre_encoder(checkpoint_path: str, device: str = 'cuda') -> TimbreEncoderWrapper:
    generator_dir = Path(__file__).resolve().parent.parent
    project_root = generator_dir.parent
    timbre_encoder_dir = project_root / 'Timbre Encoder'
    if not timbre_encoder_dir.exists():
        timbre_encoder_dir = project_root / 'Timbre_Encoder'

    for logger_name in ['model.ecapa_tdnn', 'model.ecapa_tdnn.pooling']:
        l = logging.getLogger(logger_name)
        l.propagate = False
        l.handlers.clear()
        l.setLevel(logging.WARNING)
        l.addHandler(logging.NullHandler())

    ECAPATDNN = _import_from_sibling(
        module_folder=str(timbre_encoder_dir),
        import_path='model.ecapa_tdnn', class_name='ECAPATDNN',
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    encoder = ECAPATDNN()

    if 'model_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        encoder.load_state_dict(ckpt['model'])
    elif 'encoder' in ckpt:
        encoder.load_state_dict(ckpt['encoder'])
    else:
        encoder.load_state_dict(ckpt)

    encoder = encoder.to(device)
    return TimbreEncoderWrapper(encoder)