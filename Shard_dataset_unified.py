#!/usr/bin/env python3
"""
Unified ShardDataset for ALL ZVTVC Modules — UPDATED
=====================================================

CHANGES from previous version:
- TimbreEncoderDataset auto-detects lean timbre shards (audio + speaker_id only)
- _find_shards for timbre prefers timbre_ prefix shards over normal shards
- get_safe_num_workers() checks available RAM and caps workers to prevent OOM
- Larger _max_cache for lean timbre shards (they're ~3-5x smaller)
- Removed runtime audio augmentation from TimbreEncoderDataset (done offline in shards)

=== MODULE COMPATIBILITY ===

Content Encoder:
    loader = create_dataloader(shard_dir, 'train', module='content')
    # Returns: audio, phoneme_labels, phoneme_confidence, speaker_id

Prosody Encoder:
    loader = create_dataloader(shard_dir, 'train', module='prosody')
    # Returns: features [B, 4, T] = (f0, energy, voicing, rhythm)

Timbre Encoder:
    loader = create_dataloader(shard_dir, 'train', module='timbre',
                               speakers_per_batch=32, utterances_per_speaker=2)
    # Returns: audio, speaker_id (batches are speaker-grouped for contrastive learning)

Generator:
    loader = create_dataloader(shard_dir, 'train', module='generator',
                               content_encoder=..., prosody_encoder=..., timbre_encoder=...)
    # Returns: mel, content, prosody, timbre
"""

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
from collections import defaultdict
import json
import random


# ============================================
# RAM-Adaptive Worker Count
# ============================================

def get_safe_num_workers(requested: int, shard_size_mb: float = 50.0,
                         max_cache: int = 10, reserve_gb: float = 4.0) -> int:
    """
    Cap num_workers based on available system RAM.
    
    Each forked DataLoader worker duplicates the shard cache in its own process.
    On Colab A100 with 83GB RAM, 4 workers x 200-shard cache x 150MB/shard = OOM.
    With lean timbre shards (~15MB each), 4 workers x 50 cached x 15MB = 3GB -> safe.
    
    Args:
        requested: Number of workers the user asked for
        shard_size_mb: Estimated average shard size in MB
        max_cache: Max number of shards cached per worker
        reserve_gb: GB to keep free for the main process + GPU
    
    Returns:
        Capped number of workers (may be less than requested)
    """
    if requested <= 0:
        return 0
    
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        usable_gb = max(0.0, avail_gb - reserve_gb)
        per_worker_gb = (shard_size_mb * max_cache) / 1024.0
        
        if per_worker_gb <= 0:
            return requested
        
        safe = max(0, int(usable_gb / per_worker_gb))
        capped = min(requested, safe)
        
        if capped < requested:
            print(f"[RAM CHECK] Available: {avail_gb:.1f}GB, usable: {usable_gb:.1f}GB, "
                  f"per-worker: {per_worker_gb:.1f}GB -> capping workers {requested} -> {capped}")
        else:
            print(f"[RAM CHECK] Available: {avail_gb:.1f}GB -- {requested} workers OK")
        
        return capped
    except ImportError:
        print(f"[RAM CHECK] psutil not available, using requested workers={requested}")
        return requested


# ============================================
# Base Shard Dataset
# ============================================

class BaseShardDataset(Dataset):
    """Base class for loading unified shards."""
    
    def __init__(self, shard_dir: str, split: str = 'train', lang: Optional[str] = None):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.lang = lang
        
        self.shard_files = self._find_shards()
        if not self.shard_files:
            raise ValueError(f"No shards for {split}/{lang} in {shard_dir}")
        
        self.index = self._build_index()
        self._cache: Dict[int, Dict] = {}
        self._max_cache = 200   
        
        self.metadata = self._load_metadata()
    
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
        """Build index: (shard_idx, utterance_idx) for each sample.
        
        Also populates self._shard_ranges: {shard_idx: (global_start, global_end)}
        which is used by ShardAwareSampler and SpeakerAwareBatchSampler.
        """
        index = []
        self._shard_ranges: Dict[int, Tuple[int, int]] = {}
        for shard_idx, shard_path in enumerate(self.shard_files):
            try:
                with np.load(shard_path, allow_pickle=True) as data:
                    # Detect format
                    if 'audio_flat' in data and 'audio_offsets' in data:
                        # Flat format
                        offsets = data['audio_offsets']
                        n_utts = len(offsets) - 1
                    elif 'audio' in data:
                        # Old object array format
                        n_utts = len(data['audio'])
                    elif 'mel' in data:
                        # Generator precomputed shards
                        n_utts = len(data['mel'])
                    else:
                        print(f"Warning: {shard_path} has no audio/mel field, skipping")
                        continue
                    
                    start_global = len(index)
                    for utt_idx in range(n_utts):
                        index.append((shard_idx, utt_idx))
                    self._shard_ranges[shard_idx] = (start_global, len(index))
            except Exception as e:
                print(f"Error reading {shard_path}: {e}")
                continue
        return index
    
    def _load_metadata(self) -> Optional[Dict]:
        meta_path = self.shard_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return None
    
    def _load_shard(self, idx: int) -> Dict:
        if idx in self._cache:
            # LRU touch: move to end
            self._cache[idx] = self._cache.pop(idx)
            return self._cache[idx]
        if len(self._cache) >= self._max_cache:
            # Evict oldest (first inserted) — true LRU since dicts are ordered in Py3.7+
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        data = dict(np.load(self.shard_files[idx], allow_pickle=True))
        self._cache[idx] = data
        return data
    
    def __len__(self) -> int:
        return len(self.index)


class ShardAwareSampler(Sampler):
    """Sampler that groups indices by shard to minimize shard loading."""
    
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
# Content Encoder Dataset
# ============================================

class ContentEncoderDataset(BaseShardDataset):
    """Dataset for Content Encoder: audio + phonemes."""
    
    def __init__(
        self, shard_dir: str, split: str = 'train', lang: Optional[str] = None,
        max_audio_samples: int = 96000, min_audio_samples: int = 32000,
    ):
        super().__init__(shard_dir, split, lang)
        self.max_audio_samples = max_audio_samples
        self.min_audio_samples = min_audio_samples
        self._build_speaker_mapping()
        print(f"[ContentEncoderDataset] {split}: {len(self)} samples, {self.num_speakers} speakers")
    
    def _build_speaker_mapping(self):
        speakers = set()
        for shard_path in self.shard_files:
            with np.load(shard_path, allow_pickle=True) as data:
                speakers.update(data['speaker_ids'].tolist())
        speakers = sorted(speakers)
        self.speaker_to_id = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(speakers)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, sample_idx = self.index[idx]
        shard = self._load_shard(shard_idx)
        
        audio = shard['audio'][sample_idx]
        phoneme_ids = shard['phoneme_ids'][sample_idx]
        phoneme_conf = shard['phoneme_conf'][sample_idx]
        speaker_name = shard['speaker_ids'][sample_idx]
        speaker_id = self.speaker_to_id.get(speaker_name, 0)
        
        hop = 320
        if len(audio) > self.max_audio_samples:
            start = random.randint(0, len(audio) - self.max_audio_samples)
            audio = audio[start:start + self.max_audio_samples]
            start_frame = start // hop
            n_frames = self.max_audio_samples // hop
            phoneme_ids = phoneme_ids[start_frame:start_frame + n_frames]
            phoneme_conf = phoneme_conf[start_frame:start_frame + n_frames]
        elif len(audio) < self.min_audio_samples:
            pad_len = self.min_audio_samples - len(audio)
            audio = np.pad(audio, (0, pad_len))
            n_frames_needed = self.min_audio_samples // hop
            if len(phoneme_ids) < n_frames_needed:
                pad_frames = n_frames_needed - len(phoneme_ids)
                phoneme_ids = np.pad(phoneme_ids, (0, pad_frames), constant_values=0)
                phoneme_conf = np.pad(phoneme_conf, (0, pad_frames), constant_values=0.0)
        
        return {
            'audio': torch.from_numpy(audio).float(),
            'phoneme_labels': torch.from_numpy(phoneme_ids).long(),
            'phoneme_confidence': torch.from_numpy(phoneme_conf).float(),
            'speaker_id': torch.tensor(speaker_id).long(),
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        max_audio = max(s['audio'].shape[0] for s in batch)
        max_ph = max(s['phoneme_labels'].shape[0] for s in batch)
        
        audio_batch = torch.zeros(B, max_audio)
        phoneme_batch = torch.zeros(B, max_ph, dtype=torch.long)
        conf_batch = torch.zeros(B, max_ph)
        speaker_batch = torch.zeros(B, dtype=torch.long)
        
        for i, s in enumerate(batch):
            audio_batch[i, :s['audio'].shape[0]] = s['audio']
            phoneme_batch[i, :s['phoneme_labels'].shape[0]] = s['phoneme_labels']
            conf_batch[i, :s['phoneme_confidence'].shape[0]] = s['phoneme_confidence']
            speaker_batch[i] = s['speaker_id']
        
        return {'audio': audio_batch, 'phoneme_labels': phoneme_batch,
                'phoneme_confidence': conf_batch, 'speaker_id': speaker_batch}


# ============================================
# Prosody Encoder Dataset
# ============================================

class ProsodyEncoderDataset(BaseShardDataset):
    """Dataset for Prosody Encoder: f0 + energy + voicing + rhythm (4 channels)."""
    
    def __init__(
        self, shard_dir: str, split: str = 'train', lang: Optional[str] = None,
        sequence_length: int = 500, use_augmentation: bool = False,
    ):
        super().__init__(shard_dir, split, lang)
        self.sequence_length = sequence_length
        self.use_augmentation = use_augmentation
        print(f"[ProsodyEncoderDataset] {split}: {len(self)} samples")
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        shard_idx, sample_idx = self.index[idx]
        shard = self._load_shard(shard_idx)
        
        f0 = shard['f0'][sample_idx]
        energy = shard['energy'][sample_idx]
        voicing = shard['voicing'][sample_idx]
        rhythm = shard['rhythm'][sample_idx]
        
        features = np.stack([f0, energy, voicing, rhythm], axis=0)
        
        if features.shape[1] < self.sequence_length:
            pad_len = self.sequence_length - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_len)), mode='edge')
        elif features.shape[1] > self.sequence_length:
            start = random.randint(0, features.shape[1] - self.sequence_length)
            features = features[:, start:start + self.sequence_length]
        
        features = torch.from_numpy(features).float()
        
        features_aug = None
        if self.use_augmentation:
            features_aug = features + torch.randn_like(features) * 0.1
        
        return features, features_aug
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, Optional[torch.Tensor]]]):
        features = torch.stack([item[0] for item in batch])
        features_aug = torch.stack([item[1] for item in batch]) if batch[0][1] is not None else None
        return features, features_aug


# ============================================
# Timbre Encoder Dataset — UPDATED
# ============================================

class SpeakerAwareBatchSampler(Sampler):
    """
    Shard-locality-aware speaker batch sampler for contrastive learning.
    
    Each batch contains exactly:
        speakers_per_batch x utterances_per_speaker samples
    
    Groups speakers by their dominant shard so each batch hits ~2-4 shards
    instead of ~28-30. Prevents cache thrashing.
    """
    
    def __init__(
        self,
        speaker_to_indices: Dict[str, List[int]],
        speakers_per_batch: int = 32,
        utterances_per_speaker: int = 2,
        drop_last: bool = True,
        index_to_shard: Optional[Dict[int, int]] = None,
    ):
        self.speaker_to_indices = speaker_to_indices
        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker
        self.drop_last = drop_last
        self.index_to_shard = index_to_shard or {}
        
        self.valid_speakers = [
            spk for spk, indices in speaker_to_indices.items()
            if len(indices) >= utterances_per_speaker
        ]
        
        if len(self.valid_speakers) < speakers_per_batch:
            print(f"[WARNING] Only {len(self.valid_speakers)} speakers have >= "
                  f"{utterances_per_speaker} utterances")
            print(f"          Reducing speakers_per_batch from {speakers_per_batch} "
                  f"to {len(self.valid_speakers)}")
            self.speakers_per_batch = len(self.valid_speakers)
        
        self.batch_size = self.speakers_per_batch * self.utterances_per_speaker
        self._build_shard_groups()
        
        total_utterances = sum(len(indices) for indices in speaker_to_indices.values())
        self._num_batches = total_utterances // self.batch_size
        
        print(f"[SpeakerAwareBatchSampler] {len(self.valid_speakers)} valid speakers, "
              f"{len(self.shard_groups)} shard groups")
        print(f"  Batch: {self.speakers_per_batch} speakers x {self.utterances_per_speaker} "
              f"utterances = {self.batch_size}")
        print(f"  ~{self._num_batches} batches per epoch")
    
    def _build_shard_groups(self):
        """Assign each speaker to their dominant shard and group speakers by shard."""
        self.shard_groups: Dict[int, List[str]] = defaultdict(list)
        self.speaker_shard_indices: Dict[str, Dict[int, List[int]]] = {}
        
        for spk in self.valid_speakers:
            indices = self.speaker_to_indices[spk]
            
            by_shard: Dict[int, List[int]] = defaultdict(list)
            for global_idx in indices:
                shard_idx = self.index_to_shard.get(global_idx, -1)
                by_shard[shard_idx].append(global_idx)
            
            self.speaker_shard_indices[spk] = dict(by_shard)
            
            if by_shard and (-1 not in by_shard or len(by_shard) > 1):
                valid_shards = {k: v for k, v in by_shard.items() if k != -1}
                if valid_shards:
                    dominant = max(valid_shards, key=lambda s: len(valid_shards[s]))
                else:
                    dominant = -1
            else:
                dominant = max(by_shard, key=lambda s: len(by_shard[s])) if by_shard else -1
            
            self.shard_groups[dominant].append(spk)
        
        self.shard_groups = dict(self.shard_groups)
        
        group_sizes = [len(spks) for spks in self.shard_groups.values()]
        if group_sizes:
            print(f"  Shard groups: {len(group_sizes)} groups, "
                  f"speakers/group: min={min(group_sizes)}, max={max(group_sizes)}, "
                  f"avg={sum(group_sizes)/len(group_sizes):.1f}")
    
    def __iter__(self) -> Iterator[List[int]]:
        shard_ids = list(self.shard_groups.keys())
        random.shuffle(shard_ids)
        
        speaker_stream = []
        for shard_id in shard_ids:
            group = self.shard_groups[shard_id].copy()
            random.shuffle(group)
            speaker_stream.extend(group)
        
        batches = []
        spk_idx = 0
        
        while spk_idx + self.speakers_per_batch <= len(speaker_stream):
            batch_speakers = speaker_stream[spk_idx:spk_idx + self.speakers_per_batch]
            
            batch_indices = []
            for spk in batch_speakers:
                selected = self._select_utterances_shard_aware(spk)
                batch_indices.extend(selected)
            
            batches.append(batch_indices)
            spk_idx += self.speakers_per_batch
        
        # Do NOT shuffle batches — shard groups are already shuffled above.
        # Shuffling destroys shard locality and causes ~3-5s decompress per batch.
        for batch in batches:
            yield batch
    
    def _select_utterances_shard_aware(self, speaker: str) -> List[int]:
        """Select utterances for a speaker, preferring indices from their dominant shard."""
        k = self.utterances_per_speaker
        shard_indices = self.speaker_shard_indices.get(speaker, {})
        
        if not shard_indices:
            all_indices = self.speaker_to_indices[speaker]
            if len(all_indices) >= k:
                return random.sample(all_indices, k)
            return random.choices(all_indices, k=k)
        
        sorted_shards = sorted(shard_indices.keys(),
                               key=lambda s: len(shard_indices[s]), reverse=True)
        
        selected = []
        for shard_id in sorted_shards:
            if len(selected) >= k:
                break
            available = shard_indices[shard_id]
            need = k - len(selected)
            if len(available) >= need:
                selected.extend(random.sample(available, need))
            else:
                selected.extend(available)
        
        if len(selected) < k:
            all_indices = self.speaker_to_indices[speaker]
            remaining = [idx for idx in all_indices if idx not in set(selected)]
            if remaining:
                selected.extend(random.sample(remaining, min(k - len(selected), len(remaining))))
        if len(selected) < k:
            selected = random.choices(self.speaker_to_indices[speaker], k=k)
        
        return selected[:k]
    
    def __len__(self) -> int:
        return self._num_batches


class TimbreEncoderDataset(BaseShardDataset):
    """
    Dataset for Timbre Encoder: audio + speaker_id.
    
    UPDATED:
    - Auto-detects lean timbre shards (only audio + speaker_ids, no phonemes etc)
    - Prefers timbre_ prefix shards over normal shards
    - Larger _max_cache for lean shards (they're ~3-5x smaller)
    - No runtime augmentation (augmentation is baked into shards offline)
    - Speaker-aware sampling via SpeakerAwareBatchSampler
    """
    
    def __init__(
        self, shard_dir: str, split: str = 'train', lang: Optional[str] = None,
        target_length: float = 10.0, sample_rate: int = 16000, is_training: bool = True,
        augmentation_config: Optional[Dict] = None,
    ):
        self._timbre_mode = True
        super().__init__(shard_dir, split, lang)
        
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.is_training = is_training
        self.target_samples = int(target_length * sample_rate)
        
        # Mel transform — computed per-sample in __getitem__ by DataLoader workers
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=320, win_length=1024,
            n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
        )
        
        # Detect if shards are lean (timbre-only) or full (normal shards)
        self._is_lean = self._detect_lean_shards()
        if self._is_lean:
            self._max_cache = 200   
        else:
            self._max_cache = 200   
        
        self._build_speaker_mapping()
        print(f"[TimbreEncoderDataset] {split}: {len(self)} samples, {self.num_speakers} speakers")
        
        # No runtime augmentation — it's baked into shards offline
        if augmentation_config and augmentation_config.get('enable_augmentation', False):
            print(f"[TimbreEncoderDataset] NOTE: Runtime augmentation is disabled. "
                  f"Use create_timbre_shards_direct.py --augment to bake augmentation "
                  f"into shards offline for fast training.")
    
    def _find_shards(self) -> List[Path]:
        """Find timbre_ prefix shards first, fall back to normal shards."""
        if not getattr(self, '_timbre_mode', False):
            return super()._find_shards()
        
        timbre_shards = []
        for f in sorted(self.shard_dir.glob("*.npz")):
            if f.stem.startswith(f'timbre_{self.split}'):
                timbre_shards.append(f)
        
        if timbre_shards:
            print(f"[TimbreEncoderDataset] Found {len(timbre_shards)} timbre-specific shards")
            return timbre_shards
        
        print(f"[TimbreEncoderDataset] No timbre_ shards found, using normal shards")
        return super()._find_shards()
    
    def _detect_lean_shards(self) -> bool:
        """Check if shards contain only audio + speaker_ids (lean) or all fields."""
        if not self.shard_files:
            return False
        try:
            with np.load(self.shard_files[0], allow_pickle=True) as data:
                keys = set(data.keys())
            return 'phoneme_ids' not in keys
        except Exception:
            return False
    
    def _build_speaker_mapping(self):
        """Build speaker_to_id AND speaker_to_indices mappings in a single pass."""
        speakers = set()
        self.speaker_to_indices: Dict[str, List[int]] = defaultdict(list)
        self.index_to_shard: Dict[int, int] = {}
        
        for shard_idx, shard_path in enumerate(self.shard_files):
            with np.load(shard_path, allow_pickle=True) as data:
                shard_speaker_ids = data['speaker_ids']
                start_global, _ = self._shard_ranges[shard_idx]
                for local_idx, speaker_name in enumerate(shard_speaker_ids):
                    speaker_name = str(speaker_name)
                    global_idx = start_global + local_idx
                    speakers.add(speaker_name)
                    self.speaker_to_indices[speaker_name].append(global_idx)
                    self.index_to_shard[global_idx] = shard_idx
        
        speakers = sorted(speakers)
        self.speaker_to_id = {s: i for i, s in enumerate(speakers)}
        self.num_speakers = len(speakers)
        self.speaker_to_indices = dict(self.speaker_to_indices)
        
        utt_counts = [len(indices) for indices in self.speaker_to_indices.values()]
        if utt_counts:
            print(f"  Utterances per speaker: min={min(utt_counts)}, max={max(utt_counts)}, "
                  f"avg={sum(utt_counts)/len(utt_counts):.1f}")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, sample_idx = self.index[idx]
        shard = self._load_shard(shard_idx)
        
        audio = shard['audio'][sample_idx]
        speaker_name = shard['speaker_ids'][sample_idx]
        speaker_id = self.speaker_to_id.get(str(speaker_name), 0)
        
        # Ensure float32 (fixes numpy object array crash)
        audio = np.asarray(audio, dtype=np.float32)
        
        # Crop/pad to target length
        if len(audio) > self.target_samples:
            if self.is_training:
                start = random.randint(0, len(audio) - self.target_samples)
            else:
                start = (len(audio) - self.target_samples) // 2
            audio = audio[start:start + self.target_samples]
        elif len(audio) < self.target_samples:
            audio = np.pad(audio, (0, self.target_samples - len(audio)))
        
        # Compute mel on CPU (workers do this in parallel — fast)
        waveform = torch.from_numpy(audio).float()
        with torch.no_grad():
            mel = self.mel_transform(waveform.unsqueeze(0))  # [1, 80, T']
            mel = torch.log(mel + 1e-6)
            # Per-sample normalization
            mean = mel.mean()
            std = mel.std() + 1e-6
            mel = (mel - mean) / std
            mel = mel.squeeze(0)  # [80, T']
        
        return {
            'mel': mel,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
        }
    
    def get_speaker_aware_sampler(
        self,
        speakers_per_batch: int = 32,
        utterances_per_speaker: int = 2
    ) -> SpeakerAwareBatchSampler:
        """Create a shard-locality-aware speaker batch sampler for contrastive learning."""
        return SpeakerAwareBatchSampler(
            speaker_to_indices=self.speaker_to_indices,
            speakers_per_batch=speakers_per_batch,
            utterances_per_speaker=utterances_per_speaker,
            index_to_shard=self.index_to_shard,
        )
    
    def get_num_speakers(self) -> int:
        return self.num_speakers
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        mel_batch = torch.stack([s['mel'] for s in batch])
        speaker_batch = torch.stack([s['speaker_id'] for s in batch])
        return {'mel': mel_batch, 'speaker_id': speaker_batch}
    
    @property
    def speaker_to_files(self) -> Dict[str, List[int]]:
        """Alias for speaker_to_indices for compatibility."""
        return self.speaker_to_indices


# ============================================
# Generator Dataset (uses trained encoders)
# ============================================

class GeneratorDataset(BaseShardDataset):
    """
    Dataset for Generator training.
    Returns: mel, content [512,T], prosody [32,T], timbre [256]
    """
    
    def __init__(
        self,
        shard_dir: str,
        split: str = 'train',
        lang: Optional[str] = None,
        content_encoder: Optional[torch.nn.Module] = None,
        prosody_encoder: Optional[torch.nn.Module] = None,
        timbre_encoder: Optional[torch.nn.Module] = None,
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
            sample_rate=sample_rate, n_fft=1024, hop_length=320, win_length=1024,
            n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
        )
        
        for enc in [content_encoder, prosody_encoder, timbre_encoder]:
            if enc is not None:
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad = False
        
        print(f"[GeneratorDataset] {split}: {len(self)} samples")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        shard_idx, sample_idx = self.index[idx]
        shard = self._load_shard(shard_idx)
        
        audio = shard['audio'][sample_idx]
        audio = np.asarray(audio, dtype=np.float32)
        audio_tensor = torch.from_numpy(audio).float()
        
        with torch.no_grad():
            mel = self.mel_transform(audio_tensor.unsqueeze(0))
            mel = torch.log(mel + 1e-6)
            mean = mel.mean(dim=(1, 2), keepdim=True)
            std = mel.std(dim=(1, 2), keepdim=True) + 1e-6
            mel = (mel - mean) / std
            mel = mel.squeeze(0)
        
        T = mel.shape[1]
        mel_batch = mel.unsqueeze(0)
        
        with torch.no_grad():
            if self.content_encoder is not None:
                content = self.content_encoder(mel_batch.to(self.device))
                if isinstance(content, tuple):
                    content = content[0]
                content = content.squeeze(0).cpu()
            else:
                content = torch.randn(512, T)
            
            if self.prosody_encoder is not None:
                f0 = torch.from_numpy(shard['f0'][sample_idx]).float().unsqueeze(0)
                energy = torch.from_numpy(shard['energy'][sample_idx]).float().unsqueeze(0)
                voicing = torch.from_numpy(shard['voicing'][sample_idx]).float().unsqueeze(0)
                rhythm = torch.from_numpy(shard['rhythm'][sample_idx]).float().unsqueeze(0)
                
                min_len = min(f0.shape[1], energy.shape[1], voicing.shape[1], rhythm.shape[1], T)
                prosody_features = torch.stack([
                    f0[:, :min_len], energy[:, :min_len],
                    voicing[:, :min_len], rhythm[:, :min_len],
                ], dim=0).unsqueeze(0).to(self.device)
                prosody = self.prosody_encoder(explicit_features=prosody_features)
                if isinstance(prosody, tuple):
                    prosody = prosody[0]
                prosody = prosody.squeeze(0).cpu()
            else:
                prosody = torch.randn(32, T)
            
            if self.timbre_encoder is not None:
                timbre = self.timbre_encoder(mel_batch.to(self.device))
                timbre = timbre.squeeze(0).cpu()
            else:
                timbre = torch.randn(256)
        
        mel, content, prosody = self._crop_sequence(mel, content, prosody)
        
        return {'mel': mel, 'content': content, 'prosody': prosody, 'timbre': timbre}
    
    def _crop_sequence(self, mel, content, prosody):
        T = mel.shape[1]
        if T <= self.sequence_length:
            pad = self.sequence_length - T
            mel = torch.nn.functional.pad(mel, (0, pad))
            content = torch.nn.functional.pad(content, (0, pad))
            prosody = torch.nn.functional.pad(prosody, (0, pad))
        else:
            start = random.randint(0, T - self.sequence_length)
            mel = mel[:, start:start + self.sequence_length]
            content = content[:, start:start + self.sequence_length]
            prosody = prosody[:, start:start + self.sequence_length]
        return mel, content, prosody
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            'mel': torch.stack([b['mel'] for b in batch]),
            'content': torch.stack([b['content'] for b in batch]),
            'prosody': torch.stack([b['prosody'] for b in batch]),
            'timbre': torch.stack([b['timbre'] for b in batch]),
        }


# ============================================
# Factory Function
# ============================================

def create_dataloader(
    shard_dir: str,
    split: str = 'train',
    module: str = 'content',
    lang: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: Optional[bool] = None,
    speakers_per_batch: int = 32,
    utterances_per_speaker: int = 2,
    content_encoder: Optional[torch.nn.Module] = None,
    prosody_encoder: Optional[torch.nn.Module] = None,
    timbre_encoder: Optional[torch.nn.Module] = None,
    augmentation_config: Optional[Dict] = None,
    **kwargs,
) -> DataLoader:
    """Create DataLoader for specific module."""
    if shuffle is None:
        shuffle = (split == 'train')
    
    if module == 'content':
        dataset = ContentEncoderDataset(shard_dir, split, lang, **kwargs)
        sampler = ShardAwareSampler(dataset, shuffle=shuffle) if shuffle else None
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
            num_workers=num_workers,
            pin_memory=True, collate_fn=ContentEncoderDataset.collate_fn,
            persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
        )
    
    elif module == 'prosody':
        dataset = ProsodyEncoderDataset(shard_dir, split, lang, **kwargs)
        sampler = ShardAwareSampler(dataset, shuffle=shuffle) if shuffle else None
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
            num_workers=num_workers,
            pin_memory=True, collate_fn=ProsodyEncoderDataset.collate_fn,
            persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
        )
    
    elif module == 'timbre':
        dataset = TimbreEncoderDataset(
            shard_dir, split, lang, is_training=(split == 'train'),
            augmentation_config=augmentation_config, **kwargs
        )
        
        if split == 'train':
            batch_sampler = dataset.get_speaker_aware_sampler(
                speakers_per_batch=speakers_per_batch,
                utterances_per_speaker=utterances_per_speaker,
            )
            return DataLoader(
                dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                pin_memory=True, collate_fn=TimbreEncoderDataset.collate_fn,
                persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
            )
        else:
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=True, collate_fn=TimbreEncoderDataset.collate_fn,
                persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
            )
    
    elif module == 'generator':
        dataset = GeneratorDataset(
            shard_dir, split, lang,
            content_encoder=content_encoder,
            prosody_encoder=prosody_encoder,
            timbre_encoder=timbre_encoder,
            **kwargs,
        )
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            pin_memory=True, collate_fn=GeneratorDataset.collate_fn,
            persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers > 0 else None,
        )
    
    else:
        raise ValueError(f"Unknown module: {module}. Choose 'content', 'prosody', 'timbre', or 'generator'")


# ============================================
# Test
# ============================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('shard_dir')
    parser.add_argument('--module', default='content', choices=['content', 'prosody', 'timbre', 'generator'])
    parser.add_argument('--split', default='train')
    parser.add_argument('--speakers_per_batch', type=int, default=6)
    parser.add_argument('--utterances_per_speaker', type=int, default=2)
    args = parser.parse_args()
    
    loader = create_dataloader(
        args.shard_dir, args.split, args.module, batch_size=4, num_workers=0,
        speakers_per_batch=args.speakers_per_batch,
        utterances_per_speaker=args.utterances_per_speaker,
    )
    
    print(f"\nModule: {args.module}")
    print(f"Dataset size: {len(loader.dataset)}")
    
    batch = next(iter(loader))
    
    if args.module == 'prosody':
        features, features_aug = batch
        print(f"Features: {features.shape}")
        print(f"Features aug: {features_aug}")
    else:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")
    
    if args.module == 'timbre':
        speaker_ids = batch['speaker_id'].tolist()
        unique_speakers = len(set(speaker_ids))
        print(f"\n  Batch has {len(speaker_ids)} samples from {unique_speakers} speakers")
        from collections import Counter
        counts = Counter(speaker_ids)
        print(f"  Utterances per speaker: {dict(counts)}")