#!/usr/bin/env python3
"""
UNIFIED Shard Preprocessor for ZVTVC Training - WITH RESUME SUPPORT

Creates shards that work for ALL modules:
- Content Encoder: audio + phonemes (78-class IPA)
- Prosody Encoder: f0 + energy + voicing + rhythm
- Timbre Encoder: audio + speaker_id (mel computed on-the-fly)

Resume capability: --resume flag allows interrupting and restarting processing.
"""

import os
import argparse
import numpy as np
import torch
import soundfile as sf
import librosa
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import json
from tqdm import tqdm
import warnings
import logging
import re
from collections import defaultdict
import random

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)


@dataclass  
class Config:
    sample_rate: int = 16000
    max_duration: float = 10.0
    hop_length: int = 320
    f0_fmin: float = 50.0
    f0_fmax: float = 600.0
    samples_per_shard: int = 2000
    batch_size: int = 64
    io_workers: int = 4
    device: str = "cuda"


# ============================================
# Phoneme Extractor (78-class IPA vocab) 
# ============================================

class PhonemeExtractor:
    """Real phoneme extraction with unified IPA vocab (78 classes)."""
    
    MODEL_CONFIGS = {
        "en": {"model_name": "facebook/wav2vec2-lv-60-espeak-cv-ft", "sample_rate": 16000},
        "ar": {"model_name": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic", "sample_rate": 16000},
    }
    
    IPA_PHONEMES = [
        "<SIL>", "<BRE>", "<HES>", "<WB>", "<UNK>", "<NOI>", "<LAU>", "<COU>", "<SNI>", "<PAU>",
        "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
        "m", "n", "ŋ", "l", "r", "w", "j", "tʃ", "dʒ",
        "iː", "ɪ", "e", "æ", "ɑː", "ɒ", "ɔː", "ʊ", "uː", "ʌ", "ɜː", "ə",
        "eɪ", "aɪ", "ɔɪ", "aʊ", "əʊ", "ɪə", "eə", "ʊə",
        "q", "ʔ", "x", "ɣ", "ħ", "ʕ", "tˤ", "dˤ", "sˤ", "ðˤ",
        "a", "aː", "i", "u", "o", "ts", "ɛ", "ɔ", "ɾ", "ʁ", "χ", "ɫ", "ɑ", "oː",
    ]
    
    def __init__(self, languages: List[str] = ["en", "ar"], device: str = "cuda"):
        self.device = device
        self.languages = languages
        self.models = {}
        self.processors = {}
        
        seen = set()
        unique = [p for p in self.IPA_PHONEMES if p not in seen and not seen.add(p)]
        self.phoneme_to_idx = {p: i for i, p in enumerate(unique)}
        self.num_phonemes = len(unique)
        log.info(f"[PhonemeExtractor] Unified vocab: {self.num_phonemes} classes")
    
    def load_model(self, lang: str):
        if lang in self.models:
            return
        if lang not in self.MODEL_CONFIGS:
            lang = "en"
        
        model_name = self.MODEL_CONFIGS[lang]["model_name"]
        log.info(f"[PhonemeExtractor] Loading {lang}: {model_name}")
        
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            
            self.processors[lang] = processor
            self.models[lang] = model
            log.info(f"[OK] {lang} phoneme model loaded")
        except Exception as e:
            log.error(f"[FAIL] Could not load {lang} model: {e}")
            # Load a simpler model as fallback
            self._load_fallback_model(lang)
    
    def _load_fallback_model(self, lang: str):
        """Load a simpler fallback model when the main one fails."""
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            
            # Try a more generic model
            fallback_model = "facebook/wav2vec2-base-960h" if lang == "en" else "facebook/wav2vec2-large-xlsr-53"
            
            log.info(f"[PhonemeExtractor] Loading fallback model: {fallback_model}")
            processor = Wav2Vec2Processor.from_pretrained(fallback_model)
            model = Wav2Vec2ForCTC.from_pretrained(fallback_model).to(self.device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            
            self.processors[lang] = processor
            self.models[lang] = model
            log.info(f"[OK] {lang} fallback model loaded")
        except Exception as e:
            log.error(f"[FAIL] Could not load fallback model for {lang}: {e}")
    
    def load_all_models(self):
        for lang in self.languages:
            self.load_model(lang)
    
    @torch.inference_mode()
    def extract_batch(self, audios: List[np.ndarray], lang: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if lang not in self.models:
            self.load_model(lang)
        
        if lang not in self.models:
            # Return dummy phonemes if model failed to load
            results_ids, results_conf = [], []
            for audio in audios:
                n_frames = len(audio) // 320 + 1
                results_ids.append(np.full(n_frames, self.phoneme_to_idx.get("<UNK>", 4), dtype=np.int64))
                results_conf.append(np.zeros(n_frames, dtype=np.float32))
            return results_ids, results_conf
        
        processor = self.processors[lang]
        model = self.models[lang]
        
        # Process ONE BY ONE to avoid shape issues
        results_ids, results_conf = [], []
        
        for audio in audios:
            # Single audio processing
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=False)
            input_values = inputs.input_values.to(self.device)  # [1, T]
            
            logits = model(input_values).logits  # [1, frames, vocab]
            probs = torch.softmax(logits, dim=-1)
            
            pred_ids = probs.argmax(dim=-1).squeeze(0).cpu().numpy()  # [frames]
            conf = probs.max(dim=-1).values.squeeze(0).cpu().numpy()  # [frames]
            
            mapped_ids = self._map_to_unified_vocab(pred_ids, lang)
            results_ids.append(mapped_ids)
            results_conf.append(conf.astype(np.float32))
        
        return results_ids, results_conf
    
    # Arabic script → IPA mapping (Standard Arabic phonology)
    ARABIC_TO_IPA = {
        'ء': 'ʔ',   'آ': 'aː',  'أ': 'ʔ',   'ؤ': 'ʔ',   'إ': 'ʔ',   'ئ': 'ʔ',
        'ا': 'aː',  'ب': 'b',   'ة': 'a',   'ت': 't',   'ث': 'θ',
        'ج': 'dʒ',  'ح': 'ħ',   'خ': 'x',   'د': 'd',   'ذ': 'ð',
        'ر': 'r',   'ز': 'z',   'س': 's',   'ش': 'ʃ',   'ص': 'sˤ',
        'ض': 'dˤ',  'ط': 'tˤ',  'ظ': 'ðˤ',  'ع': 'ʕ',   'غ': 'ɣ',
        'ف': 'f',   'ق': 'q',   'ك': 'k',   'ل': 'l',   'م': 'm',
        'ن': 'n',   'ه': 'h',   'و': 'w',   'ي': 'j',
        'َ': 'a',   'ُ': 'u',   'ِ': 'i',   'ً': 'a',   'ٌ': 'u',   'ٍ': 'i',
        'ّ': '',     'ْ': '',     'ـ': '',
    }

    def _map_to_unified_vocab(self, model_ids: np.ndarray, lang: str) -> np.ndarray:
        processor = self.processors.get(lang)
        vocab = {}
        if hasattr(processor, 'tokenizer') and hasattr(processor.tokenizer, 'get_vocab'):
            vocab = processor.tokenizer.get_vocab()
        
        if not vocab:
            return np.full_like(model_ids, self.phoneme_to_idx.get("<UNK>", 4), dtype=np.int64)
        
        id_to_token = {v: k for k, v in vocab.items()}
        unified_ids = np.zeros_like(model_ids, dtype=np.int64)
        
        # CTC blank/pad token ID (usually 0 in wav2vec2 models)
        pad_token_id = vocab.get("<pad>", 0)
        
        # Track the last real (non-blank) phoneme for CTC repeat
        last_real_phoneme = self.phoneme_to_idx.get("<SIL>", 0)
        
        for i, model_id in enumerate(model_ids):
            mid = int(model_id)
            
            # CTC blank frame → repeat last real phoneme
            if mid == pad_token_id:
                unified_ids[i] = last_real_phoneme
                continue
            
            token = id_to_token.get(mid, "<UNK>")
            token = token.replace("|", "").replace("<s>", "").replace("</s>", "").strip()
            
            # Actual silence/special tokens (NOT CTC blank)
            if not token or token in ["<s>", "</s>", "-"]:
                unified_ids[i] = self.phoneme_to_idx.get("<SIL>", 0)
                last_real_phoneme = self.phoneme_to_idx.get("<SIL>", 0)
            # Word boundary → SIL (brief pause between words)
            elif token in ["|", " "]:
                unified_ids[i] = self.phoneme_to_idx.get("<SIL>", 0)
                last_real_phoneme = self.phoneme_to_idx.get("<SIL>", 0)
            elif token in self.phoneme_to_idx:
                # Direct IPA match (works for English espeak model)
                unified_ids[i] = self.phoneme_to_idx[token]
                last_real_phoneme = unified_ids[i]
            elif token.lower() in self.phoneme_to_idx:
                unified_ids[i] = self.phoneme_to_idx[token.lower()]
                last_real_phoneme = unified_ids[i]
            else:
                # Try Arabic char → IPA mapping
                mapped = False
                for char in token:
                    ipa = self.ARABIC_TO_IPA.get(char)
                    if ipa and ipa in self.phoneme_to_idx:
                        unified_ids[i] = self.phoneme_to_idx[ipa]
                        last_real_phoneme = unified_ids[i]
                        mapped = True
                        break
                if not mapped:
                    # Single-char fallback
                    if len(token) > 0 and token[0] in self.phoneme_to_idx:
                        unified_ids[i] = self.phoneme_to_idx[token[0]]
                        last_real_phoneme = unified_ids[i]
                    else:
                        unified_ids[i] = self.phoneme_to_idx.get("<UNK>", 4)
        
        return unified_ids


# ============================================
# Prosody Feature Extractor - GPU torchcrepe
# ============================================

class ProsodyExtractor:
    """Extract prosody features using GPU torchcrepe."""
    
    def __init__(self, sample_rate: int = 16000, hop_length: int = 320, 
                 f0_fmin: float = 50.0, f0_fmax: float = 600.0, device: str = "cuda"):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_fmin = f0_fmin
        self.f0_fmax = f0_fmax
        self.device = device
        
        self.has_crepe = False
        try:
            import torchcrepe
            self.torchcrepe = torchcrepe
            self.has_crepe = True
            log.info("[ProsodyExtractor] Using torchcrepe (GPU)")
        except ImportError:
            log.warning("[ProsodyExtractor] torchcrepe not found, using pyin (CPU)")
    
    def extract_batch(self, audios: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
        """Extract prosody features for batch."""
        results = {'f0': [], 'energy': [], 'voicing': [], 'rhythm': []}
        
        for audio in audios:
            f0, voicing = self._extract_f0(audio)
            energy = self._extract_energy(audio, len(f0))
            rhythm = self._extract_rhythm(voicing)
            
            # Normalize
            f0_log = np.log(f0 + 1e-8)
            f0_norm = (f0_log - np.mean(f0_log)) / (np.std(f0_log) + 1e-8)
            
            energy_log = np.log(energy + 1e-8)
            energy_norm = (energy_log - np.mean(energy_log)) / (np.std(energy_log) + 1e-8)
            
            # Align lengths
            min_len = min(len(f0_norm), len(energy_norm), len(voicing), len(rhythm))
            
            results['f0'].append(f0_norm[:min_len].astype(np.float32))
            results['energy'].append(energy_norm[:min_len].astype(np.float32))
            results['voicing'].append(voicing[:min_len].astype(np.float32))
            results['rhythm'].append(rhythm[:min_len].astype(np.float32))
        
        return results
    
    @torch.inference_mode()
    def _extract_f0(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract F0 using torchcrepe (GPU) or pyin (CPU)."""
        
        if self.has_crepe:
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            
            f0, periodicity = self.torchcrepe.predict(
                audio_tensor,
                self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.f0_fmin,
                fmax=self.f0_fmax,
                model='tiny',
                decoder=self.torchcrepe.decode.argmax,
                return_periodicity=True,
                device=self.device,
            )
            
            f0 = f0.squeeze(0).cpu().numpy()
            periodicity = periodicity.squeeze(0).cpu().numpy()
            voiced_mask = periodicity > 0.5
        else:
            f0, voiced_flag, _ = librosa.pyin(
                audio, sr=self.sample_rate, fmin=self.f0_fmin, fmax=self.f0_fmax,
                hop_length=self.hop_length, frame_length=self.hop_length * 4, fill_na=None
            )
            voiced_mask = ~np.isnan(f0)
            f0 = np.nan_to_num(f0, nan=100.0)
        
        # Interpolate unvoiced
        if np.any(voiced_mask) and np.sum(voiced_mask) >= 2:
            voiced_indices = np.where(voiced_mask)[0]
            f0 = np.interp(np.arange(len(f0)), voiced_indices, f0[voiced_mask])
        else:
            f0 = np.full_like(f0, 100.0)
        
        return f0, voiced_mask.astype(np.float32)
    
    def _extract_energy(self, audio: np.ndarray, n_frames: int) -> np.ndarray:
        """Extract frame-wise RMS energy."""
        energy = librosa.feature.rms(
            y=audio, hop_length=self.hop_length, frame_length=self.hop_length * 4
        )[0]
        
        # Match length
        if len(energy) > n_frames:
            energy = energy[:n_frames]
        elif len(energy) < n_frames:
            energy = np.pad(energy, (0, n_frames - len(energy)), mode='edge')
        
        return energy
    
    def _extract_rhythm(self, voicing: np.ndarray) -> np.ndarray:
        """Extract local voicing rate as rhythm."""
        window_size = 11
        return np.convolve(voicing, np.ones(window_size) / window_size, mode='same')


# ============================================
# Audio Loading (CPU) - Uses soundfile
# ============================================

def load_audio_fast(path: str, sr: int, max_samples: int) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
    """Load audio using soundfile (WAV/FLAC) with torchaudio fallback (MP3/OGG/M4A)."""
    try:
        ext = Path(path).suffix.lower()
        
        # soundfile handles WAV/FLAC natively (fastest)
        if ext in ('.wav', '.flac'):
            audio, orig_sr = sf.read(path, dtype='float32')
        else:
            # MP3/OGG/M4A: use torchaudio (requires ffmpeg backend)
            try:
                import torchaudio
                waveform, orig_sr = torchaudio.load(path)
                # torchaudio returns [channels, samples] as torch tensor
                audio = waveform.numpy()
                if audio.ndim > 1:
                    audio = audio.mean(axis=0)  # Mono
                audio = audio.astype(np.float32)
            except Exception as e_torch:
                # Last resort: try librosa (slowest but most compatible)
                try:
                    audio, orig_sr = librosa.load(path, sr=None, mono=True)
                    audio = audio.astype(np.float32)
                except Exception as e_librosa:
                    return path, None, f"All loaders failed: sf=N/A(ext={ext}), torchaudio={e_torch}, librosa={e_librosa}"
        
        # Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Resample if needed
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
        
        # Normalize
        peak = np.abs(audio).max()
        if peak > 1e-8:
            audio = audio / peak
        
        # Skip too-short audio (< 0.5s) — not enough frames for meaningful phonemes
        min_samples = sr // 2  # 0.5 seconds
        if len(audio) < min_samples:
            return path, None, f"Too short: {len(audio)/sr:.2f}s < 0.5s"
        
        # Trim
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        return path, audio.astype(np.float32), None
    except Exception as e:
        return path, None, str(e)


def load_batch_parallel(paths: List[str], sr: int, max_samples: int, workers: int):
    """Load batch in parallel."""
    results, errors = {}, []
    
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(load_audio_fast, p, sr, max_samples): p for p in paths}
        for f in as_completed(futures):
            path, audio, err = f.result()
            if audio is not None:
                results[path] = audio
            else:
                errors.append(f"{path}: {err}")
    
    audios = [results[p] for p in paths if p in results]
    valid_paths = [p for p in paths if p in results]
    return audios, valid_paths, errors


# ============================================
# Shard Writer - with resume support
# ============================================

class ShardWriter:
    def __init__(self, out_dir: Path, split: str, lang: str, per_shard: int, start_idx: int = 0):
        self.out_dir = out_dir
        self.split = split
        self.lang = lang
        self.per_shard = per_shard
        self.start_idx = start_idx  # resume: next shard number to write
        self.buffer = {
            'audio': [], 'phoneme_ids': [], 'phoneme_conf': [],
            'f0': [], 'energy': [], 'voicing': [], 'rhythm': [],
            'speaker_ids': [], 'paths': [], 'languages': [],
        }
        self.shard_idx = start_idx
        self.total = 0
    
    def add(self, **kw):
        # Map singular keys to plural buffer keys
        key_mapping = {
            'speaker_id': 'speaker_ids',
            'path': 'paths',
            'language': 'languages'
        }
        
        for k, v in kw.items():
            # Map singular to plural if needed
            buffer_key = key_mapping.get(k, k)
            if buffer_key not in self.buffer:
                raise KeyError(f"Unknown key: {k}. Buffer has keys: {list(self.buffer.keys())}")
            self.buffer[buffer_key].append(v)
        
        self.total += 1
        if len(self.buffer['audio']) >= self.per_shard:
            self._write()
    
    def _write(self):
        if not self.buffer['audio']:
            return
        
        if self.lang:
            if self.split in ['val', 'test']:
                fn = f"{self.split}_{self.lang}.npz"
            else:
                fn = f"{self.split}_{self.lang}_{self.shard_idx:03d}.npz"
        else:
            # No lang (timbre shards merge all languages)
            if self.split in ['val', 'test']:
                fn = f"{self.split}.npz"
            else:
                fn = f"{self.split}_{self.shard_idx:03d}.npz"
        
        np.savez_compressed(
            self.out_dir / fn,
            audio=np.array(self.buffer['audio'], dtype=object),
            phoneme_ids=np.array(self.buffer['phoneme_ids'], dtype=object),
            phoneme_conf=np.array(self.buffer['phoneme_conf'], dtype=object),
            f0=np.array(self.buffer['f0'], dtype=object),
            energy=np.array(self.buffer['energy'], dtype=object),
            voicing=np.array(self.buffer['voicing'], dtype=object),
            rhythm=np.array(self.buffer['rhythm'], dtype=object),
            speaker_ids=np.array(self.buffer['speaker_ids']),
            paths=np.array(self.buffer['paths']),
            languages=np.array(self.buffer['languages']),
        )
        
        log.info(f"  [SHARD] {fn} ({len(self.buffer['audio'])} samples)")
        self.buffer = {k: [] for k in self.buffer}
        self.shard_idx += 1
    
    def finalize(self) -> int:
        if self.buffer['audio']:
            self._write()
        return self.total


# ============================================
# Helper: Get processed count and next shard index for resume
# ============================================

def get_processed_info(shard_dir: Path, split: str, lang: str, samples_per_shard: int):
    """
    Scan existing shards for this split/lang.
    - Deletes any incomplete shards (samples < samples_per_shard).
    - Returns (processed_files, next_shard_idx).
    """
    if split in ['val', 'test']:
        # For val/test, we only have one shard file (no index). If it exists, we skip the split entirely.
        # This function is only called for train splits in resume mode.
        return 0, 0
    
    pattern = f"{split}_{lang}_*.npz"
    shard_files = sorted(shard_dir.glob(pattern))
    if not shard_files:
        return 0, 0
    
    complete_shards = []
    incomplete_shards = []
    for f in shard_files:
        try:
            with np.load(f, allow_pickle=True) as data:
                n = len(data['audio'])
                if n == samples_per_shard:
                    complete_shards.append(f)
                else:
                    incomplete_shards.append(f)
        except Exception:
            # Corrupt file - treat as incomplete
            incomplete_shards.append(f)
    
    # Delete incomplete shards (they are partial and will be reprocessed)
    for f in incomplete_shards:
        log.info(f"  [RESUME] Removing incomplete shard: {f.name}")
        f.unlink()
    
    processed_files = len(complete_shards) * samples_per_shard
    next_idx = len(complete_shards)
    return processed_files, next_idx


# ============================================
# Dataset Scanner
# ============================================

def scan_dataset(data_dir: Path) -> Dict:
    exts = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}
    dataset = {s: {l: [] for l in ['en', 'ar']} for s in ['train', 'val', 'test']}
    
    for split in dataset:
        for lang in dataset[split]:
            d = data_dir / split / lang
            if d.exists():
                for ext in exts:
                    dataset[split][lang].extend(str(f) for f in d.rglob(f'*{ext}'))
                dataset[split][lang].sort()
    
    return dataset


def get_speaker_id(path: str) -> str:
    p = Path(path)
    parent = p.parent.name
    
    # LibriTTS: parent dir is the speaker ID (e.g. /1234/book/file.wav)
    if parent not in ['en', 'ar', 'train', 'val', 'test']:
        return parent
    
    # LibriTTS flat structure: filename starts with speaker ID
    # e.g. 100_121669_000001_000000.wav → speaker "100"
    first_part = p.stem.split('_')[0]
    if first_part.isdigit():
        return first_part
    
    # CommonVoice: extract numeric clip ID and hash-bucket into ~200 groups
    # e.g. common_voice_ar_19058307.mp3 → extract 19058307 → bucket 107
    nums = re.findall(r'\d+', p.stem)
    if nums:
        num = max(nums, key=len)  # longest number is the clip ID
        bucket = int(num) % 200
        return f"ar_spk_{bucket:03d}"
    
    return f"ar_spk_{hash(p.stem) % 200:03d}"


# ============================================
# Main Processing
# ============================================

def process_split(
    paths: List[str], split: str, lang: str, cfg: Config,
    phoneme_ext: Optional[PhonemeExtractor], prosody_ext: ProsodyExtractor, out_dir: Path,
    skip_first_n: int = 0, start_shard_idx: int = 0,
) -> Tuple[int, int]:
    if not paths:
        return 0, 0
    
    # Skip already processed files
    if skip_first_n > 0:
        paths = paths[skip_first_n:]
        log.info(f"\n{'='*60}")
        log.info(f"{split}/{lang}: resuming after {skip_first_n} files, {len(paths)} remaining")
        log.info(f"{'='*60}")
    else:
        log.info(f"\n{'='*60}")
        log.info(f"{split}/{lang}: {len(paths):,} files")
        log.info(f"{'='*60}")
    
    max_samples = int(cfg.sample_rate * cfg.max_duration)
    writer = ShardWriter(out_dir, split, lang, cfg.samples_per_shard, start_shard_idx)
    all_errors = []
    
    pbar = tqdm(total=len(paths), desc=f"{split}/{lang}", unit="f")
    
    for i in range(0, len(paths), cfg.batch_size):
        batch_paths = paths[i:i + cfg.batch_size]
        
        # Load audio (CPU parallel)
        audios, valid_paths, errors = load_batch_parallel(batch_paths, cfg.sample_rate, max_samples, cfg.io_workers)
        all_errors.extend(errors)
        
        if not audios:
            pbar.update(len(batch_paths))
            continue
        
        # Extract phonemes (GPU - one by one)
        if phoneme_ext:
            ph_ids, ph_conf = phoneme_ext.extract_batch(audios, lang)
        else:
            ph_ids = [np.zeros(len(a) // 320 + 1, dtype=np.int64) for a in audios]
            ph_conf = [np.zeros(len(a) // 320 + 1, dtype=np.float32) for a in audios]
        
        # Extract prosody (GPU/CPU)
        prosody = prosody_ext.extract_batch(audios)
        
        # Write to shards
        for j, path in enumerate(valid_paths):
            writer.add(
                audio=audios[j],
                phoneme_ids=ph_ids[j],
                phoneme_conf=ph_conf[j],
                f0=prosody['f0'][j],
                energy=prosody['energy'][j],
                voicing=prosody['voicing'][j],
                rhythm=prosody['rhythm'][j],
                speaker_id=get_speaker_id(path),
                path=path,
                language=lang,
            )
        
        pbar.update(len(batch_paths))
    
    pbar.close()
    total = writer.finalize()
    
    if all_errors:
        log.warning(f"  {len(all_errors)} load errors")
    
    return total, len(all_errors)


# ============================================
# Timbre Shard Creation (with resume support)
# ============================================

def create_timbre_shards(shard_dir: Path, speakers_per_shard: int = 50):
    """
    Read existing shards and write speaker-grouped timbre shards.
    Resume support: skips speakers already written.
    """
    import random
    from collections import defaultdict
    
    log.info("\n" + "=" * 60)
    log.info("CREATING TIMBRE ENCODER SHARDS (speaker-grouped)")
    log.info("=" * 60)
    
    for split in ['train', 'val', 'test']:
        # Find regular shards for this split
        shard_files = sorted([
            f for f in shard_dir.glob("*.npz")
            if f.stem.split('_')[0] == split and not f.stem.startswith('timbre_')
        ])
        
        if not shard_files:
            continue
        
        log.info(f"\n--- {split.upper()} ({len(shard_files)} input shards) ---")
        
        # Collect all samples grouped by speaker
        speaker_data = defaultdict(lambda: {
            'audio': [], 'phoneme_ids': [], 'phoneme_conf': [],
            'f0': [], 'energy': [], 'voicing': [], 'rhythm': [],
            'paths': [], 'languages': [],
        })
        
        total_samples = 0
        for shard_path in shard_files:
            data = dict(np.load(shard_path, allow_pickle=True))
            n = len(data['audio'])
            total_samples += n
            
            for i in range(n):
                spk = str(data['speaker_ids'][i])
                speaker_data[spk]['audio'].append(data['audio'][i])
                speaker_data[spk]['phoneme_ids'].append(data['phoneme_ids'][i])
                speaker_data[spk]['phoneme_conf'].append(data['phoneme_conf'][i])
                speaker_data[spk]['f0'].append(data['f0'][i])
                speaker_data[spk]['energy'].append(data['energy'][i])
                speaker_data[spk]['voicing'].append(data['voicing'][i])
                speaker_data[spk]['rhythm'].append(data['rhythm'][i])
                speaker_data[spk]['paths'].append(str(data['paths'][i]))
                speaker_data[spk]['languages'].append(str(data['languages'][i]))
            
            del data
        
        speaker_data = dict(speaker_data)
        log.info(f"  {total_samples} samples, {len(speaker_data)} speakers")
        
        utt_counts = [len(v['audio']) for v in speaker_data.values()]
        log.info(f"  Utterances/speaker: min={min(utt_counts)}, max={max(utt_counts)}, "
                 f"avg={sum(utt_counts)/len(utt_counts):.1f}")
        
        # Check for existing timbre shards — skip speakers already written (resume support)
        existing_timbre = sorted([
            f for f in shard_dir.glob("*.npz")
            if f.stem.startswith(f'timbre_{split}')
        ])
        
        already_written_speakers = set()
        if existing_timbre:
            log.info(f"  Found {len(existing_timbre)} existing timbre shards, scanning for resume...")
            for ef in existing_timbre:
                try:
                    with np.load(ef, allow_pickle=True) as edata:
                        for sid in edata['speaker_ids']:
                            already_written_speakers.add(str(sid))
                except Exception:
                    # Corrupt/partial shard — delete it
                    log.warning(f"  Removing corrupt shard: {ef.name}")
                    ef.unlink()
            
            if already_written_speakers:
                log.info(f"  {len(already_written_speakers)} speakers already in timbre shards, skipping them")
        
        # Filter out already-written speakers
        speakers = [s for s in sorted(speaker_data.keys()) if s not in already_written_speakers]
        random.shuffle(speakers)
        
        # Continue shard numbering from where we left off
        shard_idx = len([f for f in shard_dir.glob(f"timbre_{split}_*.npz")])
        total_written = 0
        spk_idx = 0
        
        # For val/test: one shard
        target_spk = len(speakers) if split != 'train' else speakers_per_shard
        max_samples = total_samples + 1 if split != 'train' else 3000
        
        while spk_idx < len(speakers):
            buffer = {
                'audio': [], 'phoneme_ids': [], 'phoneme_conf': [],
                'f0': [], 'energy': [], 'voicing': [], 'rhythm': [],
                'speaker_ids': [], 'paths': [], 'languages': [],
            }
            shard_speakers = 0
            
            while spk_idx < len(speakers):
                spk = speakers[spk_idx]
                spk_n = len(speaker_data[spk]['audio'])
                
                if shard_speakers >= target_spk and buffer['audio']:
                    break
                if len(buffer['audio']) + spk_n > max_samples and buffer['audio']:
                    break
                
                buffer['audio'].extend(speaker_data[spk]['audio'])
                buffer['phoneme_ids'].extend(speaker_data[spk]['phoneme_ids'])
                buffer['phoneme_conf'].extend(speaker_data[spk]['phoneme_conf'])
                buffer['f0'].extend(speaker_data[spk]['f0'])
                buffer['energy'].extend(speaker_data[spk]['energy'])
                buffer['voicing'].extend(speaker_data[spk]['voicing'])
                buffer['rhythm'].extend(speaker_data[spk]['rhythm'])
                buffer['speaker_ids'].extend([spk] * spk_n)
                buffer['paths'].extend(speaker_data[spk]['paths'])
                buffer['languages'].extend(speaker_data[spk]['languages'])
                
                shard_speakers += 1
                spk_idx += 1
            
            if not buffer['audio']:
                break
            
            # Shuffle within shard
            n = len(buffer['audio'])
            indices = list(range(n))
            random.shuffle(indices)
            for key in buffer:
                buffer[key] = [buffer[key][i] for i in indices]
            
            # Write with timbre_ prefix
            if split in ['val', 'test']:
                fn = f"timbre_{split}.npz"
            else:
                fn = f"timbre_{split}_{shard_idx:03d}.npz"
            
            np.savez_compressed(
                shard_dir / fn,
                audio=np.array(buffer['audio'], dtype=object),
                phoneme_ids=np.array(buffer['phoneme_ids'], dtype=object),
                phoneme_conf=np.array(buffer['phoneme_conf'], dtype=object),
                f0=np.array(buffer['f0'], dtype=object),
                energy=np.array(buffer['energy'], dtype=object),
                voicing=np.array(buffer['voicing'], dtype=object),
                rhythm=np.array(buffer['rhythm'], dtype=object),
                speaker_ids=np.array(buffer['speaker_ids']),
                paths=np.array(buffer['paths']),
                languages=np.array(buffer['languages']),
            )
            
            log.info(f"  [TIMBRE SHARD] {fn}: {shard_speakers} speakers, {n} samples")
            total_written += n
            shard_idx += 1
        
        log.info(f"  → {shard_idx} timbre shards total, {total_written} new samples written")
        if already_written_speakers:
            log.info(f"  → {len(already_written_speakers)} speakers were already written (resumed)")
        del speaker_data
    
    log.info("\nTimbre shards complete.")



# ============================================
# Create Timbre Shards DIRECTLY from raw audio
# ============================================

def create_timbre_shards_from_raw(
    data_dir, out_dir, speakers_per_shard=50, max_shard_samples=3000,
    sample_rate=16000, max_duration=10.0, batch_size=64, workers=4, resume=False,
):
    """Create speaker-grouped timbre shards directly from raw audio. No normal shards needed."""
    import gc
    
    log.info("=" * 60)
    log.info("TIMBRE SHARDS — DIRECT FROM RAW AUDIO")
    log.info("=" * 60)
    
    max_samples = int(sample_rate * max_duration)
    ds = scan_dataset(data_dir)
    total = sum(len(ds[s][l]) for s in ds for l in ds[s])
    log.info(f"Total audio files: {total:,}")
    
    for split in ['train', 'val', 'test']:
        paths = []
        for lang in ['en', 'ar']:
            paths.extend(ds[split][lang])
        if not paths:
            continue
        
        log.info(f"\n--- {split.upper()} ({len(paths):,} files) ---")
        
        speaker_paths = defaultdict(list)
        for p in paths:
            speaker_paths[get_speaker_id(p)].append(p)
        
        speakers = sorted(speaker_paths.keys())
        random.shuffle(speakers)
        log.info(f"  {len(speakers)} speakers")
        
        groups = [speakers[i:i+speakers_per_shard] for i in range(0, len(speakers), speakers_per_shard)]
        
        shard_idx, start_group = 0, 0
        if resume:
            existing = sorted(out_dir.glob(f"timbre_{split}_*.npz"))
            if existing:
                done_spk = set()
                for ef in existing:
                    try:
                        with np.load(ef, allow_pickle=True) as d:
                            for s in d['speaker_ids']: done_spk.add(str(s))
                    except: ef.unlink()
                shard_idx = len(list(out_dir.glob(f"timbre_{split}_*.npz")))
                while start_group < len(groups) and all(s in done_spk for s in groups[start_group]):
                    start_group += 1
                log.info(f"  Resume: skip {start_group} groups")
        
        total_written, total_errors = 0, 0
        for gi in range(start_group, len(groups)):
            group = groups[gi]
            items = [(p, spk) for spk in group for p in speaker_paths[spk]]
            random.shuffle(items)
            
            audio_buf, sid_buf, path_buf, lang_buf = [], [], [], []
            for bi in range(0, len(items), batch_size):
                chunk = items[bi:bi+batch_size]
                p_only = [c[0] for c in chunk]
                spk_map = {c[0]: c[1] for c in chunk}
                # load_batch_parallel returns (audios_list, valid_paths_list, errors_list)
                audios, valid_paths, errs = load_batch_parallel(
                    p_only, sample_rate, max_samples, workers
                )
                total_errors += len(errs)
                for p, a in zip(valid_paths, audios):
                    audio_buf.append(a)
                    sid_buf.append(spk_map[p])
                    path_buf.append(p)
                    lang_buf.append('ar' if '/ar/' in p or '\\ar\\' in p else 'en')
            
            if audio_buf:
                for start in range(0, len(audio_buf), max_shard_samples):
                    end = min(start + max_shard_samples, len(audio_buf))
                    fn = f"timbre_{split}.npz" if split in ('val','test') and shard_idx == 0 else f"timbre_{split}_{shard_idx:03d}.npz"
                    n = end - start
                    idx = list(range(n)); random.shuffle(idx)
                    np.savez_compressed(out_dir / fn,
                        audio=np.array([audio_buf[start+i] for i in idx], dtype=object),
                        speaker_ids=np.array([sid_buf[start+i] for i in idx]),
                        paths=np.array([path_buf[start+i] for i in idx]),
                        languages=np.array([lang_buf[start+i] for i in idx]),
                    )
                    mb = (out_dir / fn).stat().st_size / 1e6
                    log.info(f"    {fn}: {n} utts, {len(set(sid_buf[start:end]))} spk, {mb:.0f}MB")
                    total_written += n; shard_idx += 1
            del audio_buf, sid_buf, path_buf, lang_buf; gc.collect()
        
        log.info(f"  -> {total_written:,} samples in {shard_idx} shards")
    log.info("\nTimbre shards (direct) complete.")


# ============================================
# Main
# ============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None,
                        help="Raw data directory (required unless --timbre_only)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples_per_shard", type=int, default=2000)
    parser.add_argument("--max_duration", type=float, default=10.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip_phonemes", action="store_true")
    parser.add_argument("--timbre_shards", action="store_true",
                        help="Also create speaker-grouped shards for Timbre Encoder (timbre_train_*.npz)")
    parser.add_argument("--timbre_speakers_per_shard", type=int, default=50,
                        help="Speakers per timbre shard (default: 50)")
    parser.add_argument("--timbre_only", action="store_true",
                        help="Only create timbre shards from existing regular shards (skip preprocessing)")
    parser.add_argument("--timbre_direct", action="store_true",
                        help="Create timbre shards directly from raw audio (no normal shards needed)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted preprocessing (skips already processed files)")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    
    # --timbre_only: skip all preprocessing, just reshard existing shards
    if args.timbre_only:
        log.info("\n[timbre_only] Creating timbre shards from existing shards...")
        create_timbre_shards(out_dir, speakers_per_shard=args.timbre_speakers_per_shard)
        return 0
    
    # --timbre_direct: create timbre shards directly from raw audio
    if args.timbre_direct:
        if args.data_dir is None:
            parser.error("--data_dir is required with --timbre_direct")
        log.info("\n[timbre_direct] Creating timbre shards directly from raw audio...")
        create_timbre_shards_from_raw(
            data_dir=Path(args.data_dir),
            out_dir=out_dir,
            speakers_per_shard=args.timbre_speakers_per_shard,
            max_shard_samples=args.samples_per_shard,
            sample_rate=16000,
            max_duration=args.max_duration,
            batch_size=args.batch_size,
            workers=args.workers,
            resume=args.resume,
        )
        return 0
    
    # Normal mode requires --data_dir
    if args.data_dir is None:
        parser.error("--data_dir is required (unless using --timbre_only)")
    
    data_dir = Path(args.data_dir)
    
    cfg = Config(
        device=args.device, batch_size=args.batch_size, io_workers=args.workers,
        samples_per_shard=args.samples_per_shard, max_duration=args.max_duration,
    )
    
    log.info("")
    log.info("=" * 60)
    log.info("UNIFIED SHARD PREPROCESSOR (WITH RESUME)")
    log.info("=" * 60)
    log.info(f"Batch: {cfg.batch_size} | Workers: {cfg.io_workers}")
    log.info(f"Skip phonemes: {args.skip_phonemes}")
    log.info(f"Resume: {args.resume}")
    log.info("=" * 60)
    
    # Scan
    log.info("\n[1/4] Scanning...")
    dataset = scan_dataset(data_dir)
    total = sum(len(dataset[s][l]) for s in dataset for l in dataset[s])
    log.info(f"  Total: {total:,} files")
    
    if total == 0:
        log.error("No files found!")
        return 1
    
    # Sanity check
    log.info("\nSanity checking audio loading...")
    sample_paths = []
    for s in dataset:
        for l in dataset[s]:
            sample_paths.extend(dataset[s][l][:5])
    
    ok_count = 0
    for p in sample_paths[:15]:
        _, audio, err = load_audio_fast(p, cfg.sample_rate, int(cfg.max_duration * cfg.sample_rate))
        if audio is not None:
            ok_count += 1
        else:
            log.warning(f"  Failed: {p}: {err}")
    
    if ok_count == 0:
        log.error("Sanity check failed - no audio could be loaded!")
        return 1
    log.info(f"Sanity check passed: {ok_count}/15 loaded")
    
    # Init extractors
    log.info("\n[2/4] Loading extractors...")
    phoneme_ext = None
    if not args.skip_phonemes:
        phoneme_ext = PhonemeExtractor(languages=["en", "ar"], device=cfg.device)
        phoneme_ext.load_all_models()
    
    prosody_ext = ProsodyExtractor(
        sample_rate=cfg.sample_rate, hop_length=cfg.hop_length, device=cfg.device
    )
    
    # Process
    log.info("\n[3/4] Processing...")
    start = time.time()
    stats = {'processed': 0, 'errors': 0}
    
    for split in ['train', 'val', 'test']:
        for lang in ['en', 'ar']:
            if not dataset[split][lang]:
                continue
            
            # For val/test, if resuming and the single shard already exists, skip
            if args.resume and split in ['val', 'test']:
                expected_shard = out_dir / f"{split}_{lang}.npz"
                if expected_shard.exists():
                    log.info(f"\n[SKIP] {split}/{lang} shard already exists, skipping.")
                    # Still count the files as processed? No, they are already processed.
                    stats['processed'] += len(dataset[split][lang])
                    continue
            
            # For train splits, get resume info
            skip_first_n = 0
            start_shard_idx = 0
            if args.resume and split == 'train':
                processed, start_shard_idx = get_processed_info(out_dir, split, lang, cfg.samples_per_shard)
                if processed > 0:
                    log.info(f"\n[RESUME] {split}/{lang}: {processed} files already processed, starting at shard {start_shard_idx}")
                skip_first_n = processed
            
            p, e = process_split(
                dataset[split][lang], split, lang, cfg, phoneme_ext, prosody_ext, out_dir,
                skip_first_n=skip_first_n, start_shard_idx=start_shard_idx
            )
            stats['processed'] += p
            stats['errors'] += e
    
    elapsed = time.time() - start
    
    # Metadata
    log.info("\n[4/4] Saving metadata...")
    metadata = {
        'format': 'unified_v2',
        'features': ['audio', 'phoneme_ids', 'phoneme_conf', 'f0', 'energy', 'voicing', 'rhythm'],
        'phoneme_vocab_size': phoneme_ext.num_phonemes if phoneme_ext else 0,
        'stats': stats,
        'elapsed_seconds': elapsed,
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    log.info("")
    log.info("=" * 60)
    log.info("DONE")
    log.info("=" * 60)
    log.info(f"Processed: {stats['processed']:,} | Errors: {stats['errors']}")
    log.info(f"Time: {elapsed/60:.1f} min ({stats['processed']/(elapsed+1):.1f} files/sec)")
    log.info("=" * 60)
    
    # Create timbre shards if requested
    if args.timbre_shards:
        create_timbre_shards(out_dir, speakers_per_shard=args.timbre_speakers_per_shard)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())