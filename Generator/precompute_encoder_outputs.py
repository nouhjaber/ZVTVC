#!/usr/bin/env python3
"""
Pre-compute ALL encoder outputs for Generator training.

Reads existing shards from /content/shards (audio + f0/energy/voicing/rhythm),
runs frozen Content Encoder, Prosody Encoder, Timbre Encoder on GPU,
and saves the results as new .npz shards in /content/outputs.

Output shard format:
    mel:      [N] array of float32 arrays, each [80, T]  (per-utterance normalized log-mel)
    content:  [N] array of float32 arrays, each [512, T]
    prosody:  [N] array of float32 arrays, each [32, T]
    timbre:   [N] array of float32 arrays, each [256]
    speaker_ids: [N] array of strings
    paths:       [N] array of strings

Usage (on Colab):
    python precompute_encoder_outputs.py \
        --shard_dir /content/shards \
        --output_dir /content/outputs \
        --content_encoder_ckpt /content/drive/MyDrive/ZVTVC/Content\ Encoder/checkpoints/stage_2_final.pt \
        --prosody_encoder_ckpt /content/drive/MyDrive/ZVTVC/Prosody\ Encoder/checkpoints/best.pt \
        --timbre_encoder_ckpt /content/drive/MyDrive/ZVTVC/Timbre\ Encoder/checkpoints/stage1_stage1_foundation.pt \
        --batch_size 32 \
        --num_workers 4
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import time
import json
import logging
import importlib
import importlib.util

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# ============================================
# Encoder Import Helper (same as Generator/training/dataset.py)
# ============================================

def _import_from_sibling(module_folder: str, import_path: str, class_name: str):
    """
    Import a class from a sibling module folder, avoiding collision with
    any existing 'model' package in sys.modules.
    """
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


# ============================================
# Encoder Loaders
# ============================================

def load_content_encoder(checkpoint_path: str, project_root: str, device: str = 'cuda'):
    content_encoder_dir = Path(project_root) / 'Content Encoder'
    if not content_encoder_dir.exists():
        content_encoder_dir = Path(project_root) / 'Content_Encoder'

    ContentEncoder = _import_from_sibling(
        module_folder=str(content_encoder_dir),
        import_path='model.content_encoder',
        class_name='ContentEncoder',
    )

    log.info(f"Loading Content Encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = ContentEncoder()

    if 'encoder' in ckpt:
        state_dict = ckpt['encoder']
        log.info(f"  Found 'encoder' key (stage {ckpt.get('stage', '?')}, iter {ckpt.get('iteration', '?')})")
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    # Remap keys if needed
    model_keys = set(encoder.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    if model_keys != ckpt_keys and 'preprocess_conv.conv.weight' in ckpt_keys:
        log.info("  Remapping old checkpoint keys...")
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
        log.warning(f"  [CE] {len(incompatible.missing_keys)} missing keys "
                    f"(first 5: {incompatible.missing_keys[:5]})")
    if incompatible.unexpected_keys:
        log.warning(f"  [CE] {len(incompatible.unexpected_keys)} unexpected keys "
                    f"(first 5: {incompatible.unexpected_keys[:5]})")
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log.info("  Content Encoder loaded.")
    return encoder


def load_prosody_encoder(checkpoint_path: str, project_root: str, device: str = 'cuda'):
    prosody_encoder_dir = Path(project_root) / 'Prosody Encoder'
    if not prosody_encoder_dir.exists():
        prosody_encoder_dir = Path(project_root) / 'Prosody_Encoder'

    ProsodyEncoder = _import_from_sibling(
        module_folder=str(prosody_encoder_dir),
        import_path='model.prosody_encoder',
        class_name='ProsodyEncoder',
    )

    log.info(f"Loading Prosody Encoder from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # use_reconstruction_heads=True matches the training default in Prosody_Encoder/train.py
    # (ProsodyEncoder is constructed with only refined_dim=32; all other args take defaults,
    #  and use_reconstruction_heads defaults to True in prosody_encoder.py:41).
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
        log.warning(f"  [PE] {len(incompatible.missing_keys)} missing keys "
                    f"(first 5: {incompatible.missing_keys[:5]})")
    if incompatible.unexpected_keys:
        log.warning(f"  [PE] {len(incompatible.unexpected_keys)} unexpected keys "
                    f"(first 5: {incompatible.unexpected_keys[:5]})")
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log.info("  Prosody Encoder loaded.")
    return encoder


def load_timbre_encoder(checkpoint_path: str, project_root: str, device: str = 'cuda'):
    timbre_encoder_dir = Path(project_root) / 'Timbre Encoder'
    if not timbre_encoder_dir.exists():
        timbre_encoder_dir = Path(project_root) / 'Timbre_Encoder'

    # Silence ECAPA debug spam
    for logger_name in ['model.ecapa_tdnn', 'model.ecapa_tdnn.pooling']:
        l = logging.getLogger(logger_name)
        l.propagate = False
        l.handlers.clear()
        l.setLevel(logging.WARNING)
        l.addHandler(logging.NullHandler())

    ECAPATDNN = _import_from_sibling(
        module_folder=str(timbre_encoder_dir),
        import_path='model.ecapa_tdnn',
        class_name='ECAPATDNN',
    )

    log.info(f"Loading Timbre Encoder from {checkpoint_path}...")
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

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    log.info("  Timbre Encoder loaded.")
    return encoder


# ============================================
# CPU Dataset for DataLoader workers
# ============================================

class ShardSampleDataset(Dataset):
    """
    Reads samples from a single shard file.
    Returns raw numpy arrays — no GPU ops, safe for num_workers > 0.
    Mel is computed on CPU here.
    """

    def __init__(self, shard_path: str, sample_rate: int = 16000):
        self.data = dict(np.load(shard_path, allow_pickle=True))
        self.n = len(self.data['audio'])
        self.sample_rate = sample_rate

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=320,
            win_length=1024, n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
        )

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        audio = np.asarray(self.data['audio'][idx], dtype=np.float32)
        f0 = np.asarray(self.data['f0'][idx], dtype=np.float32)
        energy = np.asarray(self.data['energy'][idx], dtype=np.float32)
        voicing = np.asarray(self.data['voicing'][idx], dtype=np.float32)
        rhythm = np.asarray(self.data['rhythm'][idx], dtype=np.float32)

        speaker_id = str(self.data['speaker_ids'][idx])
        path = str(self.data['paths'][idx])

        # Compute mel on CPU — three variants for three different consumers
        waveform = torch.from_numpy(audio)
        mel_raw = self.mel_transform(waveform)  # [80, T] linear power mel

        # Generator target: raw log-mel, no normalization.
        # The Generator learns to predict this directly (vocoder-ready scale).
        mel_target = torch.log(mel_raw + 1e-6)

        # Content Encoder input: log(clamp), no per-utterance normalization.
        # Matches exactly what Content_Encoder/train.py:adapt_batch_for_content_encoder uses.
        mel_for_ce = torch.log(mel_raw.clamp(min=1e-5))

        # Timbre Encoder input: log + per-utterance mean/std normalization.
        # Matches Shard_dataset_unified.py:TimbreEncoderDataset (what the Timbre Encoder was trained on).
        mel_for_timbre = torch.log(mel_raw + 1e-6)
        mel_for_timbre = (mel_for_timbre - mel_for_timbre.mean()) / (mel_for_timbre.std() + 1e-6)

        return {
            'mel_target': mel_target,        # [80, T] raw log-mel  → stored as Generator target
            'mel_for_ce': mel_for_ce,        # [80, T] log(clamp)   → Content Encoder input
            'mel_for_timbre': mel_for_timbre, # [80, T] normalized   → Timbre Encoder input
            'f0': torch.from_numpy(f0),
            'energy': torch.from_numpy(energy),
            'voicing': torch.from_numpy(voicing),
            'rhythm': torch.from_numpy(rhythm),
            'speaker_id': speaker_id,
            'path': path,
        }


def collate_variable_length(batch):
    """
    Collate variable-length samples WITHOUT padding.
    Returns lists of tensors (not stacked), since each sample has different T.
    """
    return {
        'mel_target': [b['mel_target'] for b in batch],      # raw log-mel (Generator target)
        'mel_for_ce': [b['mel_for_ce'] for b in batch],      # log(clamp) for Content Encoder
        'mel_for_timbre': [b['mel_for_timbre'] for b in batch],  # normalized for Timbre Encoder
        'f0': [b['f0'] for b in batch],
        'energy': [b['energy'] for b in batch],
        'voicing': [b['voicing'] for b in batch],
        'rhythm': [b['rhythm'] for b in batch],
        'speaker_id': [b['speaker_id'] for b in batch],
        'path': [b['path'] for b in batch],
    }


# ============================================
# Main Processing
# ============================================

def align_temporal(x: torch.Tensor, target_T: int) -> torch.Tensor:
    """Pad or truncate temporal dim to match target_T."""
    T = x.shape[-1]
    if T == target_T:
        return x
    elif T < target_T:
        return torch.nn.functional.pad(x, (0, target_T - T))
    else:
        return x[..., :target_T]


@torch.no_grad()
def process_batch_gpu(
    batch: dict,
    content_encoder: nn.Module,
    prosody_encoder: nn.Module,
    timbre_encoder: nn.Module,
    device: str,
    streams: dict = None,
) -> List[dict]:
    """
    Run all 3 encoders in PARALLEL using CUDA streams.
    Each encoder runs on its own stream — truly concurrent GPU execution.
    """
    mels_target = batch['mel_target']     # list of [80, T_i] — raw log-mel, stored as target
    mels_ce = batch['mel_for_ce']         # list of [80, T_i] — log(clamp), for Content Encoder
    mels_timbre = batch['mel_for_timbre'] # list of [80, T_i] — normalized, for Timbre Encoder
    f0s = batch['f0']         # list of [T_i]
    energies = batch['energy']
    voicings = batch['voicing']
    rhythms = batch['rhythm']

    B = len(mels_ce)
    lengths = [m.shape[1] for m in mels_ce]  # all three mels have the same T per sample
    max_T = max(lengths)

    # --- Pad and stack CE mel: [B, 80, max_T] (log clamp, no norm) ---
    mel_ce_padded = torch.zeros(B, 80, max_T)
    for i, m in enumerate(mels_ce):
        mel_ce_padded[i, :, :lengths[i]] = m
    mel_ce_gpu = mel_ce_padded.to(device, non_blocking=True)

    # --- Pad and stack timbre mel: [B, 80, max_T] (log + per-utterance norm) ---
    mel_timbre_padded = torch.zeros(B, 80, max_T)
    for i, m in enumerate(mels_timbre):
        mel_timbre_padded[i, :, :m.shape[1]] = m
    mel_timbre_gpu = mel_timbre_padded.to(device, non_blocking=True)

    # --- Pad and stack prosody features: [B, 4, max_T_prosody] ---
    prosody_lengths = [f.shape[0] for f in f0s]
    max_T_p = max(prosody_lengths)
    prosody_padded = torch.zeros(B, 4, max_T_p)
    for i in range(B):
        L = prosody_lengths[i]
        prosody_padded[i, 0, :L] = f0s[i]
        prosody_padded[i, 1, :L] = energies[i]
        prosody_padded[i, 2, :L] = voicings[i]
        prosody_padded[i, 3, :L] = rhythms[i]
    prosody_gpu = prosody_padded.to(device, non_blocking=True)

    # --- Create CUDA streams if not provided ---
    if streams is None:
        streams = {
            'content': torch.cuda.Stream(device=device),
            'prosody': torch.cuda.Stream(device=device),
            'timbre': torch.cuda.Stream(device=device),
        }

    # --- Launch all 3 encoders on SEPARATE streams (parallel) ---
    # Default stream must finish the H2D transfers first
    torch.cuda.current_stream(device).synchronize()

    # Content encoder on stream 1 — receives log(clamp) mel, no normalization
    with torch.cuda.stream(streams['content']):
        content_out = content_encoder(mel_ce_gpu)
        if isinstance(content_out, tuple):
            content_out = content_out[0]
        content_all = content_out.cpu()

    # Prosody encoder on stream 2 — receives explicit prosody features (unchanged)
    with torch.cuda.stream(streams['prosody']):
        prosody_out = prosody_encoder(explicit_features=prosody_gpu)
        if isinstance(prosody_out, tuple):
            prosody_out = prosody_out[0]
        prosody_all = prosody_out.cpu()

    # Timbre encoder on stream 3 — receives per-utterance normalized mel
    with torch.cuda.stream(streams['timbre']):
        timbre_out = timbre_encoder(mel_timbre_gpu)
        if isinstance(timbre_out, tuple):
            timbre_out = timbre_out[0]
        timbre_all = timbre_out.cpu()

    # --- Wait for ALL streams to finish ---
    for s in streams.values():
        s.synchronize()

    # --- Trim back to original lengths ---
    results = []
    for i in range(B):
        T = lengths[i]
        content_i = align_temporal(content_all[i], T)
        prosody_i = align_temporal(prosody_all[i], T)
        timbre_i = timbre_all[i]

        results.append({
            'mel': mels_target[i].numpy(),   # raw log-mel — Generator's training target
            'content': content_i.numpy(),
            'prosody': prosody_i.numpy(),
            'timbre': timbre_i.numpy(),
            'speaker_id': batch['speaker_id'][i],
            'path': batch['path'][i],
        })

    return results


def process_shard(
    shard_path: Path,
    output_dir: Path,
    content_encoder: nn.Module,
    prosody_encoder: nn.Module,
    timbre_encoder: nn.Module,
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
    streams: dict = None,
):
    """Process a single shard file and save output."""

    dataset = ShardSampleDataset(str(shard_path))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_variable_length,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    all_results = []
    for batch in loader:
        results = process_batch_gpu(
            batch, content_encoder, prosody_encoder, timbre_encoder, device,
            streams=streams,
        )
        all_results.extend(results)

    # Save output shard
    # NOTE: np.array([...], dtype=object) fails when sub-arrays share a leading
    # dimension (e.g. all [80, T_i]) — numpy tries to stack instead of making
    # an object array.  Pre-allocate an empty object array and fill it.
    n = len(all_results)

    def _make_object_array(items):
        arr = np.empty(len(items), dtype=object)
        for i, v in enumerate(items):
            arr[i] = v
        return arr

    out_path = output_dir / shard_path.name
    np.savez_compressed(
        out_path,
        mel=_make_object_array([r['mel'] for r in all_results]),
        content=_make_object_array([r['content'] for r in all_results]),
        prosody=_make_object_array([r['prosody'] for r in all_results]),
        timbre=_make_object_array([r['timbre'] for r in all_results]),
        speaker_ids=np.array([r['speaker_id'] for r in all_results]),
        paths=np.array([r['path'] for r in all_results]),
    )

    return len(all_results)


def main():
    parser = argparse.ArgumentParser(description='Pre-compute encoder outputs for Generator training')
    parser.add_argument('--shard_dir', type=str, required=True,
                        help='Path to input shards (e.g. /content/shards)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output dir (e.g. /content/outputs)')
    parser.add_argument('--project_root', type=str, default=None,
                        help='ZVTVC project root (default: auto-detect from shard_dir)')

    parser.add_argument('--content_encoder_ckpt', type=str, required=True)
    parser.add_argument('--prosody_encoder_ckpt', type=str, required=True)
    parser.add_argument('--timbre_encoder_ckpt', type=str, required=True)

    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size per encoder forward pass (default: 256, uses ~4GB VRAM)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers for CPU I/O and mel (default: 8)')
    parser.add_argument('--parallel_shards', type=int, default=3,
                        help='Number of shards to process concurrently (default: 3)')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    shard_dir = Path(args.shard_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect project root
    if args.project_root is None:
        # Try common Colab layouts
        for candidate in [
            Path('/content/drive/MyDrive/ZVTVC'),
            shard_dir.parent,
            Path.cwd(),
        ]:
            if (candidate / 'Content Encoder').exists() or (candidate / 'Content_Encoder').exists():
                args.project_root = str(candidate)
                break
        if args.project_root is None:
            log.error("Cannot auto-detect project root. Pass --project_root explicitly.")
            sys.exit(1)

    log.info(f"Project root: {args.project_root}")
    log.info(f"Input shards: {shard_dir}")
    log.info(f"Output dir:   {output_dir}")

    # Load encoders
    content_enc = load_content_encoder(args.content_encoder_ckpt, args.project_root, args.device)
    prosody_enc = load_prosody_encoder(args.prosody_encoder_ckpt, args.project_root, args.device)
    timbre_enc = load_timbre_encoder(args.timbre_encoder_ckpt, args.project_root, args.device)

    # Find all shards
    shard_files = sorted(shard_dir.glob("*.npz"))
    if not shard_files:
        log.error(f"No .npz files found in {shard_dir}")
        sys.exit(1)

    log.info(f"Found {len(shard_files)} shard files")

    # Create CUDA streams — each encoder gets its own for parallel execution
    cuda_streams = {
        'content': torch.cuda.Stream(device=args.device),
        'prosody': torch.cuda.Stream(device=args.device),
        'timbre': torch.cuda.Stream(device=args.device),
    }
    log.info(f"3 CUDA streams created | batch_size={args.batch_size} | "
             f"num_workers={args.num_workers} | parallel_shards={args.parallel_shards}")

    # Count already-done shards
    total_samples = 0
    pending_shards = []
    for shard_path in shard_files:
        out_path = output_dir / shard_path.name
        if out_path.exists():
            try:
                with np.load(out_path, allow_pickle=True) as data:
                    total_samples += len(data['mel'])
                log.info(f"  SKIP {shard_path.name} (already exists)")
            except Exception:
                # Corrupted — redo it
                out_path.unlink()
                pending_shards.append(shard_path)
        else:
            pending_shards.append(shard_path)

    log.info(f"Skipped {len(shard_files) - len(pending_shards)} already-done shards "
             f"({total_samples} samples)")
    log.info(f"Remaining: {len(pending_shards)} shards to process")

    if not pending_shards:
        log.info("All shards already processed!")
        return

    # --- Process remaining shards with thread pool ---
    # Why threads (not processes): GPU ops go through CUDA driver (thread-safe),
    # disk I/O releases the GIL, and we share encoder weights in-process.
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    gpu_lock = threading.Lock()  # serialize GPU forward passes
    start_time = time.time()
    completed = 0

    def _process_one_shard(shard_path):
        """Load shard → compute on GPU → save. Thread-safe."""
        out_path = output_dir / shard_path.name
        if out_path.exists():
            return shard_path.name, -1  # already done

        # CPU: load shard and compute mels via DataLoader
        dataset = ShardSampleDataset(str(shard_path))
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 4),  # per-shard workers
            collate_fn=collate_variable_length,
            pin_memory=True,
            persistent_workers=False,
            prefetch_factor=4 if args.num_workers > 0 else None,
        )

        all_results = []
        for batch in loader:
            # Serialize GPU access across threads
            with gpu_lock:
                results = process_batch_gpu(
                    batch, content_enc, prosody_enc, timbre_enc,
                    args.device, streams=cuda_streams,
                )
            all_results.extend(results)

        # CPU: save (releases GIL during I/O)
        n = len(all_results)

        def _make_object_array(items):
            arr = np.empty(len(items), dtype=object)
            for i, v in enumerate(items):
                arr[i] = v
            return arr

        np.savez_compressed(
            out_path,
            mel=_make_object_array([r['mel'] for r in all_results]),
            content=_make_object_array([r['content'] for r in all_results]),
            prosody=_make_object_array([r['prosody'] for r in all_results]),
            timbre=_make_object_array([r['timbre'] for r in all_results]),
            speaker_ids=np.array([r['speaker_id'] for r in all_results]),
            paths=np.array([r['path'] for r in all_results]),
        )

        return shard_path.name, n

    # Launch parallel shard processing
    # With parallel_shards=3: while shard A is on GPU, shard B loads from disk,
    # shard C saves to disk. Full overlap of I/O and compute.
    with ThreadPoolExecutor(max_workers=args.parallel_shards) as executor:
        futures = {
            executor.submit(_process_one_shard, sp): sp
            for sp in pending_shards
        }

        for future in as_completed(futures):
            shard_path = futures[future]
            try:
                name, n = future.result()
                if n > 0:
                    completed += 1
                    total_samples += n
                    elapsed = time.time() - start_time
                    speed = total_samples / elapsed if elapsed > 0 else 0
                    remaining = len(pending_shards) - completed
                    eta_min = (remaining * elapsed / max(completed, 1)) / 60

                    log.info(
                        f"[{completed}/{len(pending_shards)}] {name}: "
                        f"{n} samples | "
                        f"Total: {total_samples} | Speed: {speed:.0f} samp/s | "
                        f"ETA: {eta_min:.1f}min"
                    )
            except Exception as e:
                log.error(f"FAILED {shard_path.name}: {e}")

            # Clear GPU cache periodically
            if completed % 10 == 0:
                torch.cuda.empty_cache()

    total_time = time.time() - start_time
    log.info(f"\nDone! {total_samples} samples in {total_time/60:.1f} min")
    log.info(f"Output saved to: {output_dir}")

    # Save metadata
    meta = {
        'total_samples': total_samples,
        'num_shards': len(shard_files),
        'format': 'precomputed_encoder_outputs',
        'keys': ['mel', 'content', 'prosody', 'timbre', 'speaker_ids', 'paths'],
        'shapes': {
            'mel': '[80, T] per-utterance normalized log-mel',
            'content': '[512, T] Content Encoder output',
            'prosody': '[32, T] Prosody Encoder output',
            'timbre': '[256] Timbre Encoder output (L2-normalized)',
        },
        'encoders': {
            'content': args.content_encoder_ckpt,
            'prosody': args.prosody_encoder_ckpt,
            'timbre': args.timbre_encoder_ckpt,
        },
    }
    with open(output_dir / 'precompute_metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    main()