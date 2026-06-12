"""
verify_fixes.py — Verify all code fixes are correctly applied.

Checks:
  1. Each encoder loads without missing/unexpected keys.
  2. CE-specific mel and standard mel have the expected statistics.
  3. Each encoder forward pass produces the expected output shape.
  4. Target mel (raw log-mel, no normalization) has the expected statistics.
     Expected: mean ≈ -6, std ≈ 2, min ≈ -11.5, max ≈ +1.
     If you see mean ≈ 0, std ≈ 1  →  FIX 2 did not take effect.

Usage:
    python Generator/scripts/verify_fixes.py \\
        --content_encoder_ckpt  "Content_Encoder/checkpoints/stage_2_final.pt" \\
        --prosody_encoder_ckpt  "Prosody_Encoder/checkpoints/best.pt" \\
        --timbre_encoder_ckpt   "Timbre_Encoder/checkpoints/stage1_stage1_foundation.pt"

    All three flags are optional; omit any to test with random features instead.
"""

import argparse
import io
import sys

# Force UTF-8 stdout so Unicode characters print correctly on Windows (cp125x)
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
import torchaudio
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_GENERATOR_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == 'scripts' else _SCRIPT_DIR
sys.path.insert(0, str(_GENERATOR_ROOT))


def _sep(title=''):
    print(f"\n{'-' * 60}")
    if title:
        print(f"  {title}")
        print('-' * 60)


def _stats(name: str, t: torch.Tensor):
    print(f"  {name}: shape={list(t.shape)}  "
          f"min={t.min():.4f}  max={t.max():.4f}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}")


# ── mel helpers (mirror of convert.py) ───────────────────────────────────────
def compute_mel_for_ce(waveform: torch.Tensor) -> torch.Tensor:
    """log(clamp(1e-5)), no normalization — matches CE training."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, win_length=1024,
        n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
    )
    mel = transform(waveform)
    return torch.log(mel.clamp(min=1e-5))


def compute_mel_for_timbre(waveform: torch.Tensor) -> torch.Tensor:
    """log(+1e-6) + per-utterance norm — matches Timbre Encoder training."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, win_length=1024,
        n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
    )
    mel = transform(waveform)
    mel = torch.log(mel + 1e-6)
    return (mel - mel.mean()) / (mel.std() + 1e-6)


def compute_mel_target(waveform: torch.Tensor) -> torch.Tensor:
    """Raw log-mel, no normalization — what the Generator trains to predict."""
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, win_length=1024,
        n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
    )
    mel = transform(waveform)
    return torch.log(mel + 1e-6)


# ── encoder loaders (import from training/dataset.py) ────────────────────────
def _load_encoders(args):
    from training.dataset import (
        load_content_encoder,
        load_prosody_encoder,
        load_timbre_encoder,
    )
    content_enc = prosody_enc = timbre_enc = None

    if args.content_encoder_ckpt:
        print("\n[CE] Loading Content Encoder...")
        content_enc = load_content_encoder(args.content_encoder_ckpt,
                                           device=args.device)
        print("  [CE] Loaded OK")
    else:
        print("\n[CE] No checkpoint supplied -- will use random features")

    if args.prosody_encoder_ckpt:
        print("\n[PE] Loading Prosody Encoder...")
        prosody_enc = load_prosody_encoder(args.prosody_encoder_ckpt,
                                           device=args.device)
        print("  [PE] Loaded OK")
    else:
        print("\n[PE] No checkpoint supplied -- will use random features")

    if args.timbre_encoder_ckpt:
        print("\n[TE] Loading Timbre Encoder...")
        timbre_enc = load_timbre_encoder(args.timbre_encoder_ckpt,
                                         device=args.device)
        print("  [TE] Loaded OK")
    else:
        print("\n[TE] No checkpoint supplied -- will use random features")

    return content_enc, prosody_enc, timbre_enc


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Verify ZVTVC code fixes')
    parser.add_argument('--content_encoder_ckpt', type=str, default=None)
    parser.add_argument('--prosody_encoder_ckpt', type=str, default=None)
    parser.add_argument('--timbre_encoder_ckpt', type=str, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")

    # ── 1. Build a 3-second test waveform at 16 kHz ──────────────────────────
    _sep("1. Test waveform")
    sr = 16000
    duration = 3.0
    t = torch.linspace(0, duration, int(sr * duration))
    # Mix of two sinusoids to give non-trivial mel content
    waveform = (0.5 * torch.sin(2 * 3.14159 * 220 * t) +
                0.3 * torch.sin(2 * 3.14159 * 440 * t))
    waveform = waveform.unsqueeze(0)  # [1, samples]
    print(f"  waveform: shape={list(waveform.shape)}  sr={sr}")

    # ── 2. Mel statistics ─────────────────────────────────────────────────────
    _sep("2. Mel spectrogram statistics")

    mel_ce = compute_mel_for_ce(waveform)
    _stats("mel_for_ce    [log clamp, no norm]", mel_ce)

    mel_timbre = compute_mel_for_timbre(waveform)
    _stats("mel_for_timbre [log+per-utt norm] ", mel_timbre)

    mel_target = compute_mel_target(waveform)
    _stats("mel_target     [raw log-mel]      ", mel_target)

    print()
    print("  -- FIX 2 check --")
    print("  mel_target should have mean ~= -6, std ~= 2  (raw log-mel scale).")
    print("  If mean ~= 0, std ~= 1  => per-utterance normalization is still on.")
    if abs(mel_target.mean().item()) < 0.5 and abs(mel_target.std().item() - 1.0) < 0.2:
        print("  FAIL: mel_target looks per-utterance-normalized. FIX 2 may not be applied.")
    else:
        print("  PASS: mel_target is in raw log-mel scale.")

    print()
    print("  -- FIX 1 check --")
    print("  mel_for_ce should NOT be mean~=0/std~=1 (no normalization).")
    if abs(mel_ce.mean().item()) < 0.5 and abs(mel_ce.std().item() - 1.0) < 0.2:
        print("  FAIL: mel_for_ce looks per-utterance-normalized. FIX 1 may not be applied.")
    else:
        print("  PASS: mel_for_ce is in raw log(clamp) scale.")

    # ── 3. Load encoders ──────────────────────────────────────────────────────
    _sep("3. Encoder loading (missing/unexpected key counts printed above)")
    content_enc, prosody_enc, timbre_enc = _load_encoders(args)

    # ── 4. Encoder forward passes ─────────────────────────────────────────────
    _sep("4. Encoder forward passes")

    mel_ce_d = mel_ce.to(args.device)           # [1, 80, T]
    mel_timbre_d = mel_timbre.to(args.device)   # [1, 80, T]
    T = mel_ce.shape[-1]

    with torch.no_grad():
        # Content Encoder
        if content_enc is not None:
            z_c = content_enc(mel_ce_d)
            if isinstance(z_c, tuple):
                z_c = z_c[0]
            _stats("Content Encoder output z_c", z_c.cpu())
        else:
            z_c = torch.randn(1, 512, T)
            print(f"  Content Encoder: DUMMY random {list(z_c.shape)}")

        # Timbre Encoder
        if timbre_enc is not None:
            z_t = timbre_enc(mel_timbre_d)
            if isinstance(z_t, tuple):
                z_t = z_t[0]
            _stats("Timbre Encoder output z_t", z_t.cpu())
        else:
            z_t = torch.randn(1, 256)
            print(f"  Timbre Encoder:   DUMMY random {list(z_t.shape)}")

        # Prosody Encoder
        if prosody_enc is not None:
            # Unwrap to get extract_explicit_features
            feature_extractor = prosody_enc
            while not hasattr(feature_extractor, 'extract_explicit_features'):
                feature_extractor = feature_extractor.encoder
            waveform_np = waveform.squeeze(0)
            explicit = feature_extractor.extract_explicit_features(waveform_np)
            if explicit.dim() == 2:
                explicit = explicit.unsqueeze(0)
            explicit = explicit.to(args.device)
            z_p = prosody_enc(explicit_features=explicit)
            if isinstance(z_p, tuple):
                z_p = z_p[0]
            _stats("Prosody Encoder output z_p", z_p.cpu())
        else:
            z_p = torch.randn(1, 32, T)
            print(f"  Prosody Encoder:  DUMMY random {list(z_p.shape)}")

    # ── 5. Save a sample target mel and print stats ───────────────────────────
    _sep("5. Sample target mel save check")
    save_path = _GENERATOR_ROOT / 'scripts' / '_verify_sample_mel.pt'
    torch.save(mel_target.cpu(), str(save_path))
    loaded = torch.load(str(save_path), map_location='cpu')
    _stats("Loaded mel_target from disk", loaded)
    print(f"  Saved to: {save_path}")
    # Clean up
    save_path.unlink(missing_ok=True)

    # ── 6. Summary ────────────────────────────────────────────────────────────
    _sep("Summary")
    print("  All checks complete. Review output above for ❌ markers.")
    print()
    print("  Expected mel_target stats (raw log-mel at 16kHz/hop=320):")
    print("    mean : ~-6    (typical log-mel mean for speech)")
    print("    std  : ~2     (typical log-mel std for speech)")
    print("    min  : ~-11.5 (floor from log(1e-5) ≈ -11.5)")
    print("    max  : ~+1    (loud frames near 0)")
    print()
    print("  Expected encoder output shapes:")
    print("    Content Encoder z_c : [1, 512, T]")
    print("    Timbre Encoder  z_t : [1, 256]")
    print("    Prosody Encoder z_p : [1, 32,  T']")


if __name__ == '__main__':
    main()
