"""
Voice Conversion Script for Generator (Module 4)
End-to-end: source audio + reference audio -> converted audio.

Usage:
    # With real encoders + HiFi-GAN vocoder
    python scripts/convert.py ^
        --checkpoint Generator/checkpoints/best_model.pt ^
        --source nouh.ogg ^
        --reference huthaifa.ogg ^
        --output converted.wav ^
        --content_encoder_ckpt "Content Encoder/checkpoints/stage_2_final.pt" ^
        --prosody_encoder_ckpt "Prosody Encoder/checkpoints/best.pt" ^
        --timbre_encoder_ckpt "Timbre Encoder/checkpoints/stage1_stage1_foundation.pt" ^
        --vocoder_ckpt C:/Users/Nouh/Desktop/lap/hifi-gan/UNIVERSAL_V1/g_02500000 ^
        --vocoder_config C:/Users/Nouh/Desktop/lap/hifi-gan/UNIVERSAL_V1/config.json

    # Without vocoder (saves mel .pt file)
    python scripts/convert.py ^
        --checkpoint Generator/checkpoints/best_model.pt ^
        --source nouh.ogg ^
        --reference huthaifa.ogg ^
        --output converted.wav
"""

import torch
import torchaudio
import argparse
import logging
import traceback
import json
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_GENERATOR_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == 'scripts' else _SCRIPT_DIR
sys.path.insert(0, str(_GENERATOR_ROOT))

from model.unet import FlowMatchingUNet
from inference import MelGenerator


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    return logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """
    Load audio in any format (wav, ogg, mp3, flac, etc).
    Converts to mono, resamples to target_sr. Returns [1, samples].
    """
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = None, None

    try:
        waveform, sr = torchaudio.load(path)
    except Exception:
        pass

    if waveform is None:
        import subprocess, io
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', path, '-f', 'wav', '-acodec', 'pcm_s16le',
                 '-ar', str(target_sr), '-ac', '1', 'pipe:1'],
                capture_output=True, timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
            buf = io.BytesIO(result.stdout)
            waveform, sr = torchaudio.load(buf)
        except FileNotFoundError:
            raise RuntimeError(
                f"Cannot load '{path}': torchaudio failed and ffmpeg is not installed."
            )

    if waveform is None:
        raise RuntimeError(f"Failed to load audio: {path}")

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.transforms.Resample(sr, target_sr)(waveform)

    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {path}")
    if torch.isnan(waveform).any():
        raise ValueError(f"Audio file contains NaN: {path}")

    return waveform


def compute_mel_for_ce(waveform: torch.Tensor) -> torch.Tensor:
    """
    Mel for the Content Encoder.
    Matches Content_Encoder/train.py:adapt_batch_for_content_encoder exactly:
      log(clamp(mel, min=1e-5)), NO per-utterance normalization.
    sr=16000, n_fft=1024, hop=320, win=1024, n_mels=80.
    Returns [1, 80, T].
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, win_length=1024,
        n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
    )
    mel = mel_transform(waveform)
    mel = torch.log(mel.clamp(min=1e-5))
    return mel


def compute_mel_16k(waveform: torch.Tensor) -> torch.Tensor:
    """
    Mel for the Timbre Encoder (per-utterance normalized).
    Matches Shard_dataset_unified.py:TimbreEncoderDataset (Timbre Encoder training):
      log(mel + 1e-6) + per-utterance mean/std normalization.
    sr=16000, n_fft=1024, hop=320, win=1024, n_mels=80.
    Returns [1, 80, T].
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=320, win_length=1024,
        n_mels=80, f_min=0.0, f_max=8000.0, power=2.0,
    )
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-6)
    mel = (mel - mel.mean()) / (mel.std() + 1e-6)
    return mel


# ---------------------------------------------------------------------------
# HiFi-GAN loader
# ---------------------------------------------------------------------------
def load_hifigan(checkpoint_path: str, config_path: str, device: str):
    """
    Load HiFi-GAN Generator from jik876/hifi-gan.
    Imports models.py directly from the hifi-gan repo folder.
    """
    ckpt_path = Path(checkpoint_path)
    cfg_path = Path(config_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"HiFi-GAN checkpoint not found: {ckpt_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"HiFi-GAN config not found: {cfg_path}")

    # Load config
    with open(cfg_path) as f:
        config = json.load(f)

    # Import hifi-gan models.py from its repo folder
    hifigan_root = str(ckpt_path.parent.parent)  # UNIVERSAL_V1 -> hifi-gan root
    if hifigan_root not in sys.path:
        sys.path.insert(0, hifigan_root)

    try:
        from models import Generator as HiFiGANGenerator
    except ImportError as e:
        raise ImportError(
            f"Cannot import HiFi-GAN models.py from {hifigan_root}. "
            f"Make sure hifi-gan repo is at: {hifigan_root}\n"
            f"Original error: {e}"
        )

    # AttrDict for config
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    h = AttrDict(config)

    # Build and load model
    generator = HiFiGANGenerator(h).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # checkpoint may be raw state dict or wrapped
    if 'generator' in ckpt:
        generator.load_state_dict(ckpt['generator'])
    else:
        generator.load_state_dict(ckpt)

    generator.eval()
    generator.remove_weight_norm()
    return generator


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_generator_model(args, logger) -> FlowMatchingUNet:
    logger.info("Creating Generator model...")
    model = FlowMatchingUNet(
        mel_channels=80,
        model_channels=args.model_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=list(args.channel_mult),
        attention_resolutions=list(args.attention_resolutions),
        num_heads=args.num_heads,
        dropout=args.dropout,
        content_dim=512,
        prosody_dim=32,
        timbre_dim=256,
    ).to(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
    except Exception as e:
        logger.error(f"torch.load failed: {e}")
        sys.exit(1)

    is_dict = isinstance(ckpt, dict)
    loaded = False

    if not loaded and is_dict:
        ema = ckpt.get('ema_state_dict')
        if ema is not None:
            shadow = ema.get('shadow') if isinstance(ema, dict) else None
            if shadow is not None:
                try:
                    model.load_state_dict(shadow)
                    logger.info("  Loaded EMA shadow weights")
                    loaded = True
                except Exception as e:
                    logger.warning(f"  EMA load failed: {e}")

    if not loaded and is_dict and 'model_state_dict' in ckpt:
        try:
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info("  Loaded model_state_dict")
            loaded = True
        except Exception as e:
            logger.warning(f"  model_state_dict failed: {e}")

    if not loaded:
        try:
            model.load_state_dict(ckpt if is_dict else ckpt)
            logger.info("  Loaded as raw state dict")
            loaded = True
        except Exception as e:
            logger.error(f"  Cannot load weights: {e}")
            sys.exit(1)

    if is_dict:
        logger.info(f"  Iteration: {ckpt.get('iteration', '?')}")

    return model


# ---------------------------------------------------------------------------
# Encoder loading
# ---------------------------------------------------------------------------
def load_encoders(args, logger):
    """
    Load real encoders from checkpoint paths. Returns (content_enc, prosody_enc,
    timbre_enc, failed) where `failed` is a list of names that were requested
    (a --*_encoder_ckpt was passed) but failed to load. The caller can fail
    fast if any are in `failed` and the user didn't pass --allow_dummy.
    """
    content_enc = None
    prosody_enc = None
    timbre_enc = None
    failed = []

    if args.content_encoder_ckpt:
        try:
            from training.dataset import load_content_encoder
            content_enc = load_content_encoder(args.content_encoder_ckpt, device=args.device)
            logger.info(f"  Content Encoder loaded: {args.content_encoder_ckpt}")
        except Exception as e:
            logger.error(f"  Content Encoder load FAILED: {e}")
            failed.append('content')

    if args.prosody_encoder_ckpt:
        try:
            from training.dataset import load_prosody_encoder
            prosody_enc = load_prosody_encoder(args.prosody_encoder_ckpt, device=args.device)
            logger.info(f"  Prosody Encoder loaded: {args.prosody_encoder_ckpt}")
        except Exception as e:
            logger.error(f"  Prosody Encoder load FAILED: {e}")
            failed.append('prosody')

    if args.timbre_encoder_ckpt:
        try:
            from training.dataset import load_timbre_encoder
            timbre_enc = load_timbre_encoder(args.timbre_encoder_ckpt, device=args.device)
            logger.info(f"  Timbre Encoder loaded: {args.timbre_encoder_ckpt}")
        except Exception as e:
            logger.error(f"  Timbre Encoder load FAILED: {e}")
            failed.append('timbre')

    return content_enc, prosody_enc, timbre_enc, failed


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_content(mel: torch.Tensor, encoder, device: str) -> torch.Tensor:
    with torch.no_grad():
        if encoder is None:
            return torch.randn(1, 512, mel.shape[-1], device=device)
        out = encoder(mel.to(device))
        if isinstance(out, tuple):
            out = out[0]
        return out


def extract_prosody(audio: torch.Tensor, encoder, device: str,
                    target_T: int = None) -> torch.Tensor:
    """
    Extract prosody embedding from raw audio.

    The Prosody Encoder has its own `extract_explicit_features` method that
    computes [F0, energy, voicing, rhythm] using torchcrepe + librosa. We
    must use that — the previous version of this function fed RANDOM noise
    as the explicit features, which produced garbled output.
    """
    with torch.no_grad():
        if encoder is None:
            T = target_T if target_T is not None else audio.shape[1] // 320
            return torch.randn(1, 32, T, device=device)

        # Extract REAL explicit features from raw audio.
        # The encoder may be wrapped (ProsodyEncoderWrapper in training/dataset.py)
        # which doesn't forward `extract_explicit_features`. Unwrap if needed.
        feature_extractor = encoder
        while not hasattr(feature_extractor, 'extract_explicit_features'):
            if not hasattr(feature_extractor, 'encoder'):
                raise AttributeError(
                    "Could not find extract_explicit_features on the prosody "
                    "encoder or any wrapped sub-module."
                )
            feature_extractor = feature_extractor.encoder

        # extract_explicit_features expects [B, L] or [L]; our audio is [1, L]
        explicit_features = feature_extractor.extract_explicit_features(audio.cpu())
        # Ensure shape is [1, 4, T]
        if explicit_features.dim() == 2:
            explicit_features = explicit_features.unsqueeze(0)
        explicit_features = explicit_features.to(device)

        out = encoder(explicit_features=explicit_features)
        if isinstance(out, tuple):
            out = out[0]

        # Align time dimension to target_T (content length) if requested.
        # Prosody is usually off by 0-2 frames vs mel due to torchcrepe's hop.
        if target_T is not None and out.shape[-1] != target_T:
            cur_T = out.shape[-1]
            if cur_T > target_T:
                out = out[..., :target_T]
            else:
                # Pad with edge value (replicate last frame)
                pad = target_T - cur_T
                out = torch.nn.functional.pad(out, (0, pad), mode='replicate')
        return out


def extract_timbre(mel: torch.Tensor, encoder, device: str) -> torch.Tensor:
    with torch.no_grad():
        if encoder is None:
            return torch.randn(1, 256, device=device)
        out = encoder(mel.to(device))
        if isinstance(out, tuple):
            out = out[0]
        return out


# ---------------------------------------------------------------------------
# Voice conversion pipeline
# ---------------------------------------------------------------------------
def voice_conversion(
    source_path: str,
    reference_path: str,
    output_path: str,
    generator: MelGenerator,
    content_enc, prosody_enc, timbre_enc,
    vocoder=None,
    device: str = 'cpu',
    logger=None,
):
    log = logger or logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("Voice Conversion Pipeline")
    log.info("=" * 70)

    # 1. Load audio at 16k (for encoders)
    log.info("Step 1: Loading audio...")
    try:
        source_wav = load_audio(source_path, target_sr=16000).to(device)
        reference_wav = load_audio(reference_path, target_sr=16000).to(device)
    except Exception as e:
        log.error(f"  Failed to load audio: {e}")
        return False
    log.info(f"  Source:    {source_wav.shape[1]} samples ({source_wav.shape[1]/16000:.2f}s)")
    log.info(f"  Reference: {reference_wav.shape[1]} samples ({reference_wav.shape[1]/16000:.2f}s)")

    # 2. Compute mel at 16k — two variants needed:
    #    mel_for_ce   : log(clamp), no norm  → Content Encoder (matches CE training)
    #    mel_16k      : log+per-utt norm     → Timbre Encoder (matches Timbre training)
    log.info("Step 2: Computing mel spectrograms (16k)...")
    source_mel_for_ce = compute_mel_for_ce(source_wav).to(device)
    source_mel_16k = compute_mel_16k(source_wav).to(device)
    reference_mel_16k = compute_mel_16k(reference_wav).to(device)
    log.info(f"  Source mel (CE):      {list(source_mel_for_ce.shape)}")
    log.info(f"  Source mel (timbre):  {list(source_mel_16k.shape)}")
    log.info(f"  Reference mel:        {list(reference_mel_16k.shape)}")

    # 3. Extract content from SOURCE using CE-specific mel (log clamp, no norm)
    log.info("Step 3: Extracting content from source...")
    content = extract_content(source_mel_for_ce, content_enc, device)
    log.info(f"  Content: {list(content.shape)}")

    # 4. Extract prosody from SOURCE (uses raw audio, not mel — the prosody
    #    encoder's extract_explicit_features needs the waveform for F0/energy).
    #    Align to content's T to avoid off-by-one mismatch from torchcrepe hop.
    log.info("Step 4: Extracting prosody from source...")
    prosody = extract_prosody(source_wav, prosody_enc, device,
                              target_T=content.shape[-1])
    log.info(f"  Prosody: {list(prosody.shape)}")

    # 5. Extract timbre from REFERENCE
    log.info("Step 5: Extracting timbre from reference...")
    timbre = extract_timbre(reference_mel_16k, timbre_enc, device)
    log.info(f"  Timbre: {list(timbre.shape)}")

    # 6. Generate converted mel
    log.info("Step 6: Generating converted mel...")
    try:
        converted_mel = generator.generate(content, prosody, timbre)
    except Exception as e:
        log.error(f"  Generation failed: {e}")
        traceback.print_exc()
        return False
    log.info(f"  Converted mel: {list(converted_mel.shape)}")

    if torch.isnan(converted_mel).any() or torch.isinf(converted_mel).any():
        log.error("  Generated mel contains NaN/Inf!")
        return False

    log.info(f"  Mel stats: min={converted_mel.min():.3f}, max={converted_mel.max():.3f}, "
             f"mean={converted_mel.mean():.3f}, std={converted_mel.std():.3f}")

    # 7. Vocoder or save mel
    log.info("Step 7: Converting to audio...")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if vocoder is not None:
        try:
            log.info("  Running HiFi-GAN vocoder...")
            # After FIX 2 the Generator outputs raw log-mel at 16kHz/hop=320.
            # Pass it directly to the vocoder — no interpolation, no denormalization.
            log.info(f"  Vocoder input mel: {list(converted_mel.shape)}")

            with torch.no_grad():
                audio = vocoder(converted_mel)  # [1, 1, samples] or [1, samples]

            # Normalize shape to [1, samples]
            if audio.dim() == 3:
                audio = audio.squeeze(1)
            elif audio.dim() == 1:
                audio = audio.unsqueeze(0)

            # Clamp to valid range
            audio = audio.clamp(-1.0, 1.0)

            torchaudio.save(str(output), audio.cpu(), 16000)
            log.info(f"  Saved audio: {output} (16000 Hz)")

        except Exception as e:
            log.error(f"  Vocoder failed: {e}")
            traceback.print_exc()
            # Fall back to saving mel
            mel_path = output.with_suffix('.pt')
            torch.save(converted_mel.cpu(), str(mel_path))
            log.info(f"  Saved mel instead: {mel_path}")
            return False
    else:
        mel_path = output.with_suffix('.pt')
        torch.save(converted_mel.cpu(), str(mel_path))
        log.info(f"  No vocoder — saved mel: {mel_path}")

    log.info("=" * 70)
    log.info("Done!")
    log.info("=" * 70)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Voice Conversion (Generator Module 4)')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Generator checkpoint')
    parser.add_argument('--source', type=str, required=True,
                        help='Source audio (provides content + prosody)')
    parser.add_argument('--reference', type=str, required=True,
                        help='Reference audio (provides timbre)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path (.wav with vocoder, .pt without)')

    # Encoder checkpoints (optional)
    parser.add_argument('--content_encoder_ckpt', type=str, default=None)
    parser.add_argument('--prosody_encoder_ckpt', type=str, default=None)
    parser.add_argument('--timbre_encoder_ckpt', type=str, default=None)

    # Vocoder
    parser.add_argument('--vocoder_ckpt', type=str, default=None,
                        help='Path to HiFi-GAN generator checkpoint (g_02500000)')
    parser.add_argument('--vocoder_config', type=str, default=None,
                        help='Path to HiFi-GAN config.json')

    # Generation
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--sampler', type=str, default='euler',
                        choices=['euler', 'midpoint', 'heun'])
    parser.add_argument('--use_cfg', action='store_true')
    parser.add_argument('--allow_dummy', action='store_true',
                        help='Allow encoder load failures to fall back to '
                             'random features. Without this flag, any encoder '
                             'path passed via --*_encoder_ckpt that fails to '
                             'load will cause the program to exit. Use only '
                             'when intentionally testing with random features.')
    parser.add_argument('--cfg_scale', type=float, default=1.5)

    # Model architecture (must match training)
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2])
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[4])
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)

    args = parser.parse_args()
    logger = setup_logging()

    # Validate audio paths
    for name, path in [('source', args.source), ('reference', args.reference)]:
        if not Path(path).exists():
            logger.error(f"{name} file not found: {path}")
            sys.exit(1)

    # Validate vocoder args — if one is given, both are required
    if bool(args.vocoder_ckpt) != bool(args.vocoder_config):
        logger.error("--vocoder_ckpt and --vocoder_config must both be provided together")
        sys.exit(1)

    # Load Generator
    model = load_generator_model(args, logger)
    generator = MelGenerator(
        model=model,
        sampler_method=args.sampler,
        num_steps=args.num_steps,
        use_cfg=args.use_cfg,
        cfg_scale=args.cfg_scale,
        device=args.device,
    )
    logger.info(f"\n{generator}")

    # Load encoders
    logger.info("\nLoading encoders...")
    content_enc, prosody_enc, timbre_enc, failed = load_encoders(args, logger)

    # Fatal-on-failure: if user passed --content_encoder_ckpt etc. but the file
    # couldn't load, exit. Otherwise we'd silently fall back to random features
    # and the user wastes time listening to noise wondering what went wrong.
    if failed and not args.allow_dummy:
        logger.error("")
        logger.error("=" * 70)
        logger.error(f"FATAL: encoder(s) failed to load: {', '.join(failed)}")
        logger.error("Check the paths above. Common causes:")
        logger.error("  - Wrong relative path (try '../' if running inside Generator/)")
        logger.error("  - Folder name typo (Content_Encoder vs Content Encoder)")
        logger.error("  - Checkpoint file missing")
        logger.error("")
        logger.error("If you actually want to test with random features, pass --allow_dummy")
        logger.error("=" * 70)
        sys.exit(1)

    # Also fail-fast if NO encoder paths were given at all — the output is
    # guaranteed to be noise in that case.
    if all(e is None for e in [content_enc, prosody_enc, timbre_enc]):
        if not args.allow_dummy:
            logger.error("No encoder checkpoints provided. Output would be noise.")
            logger.error("Pass --content_encoder_ckpt / --prosody_encoder_ckpt / "
                         "--timbre_encoder_ckpt, or use --allow_dummy to proceed anyway.")
            sys.exit(1)
        logger.warning("\n  WARNING: All encoders are dummy (random features).")
        logger.warning("  Output will be noise.\n")

    # Load vocoder
    vocoder = None
    if args.vocoder_ckpt and args.vocoder_config:
        logger.info(f"\nLoading HiFi-GAN vocoder...")
        logger.info(f"  Checkpoint: {args.vocoder_ckpt}")
        logger.info(f"  Config:     {args.vocoder_config}")
        try:
            vocoder = load_hifigan(args.vocoder_ckpt, args.vocoder_config, args.device)
            logger.info("  HiFi-GAN loaded successfully")
        except Exception as e:
            logger.error(f"  HiFi-GAN load failed: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Run conversion
    success = voice_conversion(
        source_path=args.source,
        reference_path=args.reference,
        output_path=args.output,
        generator=generator,
        content_enc=content_enc,
        prosody_enc=prosody_enc,
        timbre_enc=timbre_enc,
        vocoder=vocoder,
        device=args.device,
        logger=logger,
    )

    if not success:
        logger.error("Voice conversion failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()