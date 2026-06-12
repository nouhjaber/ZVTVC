"""
Evaluation Script for Generator (Module 4)
Computes: reconstruction quality, mel statistics, inference latency, step comparison.

Usage:
    # Precomputed shards (recommended — fastest, most accurate)
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --shard_dir /content/outputs \
        --use_precomputed

    # Raw shards + dummy encoders (random features — MSE meaningless, but tests pipeline)
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --shard_dir /content/shards \
        --use_dummy_encoders

    # Latency benchmark only (no data needed)
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --eval_latency

    # Compare ODE step counts
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --shard_dir /content/outputs \
        --use_precomputed \
        --eval_steps

    # Everything
    python scripts/evaluate.py \
        --checkpoint checkpoints/best_model.pt \
        --shard_dir /content/outputs \
        --use_precomputed \
        --eval_latency \
        --eval_steps
"""

import torch
import torch.nn as nn
import argparse
import logging
import time
import traceback
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup — allow running from Generator/ or Generator/scripts/
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_GENERATOR_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == 'scripts' else _SCRIPT_DIR
sys.path.insert(0, str(_GENERATOR_ROOT))

from model.unet import FlowMatchingUNet
from inference import MelGenerator
from training import get_dataloader, DummyEncoderWrapper
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
# Model loading — handles every checkpoint format we've ever saved
# ---------------------------------------------------------------------------
def load_model(args, logger) -> FlowMatchingUNet:
    """Create model + load weights.  Dies with clear message if anything fails."""

    logger.info("Creating model...")
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

    # Determine what's inside
    is_dict = isinstance(ckpt, dict)

    # Prioritised key search
    loaded = False

    # 1. EMA weights (best quality)
    if not loaded and args.use_ema and is_dict:
        ema = ckpt.get('ema_state_dict')
        if ema is not None:
            shadow = ema.get('shadow') if isinstance(ema, dict) else None
            if shadow is not None:
                try:
                    model.load_state_dict(shadow)
                    logger.info("  Loaded EMA shadow weights")
                    loaded = True
                except Exception as e:
                    logger.warning(f"  EMA shadow load failed: {e}")

    # 2. model_state_dict
    if not loaded and is_dict and 'model_state_dict' in ckpt:
        try:
            model.load_state_dict(ckpt['model_state_dict'])
            logger.info("  Loaded model_state_dict")
            loaded = True
        except Exception as e:
            logger.warning(f"  model_state_dict load failed: {e}")

    # 3. state_dict (some older saves)
    if not loaded and is_dict and 'state_dict' in ckpt:
        try:
            model.load_state_dict(ckpt['state_dict'])
            logger.info("  Loaded state_dict")
            loaded = True
        except Exception as e:
            logger.warning(f"  state_dict load failed: {e}")

    # 4. Raw state dict (the file IS the state dict)
    if not loaded:
        try:
            model.load_state_dict(ckpt if is_dict else ckpt)
            logger.info("  Loaded checkpoint as raw state dict")
            loaded = True
        except Exception as e:
            logger.error(f"  Cannot load weights from checkpoint: {e}")
            sys.exit(1)

    if is_dict:
        logger.info(f"  Iteration: {ckpt.get('iteration', '?')}")
        logger.info(f"  Best val loss: {ckpt.get('best_loss', '?')}")

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"  Parameters: {param_count:.2f}M")

    return model


# ---------------------------------------------------------------------------
# Dataloader builder — precomputed or on-the-fly, val or train fallback
# ---------------------------------------------------------------------------
def build_dataloader(args, logger):
    """Returns a DataLoader or None on failure.  Never crashes."""
    if args.shard_dir is None:
        logger.error("No --shard_dir provided")
        return None

    shard_dir = Path(args.shard_dir)
    if not shard_dir.exists():
        logger.error(f"Shard directory not found: {shard_dir}")
        return None

    # Check what's inside
    npz_files = sorted(shard_dir.glob('*.npz'))
    if len(npz_files) == 0:
        logger.error(f"No .npz files in {shard_dir}")
        return None
    logger.info(f"  Found {len(npz_files)} .npz files in {shard_dir}")

    try:
        if args.use_precomputed:
            from training.dataset import PrecomputedGeneratorDataset

            # Try val split first, fall back to train
            dataset = None
            for split in ('val', 'train'):
                try:
                    ds = PrecomputedGeneratorDataset(
                        shard_dir=str(shard_dir),
                        split=split,
                        lang=None,
                        sequence_length=args.sequence_length,
                    )
                    if len(ds) > 0:
                        dataset = ds
                        logger.info(f"  Using '{split}' split: {len(dataset)} samples, "
                                    f"{len(dataset.shard_files)} shards")
                        break
                except (ValueError, Exception) as e:
                    logger.info(f"  Split '{split}' not available: {e}")

            if dataset is None:
                logger.error("  No usable split found")
                return None

            loader = get_dataloader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
                use_shard_sampler=False,  # deterministic order for eval
            )
            return loader

        elif args.use_dummy_encoders:
            from training.dataset import OnTheFlyGeneratorDataset

            content_enc = DummyEncoderWrapper(512, 'frame')
            prosody_enc = DummyEncoderWrapper(32, 'frame')
            timbre_enc = DummyEncoderWrapper(256, 'global')

            dataset = None
            for split in ('val', 'train'):
                try:
                    ds = OnTheFlyGeneratorDataset(
                        shard_dir=str(shard_dir),
                        split=split,
                        lang=None,
                        content_encoder=content_enc,
                        prosody_encoder=prosody_enc,
                        timbre_encoder=timbre_enc,
                        sequence_length=args.sequence_length,
                        device='cpu',
                    )
                    if len(ds) > 0:
                        dataset = ds
                        logger.info(f"  Using '{split}' split (dummy encoders): "
                                    f"{len(dataset)} samples")
                        break
                except (ValueError, Exception) as e:
                    logger.info(f"  Split '{split}' not available: {e}")

            if dataset is None:
                logger.error("  No usable split found")
                return None

            loader = get_dataloader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,  # on-the-fly MUST be 0
                pin_memory=True,
                drop_last=False,
                use_shard_sampler=False,
            )
            return loader

        else:
            logger.error("Must specify --use_precomputed or --use_dummy_encoders")
            return None

    except Exception as e:
        logger.error(f"Failed to build dataloader: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Reconstruction evaluation
# ---------------------------------------------------------------------------
def evaluate_reconstruction(
    generator: MelGenerator,
    dataloader,
    num_samples: int = 100,
    device: str = 'cuda',
    logger=None,
) -> dict:
    """MSE/MAE between generated and target mel.  Skips bad batches."""
    mse_list, mae_list = [], []
    mel_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
    nan_count = 0
    error_count = 0
    count = 0

    if logger:
        logger.info(f"  Running reconstruction on up to {num_samples} samples...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Reconstruction", leave=False):
            if count >= num_samples:
                break

            try:
                mel_target = batch['mel'].to(device)
                content = batch['content'].to(device)
                prosody = batch['prosody'].to(device)
                timbre = batch['timbre'].to(device)

                # Sanity: skip if inputs are bad
                if (torch.isnan(mel_target).any() or torch.isnan(content).any()
                        or torch.isnan(prosody).any() or torch.isnan(timbre).any()):
                    error_count += 1
                    continue

                mel_pred = generator.generate(content, prosody, timbre)

                # NaN / Inf check on output
                if torch.isnan(mel_pred).any() or torch.isinf(mel_pred).any():
                    nan_count += 1
                    continue

                # Align lengths
                T = min(mel_pred.shape[-1], mel_target.shape[-1])
                mel_pred = mel_pred[..., :T]
                mel_target = mel_target[..., :T]

                mse = torch.nn.functional.mse_loss(mel_pred, mel_target).item()
                mae = torch.nn.functional.l1_loss(mel_pred, mel_target).item()

                mse_list.append(mse)
                mae_list.append(mae)

                mel_stats['min'].append(mel_pred.min().item())
                mel_stats['max'].append(mel_pred.max().item())
                mel_stats['mean'].append(mel_pred.mean().item())
                mel_stats['std'].append(mel_pred.std().item())

                count += mel_target.shape[0]

            except RuntimeError as e:
                error_count += 1
                if error_count <= 5:
                    tqdm.write(f"  [!] RuntimeError in batch: {e}")
                continue
            except Exception as e:
                error_count += 1
                if error_count <= 5:
                    tqdm.write(f"  [!] Error in batch: {e}")
                continue

    # Build result dict — always return something, never crash
    empty = float('nan')
    if not mse_list:
        return {
            'mse_mean': empty, 'mse_std': empty,
            'mae_mean': empty, 'mae_std': empty,
            'num_samples': 0, 'nan_count': nan_count, 'error_count': error_count,
            'mel_min': empty, 'mel_max': empty,
            'mel_mean': empty, 'mel_std': empty,
        }

    return {
        'mse_mean': float(np.mean(mse_list)),
        'mse_std': float(np.std(mse_list)),
        'mae_mean': float(np.mean(mae_list)),
        'mae_std': float(np.std(mae_list)),
        'num_samples': count,
        'nan_count': nan_count,
        'error_count': error_count,
        'mel_min': float(np.mean(mel_stats['min'])),
        'mel_max': float(np.mean(mel_stats['max'])),
        'mel_mean': float(np.mean(mel_stats['mean'])),
        'mel_std': float(np.mean(mel_stats['std'])),
    }


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------
def evaluate_latency(
    generator: MelGenerator,
    num_frames: int = 400,
    num_runs: int = 50,
    device: str = 'cuda',
    logger=None,
) -> dict:
    """Time the generate() call.  Returns ms stats."""
    if logger:
        logger.info(f"  Benchmarking: {num_runs} runs, {num_frames} frames, "
                     f"{generator.num_steps} ODE steps")

    content = torch.randn(1, 512, num_frames, device=device)
    prosody = torch.randn(1, 32, num_frames, device=device)
    timbre = torch.randn(1, 256, device=device)

    use_cuda = 'cuda' in str(device)

    # Warmup
    for _ in range(5):
        try:
            _ = generator.generate(content, prosody, timbre)
        except Exception:
            pass
    if use_cuda:
        torch.cuda.synchronize()

    latencies = []
    for _ in tqdm(range(num_runs), desc="Latency", leave=False):
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = generator.generate(content, prosody, timbre)
        if use_cuda:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies)
    fps = num_frames / (arr.mean() / 1000.0) if arr.mean() > 0 else 0.0

    return {
        'mean_ms': float(arr.mean()),
        'std_ms': float(arr.std()),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'p99_ms': float(np.percentile(arr, 99)),
        'num_frames': num_frames,
        'num_steps': generator.num_steps,
        'frames_per_sec': fps,
    }


# ---------------------------------------------------------------------------
# ODE step comparison
# ---------------------------------------------------------------------------
def evaluate_step_comparison(
    generator: MelGenerator,
    dataloader,
    steps_list: list,
    num_samples: int = 20,
    device: str = 'cuda',
    logger=None,
) -> dict:
    """Run reconstruction at several step counts, report MSE/MAE for each."""
    if logger:
        logger.info(f"  Comparing ODE steps: {steps_list} on {num_samples} samples each")

    original_steps = generator.num_steps
    results = {}

    for n in steps_list:
        generator.set_num_steps(n)
        metrics = evaluate_reconstruction(
            generator, dataloader,
            num_samples=num_samples, device=device,
        )
        results[n] = metrics
        if logger:
            logger.info(f"    Steps={n:3d}: MSE={metrics['mse_mean']:.6f}  "
                         f"MAE={metrics['mae_mean']:.6f}  "
                         f"({metrics['num_samples']} samples)")

    generator.set_num_steps(original_steps)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Evaluate Generator (Module 4)')

    # -- Required --
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to Generator checkpoint')

    # -- Data --
    parser.add_argument('--shard_dir', type=str, default=None,
                        help='Path to shard directory')
    parser.add_argument('--use_precomputed', action='store_true', default=False,
                        help='Shards contain precomputed encoder outputs')
    parser.add_argument('--use_dummy_encoders', action='store_true', default=False,
                        help='Use random dummy encoders with raw audio shards')
    parser.add_argument('--sequence_length', type=int, default=400,
                        help='Sequence length for eval samples')

    # -- Eval modes --
    parser.add_argument('--eval_latency', action='store_true',
                        help='Run inference latency benchmark')
    parser.add_argument('--eval_steps', action='store_true',
                        help='Compare quality at different ODE step counts')

    # -- Eval params --
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Max samples for reconstruction eval')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_steps', type=int, default=10,
                        help='ODE sampling steps (default for reconstruction)')
    parser.add_argument('--sampler', type=str, default='euler',
                        choices=['euler', 'midpoint', 'heun'],
                        help='ODE solver method')

    # -- Model architecture (must match training) --
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Prefer EMA weights if available')
    parser.add_argument('--no_ema', action='store_true', default=False,
                        help='Force non-EMA weights')
    parser.add_argument('--model_channels', type=int, default=256)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2])
    parser.add_argument('--attention_resolutions', type=int, nargs='+', default=[4])
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout at eval (should be 0)')

    args = parser.parse_args()

    # Handle --no_ema override
    if args.no_ema:
        args.use_ema = False

    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("Generator (Module 4) — Evaluation")
    logger.info("=" * 70)
    logger.info(f"  Checkpoint:  {args.checkpoint}")
    logger.info(f"  Device:      {args.device}")
    logger.info(f"  ODE steps:   {args.num_steps}")
    logger.info(f"  Sampler:     {args.sampler}")

    # ---- Decide what to run ----
    has_data = args.shard_dir is not None
    do_reconstruction = has_data
    do_latency = args.eval_latency
    do_steps = args.eval_steps and has_data

    if not do_reconstruction and not do_latency and not do_steps:
        logger.info("  No eval mode selected — running latency benchmark by default.")
        do_latency = True

    if has_data and not args.use_precomputed and not args.use_dummy_encoders:
        logger.info("  No encoder mode specified — defaulting to --use_precomputed.")
        args.use_precomputed = True

    # ---- Load model ----
    model = load_model(args, logger)

    generator = MelGenerator(
        model=model,
        sampler_method=args.sampler,
        num_steps=args.num_steps,
        device=args.device,
    )
    logger.info(f"\n{generator}")

    # ================================================================
    # 1. RECONSTRUCTION
    # ================================================================
    dataloader = None
    if do_reconstruction:
        logger.info("\n" + "=" * 70)
        logger.info("RECONSTRUCTION QUALITY")
        logger.info("=" * 70)

        dataloader = build_dataloader(args, logger)
        if dataloader is not None:
            metrics = evaluate_reconstruction(
                generator, dataloader,
                num_samples=args.num_samples,
                device=args.device,
                logger=logger,
            )

            logger.info(f"\n  {'Metric':<20} {'Value':>12}")
            logger.info(f"  {'-'*20} {'-'*12}")
            logger.info(f"  {'MSE':<20} {metrics['mse_mean']:>12.6f} +/- {metrics['mse_std']:.6f}")
            logger.info(f"  {'MAE':<20} {metrics['mae_mean']:>12.6f} +/- {metrics['mae_std']:.6f}")
            logger.info(f"  {'Mel range':<20} [{metrics['mel_min']:.2f}, {metrics['mel_max']:.2f}]")
            logger.info(f"  {'Mel mean':<20} {metrics['mel_mean']:>12.4f}")
            logger.info(f"  {'Mel std':<20} {metrics['mel_std']:>12.4f}")
            logger.info(f"  {'Samples evaluated':<20} {metrics['num_samples']:>12d}")
            if metrics['nan_count'] > 0:
                logger.warning(f"  {'NaN outputs':<20} {metrics['nan_count']:>12d}")
            if metrics['error_count'] > 0:
                logger.warning(f"  {'Batch errors':<20} {metrics['error_count']:>12d}")
        else:
            logger.error("  Dataloader failed — skipping reconstruction eval.")

    # ================================================================
    # 2. STEP COMPARISON
    # ================================================================
    if do_steps:
        logger.info("\n" + "=" * 70)
        logger.info("ODE STEP COMPARISON")
        logger.info("=" * 70)

        if dataloader is None:
            dataloader = build_dataloader(args, logger)

        if dataloader is not None:
            step_results = evaluate_step_comparison(
                generator, dataloader,
                steps_list=[5, 10, 20, 50],
                num_samples=min(20, args.num_samples),
                device=args.device,
                logger=logger,
            )

            # Summary table
            logger.info(f"\n  {'Steps':>6}  {'MSE':>12}  {'MAE':>12}")
            logger.info(f"  {'-----':>6}  {'---':>12}  {'---':>12}")
            for n, m in sorted(step_results.items()):
                logger.info(f"  {n:>6}  {m['mse_mean']:>12.6f}  {m['mae_mean']:>12.6f}")
        else:
            logger.error("  Dataloader failed — skipping step comparison.")

    # ================================================================
    # 3. LATENCY
    # ================================================================
    if do_latency:
        logger.info("\n" + "=" * 70)
        logger.info("INFERENCE LATENCY")
        logger.info("=" * 70)

        lat = evaluate_latency(
            generator, num_frames=400, num_runs=50,
            device=args.device, logger=logger,
        )

        logger.info(f"\n  {'Metric':<25} {'Value':>12}")
        logger.info(f"  {'-'*25} {'-'*12}")
        logger.info(f"  {'Mean':<25} {lat['mean_ms']:>10.2f} ms")
        logger.info(f"  {'Median (p50)':<25} {lat['p50_ms']:>10.2f} ms")
        logger.info(f"  {'P95':<25} {lat['p95_ms']:>10.2f} ms")
        logger.info(f"  {'P99':<25} {lat['p99_ms']:>10.2f} ms")
        logger.info(f"  {'Min':<25} {lat['min_ms']:>10.2f} ms")
        logger.info(f"  {'Max':<25} {lat['max_ms']:>10.2f} ms")
        logger.info(f"  {'Throughput':<25} {lat['frames_per_sec']:>10.0f} frames/s")

        # Realtime factor (assuming 50 Hz mel = 20ms per frame)
        rt_factor = lat['frames_per_sec'] * 0.02
        logger.info(f"  {'Realtime factor':<25} {rt_factor:>10.1f}x")

        # Step sweep
        logger.info("\n  Step count vs latency:")
        original_steps = generator.num_steps
        for n in [5, 10, 20, 50]:
            generator.set_num_steps(n)
            lat_n = evaluate_latency(
                generator, num_frames=400, num_runs=10, device=args.device,
            )
            logger.info(f"    {n:>3} steps: {lat_n['mean_ms']:>8.1f} ms  "
                         f"({lat_n['frames_per_sec']:.0f} frames/s)")
        generator.set_num_steps(original_steps)

    # ================================================================
    # Done
    # ================================================================
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()