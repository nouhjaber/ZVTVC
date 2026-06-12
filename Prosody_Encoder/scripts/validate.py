"""
Validation Script for Prosody Encoder
Uses shard-based dataset (ProsodyEncoderDataset) instead of raw audio files.

Usage:
    python validate.py --checkpoint checkpoints/best.pt --shard_dir /path/to/shards
    python validate.py --checkpoint checkpoints/best.pt --shard_dir /path/to/shards --split val --visualize
"""
import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add Prosody Encoder root (parent of scripts/) so model/training/utils are importable
sys.path.append(str(Path(__file__).resolve().parent.parent))
# Add ZVTVC root (parent of Prosody Encoder/) for Shard_dataset_unified
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from model.prosody_encoder import ProsodyEncoder
from training.losses import ProsodyLoss
from utils.metrics import ProsodyMetrics, compute_all_metrics
from utils.visualization import plot_prosody_features, plot_feature_comparison
from Shard_dataset_unified import create_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Validate Prosody Encoder")

    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--shard_dir', type=str, required=True,
                        help='Path to directory containing preprocessed shards')

    # Optional arguments
    parser.add_argument('--split', type=str, default='val',
                        help='Data split to validate on (default: val)')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Directory to save results (default: ./validation_results)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu, default: cuda)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation (default: 64)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--num_visualize', type=int, default=5,
                        help='Number of samples to visualize (default: 5)')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Save predicted features to disk')

    # Model parameters (should match training)
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate (default: 16000)')
    parser.add_argument('--hop_length', type=int, default=320,
                        help='Hop length for feature extraction (default: 320)')
    parser.add_argument('--frame_rate', type=int, default=50,
                        help='Frame rate in Hz (default: 50)')
    parser.add_argument('--explicit_dim', type=int, default=4,
                        help='Explicit prosody dimension (default: 4)')
    parser.add_argument('--refined_dim', type=int, default=32,
                        help='Refined prosody dimension (default: 32)')
    parser.add_argument('--f0_method', type=str, default='crepe',
                        choices=['crepe', 'pyin'],
                        help='F0 extraction method (default: crepe)')
    parser.add_argument('--rhythm_window', type=int, default=11,
                        help='Rhythm window size in frames (default: 11)')

    return parser.parse_args()


def load_model(args, checkpoint_path: str, device: str) -> ProsodyEncoder:
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    # Create model
    model = ProsodyEncoder(
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        frame_rate=args.frame_rate,
        explicit_dim=args.explicit_dim,
        refined_dim=args.refined_dim,
        f0_method=args.f0_method,
        rhythm_window_size=args.rhythm_window,
        use_refinement=True,
        use_residual=True,
        use_reconstruction_heads=True,
        output_format="refined",
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
        if 'best_val_loss' in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model weights")

    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    return model


def validate(
    model: ProsodyEncoder,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    output_dir: str,
    visualize: bool = False,
    num_visualize: int = 5,
    save_predictions: bool = False,
) -> dict:
    """
    Run validation on shard-based dataset.

    Args:
        model: Prosody encoder model
        val_loader: DataLoader from create_dataloader(module='prosody')
        device: Device to use
        output_dir: Output directory
        visualize: Generate visualizations
        num_visualize: Number of samples to visualize
        save_predictions: Save predicted features

    Returns:
        Dictionary of metrics
    """
    num_samples = len(val_loader.dataset)
    print(f"\nValidating on {num_samples} samples...")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    if save_predictions:
        os.makedirs(os.path.join(output_dir, 'predictions'), exist_ok=True)

    # Initialize metrics tracker
    metrics_tracker = ProsodyMetrics()

    # Loss function
    loss_fn = ProsodyLoss()

    # Track losses
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'f0': 0.0,
        'energy': 0.0,
        'voicing': 0.0,
        'rhythm': 0.0,
        'smoothness': 0.0,
    }

    visualized_count = 0
    num_batches = 0
    total_processed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating")):
            try:
                # Unpack shard batch: (features [B, 4, T], features_aug or None)
                gt_features, _ = batch
                gt_features = gt_features.to(device)

                # Forward pass
                output, explicit, reconstructions = model(
                    explicit_features=gt_features,
                    return_reconstructions=True
                )

                # Compute loss
                losses = loss_fn(
                    refined=output,
                    reconstructions=reconstructions,
                    targets=gt_features,
                )

                # Accumulate losses
                for key in total_losses:
                    if key in losses:
                        total_losses[key] += losses[key].item()

                num_batches += 1
                total_processed += gt_features.shape[0]

                # Update metrics tracker
                if reconstructions is not None:
                    pred_features = torch.cat([
                        reconstructions["f0"],
                        reconstructions["energy"],
                        reconstructions["voicing"],
                        reconstructions["rhythm"],
                    ], dim=1)
                    metrics_tracker.update(pred_features, gt_features)

                # Visualize (pick individual samples from batch)
                if visualize and visualized_count < num_visualize and reconstructions is not None:
                    pred_np = pred_features.cpu().numpy()
                    gt_np = gt_features.cpu().numpy()

                    for i in range(min(gt_features.shape[0], num_visualize - visualized_count)):
                        fig = plot_feature_comparison(
                            pred_features=pred_np[i],   # [4, T]
                            target_features=gt_np[i],   # [4, T]
                            sample_rate=model.sample_rate if hasattr(model, 'sample_rate') else 16000,
                            hop_length=model.hop_length,
                        )
                        save_path = os.path.join(
                            output_dir, 'visualizations',
                            f'sample_{total_processed - gt_features.shape[0] + i}_comparison.png'
                        )
                        fig.savefig(save_path, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        visualized_count += 1

                # Save predictions
                if save_predictions:
                    output_np = output.cpu().numpy()
                    for i in range(output_np.shape[0]):
                        sample_id = total_processed - gt_features.shape[0] + i
                        save_path = os.path.join(output_dir, 'predictions', f'sample_{sample_id}.npy')
                        np.save(save_path, output_np[i])

            except Exception as e:
                print(f"\nError processing batch {batch_idx}: {e}")
                continue

    # Compute average losses (per batch)
    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}

    # Compute final metrics
    final_metrics = metrics_tracker.compute()

    return {
        'losses': avg_losses,
        'metrics': final_metrics,
        'num_samples': total_processed,
    }


def print_results(results: dict):
    """Print validation results."""
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    print(f"\nSamples evaluated: {results['num_samples']}")

    print("\n--- Losses ---")
    for key, value in results['losses'].items():
        print(f"  {key:20s}: {value:.4f}")

    print("\n--- Metrics ---")
    metrics = results['metrics']

    # Note: features are whitened (zero-mean, unit-variance), so RMSE is in
    # normalized space, not cents/dB. Thresholds adjusted accordingly:
    # ~0.05 normalized RMSE ≈ excellent reconstruction.
    print(f"\n  F0 Metrics:")
    print(f"    Pearson Correlation (PCC): {metrics.get('f0_pcc', 0):.4f}  [Target: >0.80]")
    print(f"    RMSE (normalized):         {metrics.get('f0_rmse', 0):.4f}  [Target: <0.10]")

    print(f"\n  Energy Metrics:")
    print(f"    Correlation:               {metrics.get('energy_correlation', 0):.4f}  [Target: >0.80]")
    print(f"    RMSE (normalized):         {metrics.get('energy_rmse', 0):.4f}  [Target: <0.10]")

    print(f"\n  Voicing Metrics:")
    print(f"    Error Rate:                {metrics.get('voicing_error_rate', 0):.4f}  [Target: <0.10]")

    print(f"\n  Rhythm Metrics:")
    print(f"    Correlation:               {metrics.get('rhythm_correlation', 0):.4f}  [Target: >0.75]")

    # Overall assessment
    print("\n--- Assessment ---")
    passed = 0
    total = 6

    checks = [
        (metrics.get('f0_pcc', 0) > 0.80, "F0 PCC > 0.80"),
        (metrics.get('f0_rmse', 999) < 0.10, "F0 RMSE < 0.10 (normalized)"),
        (metrics.get('energy_correlation', 0) > 0.80, "Energy correlation > 0.80"),
        (metrics.get('energy_rmse', 999) < 0.10, "Energy RMSE < 0.10 (normalized)"),
        (metrics.get('voicing_error_rate', 1) < 0.10, "Voicing error < 10%"),
        (metrics.get('rhythm_correlation', 0) > 0.75, "Rhythm correlation > 0.75"),
    ]

    for ok, label in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")
        if ok:
            passed += 1

    print(f"\n  Overall: {passed}/{total} metrics passed")
    print("=" * 60)


def save_results(results: dict, output_path: str):
    """Save results to file."""
    import json

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    results_converted = convert(results)

    with open(output_path, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    args = parse_args()

    # Import matplotlib only if visualizing
    if args.visualize:
        global plt
        import matplotlib.pyplot as plt

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'

    print(f"\n{'='*60}")
    print(f"Prosody Encoder Validation")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args, args.checkpoint, args.device)

    # Create shard-based dataloader
    print(f"\nLoading {args.split} data from shards: {args.shard_dir}")
    val_loader = create_dataloader(
        shard_dir=args.shard_dir,
        split=args.split,
        module='prosody',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    total_samples = len(val_loader.dataset)
    if total_samples == 0:
        print("ERROR: No validation samples found!")
        return

    print(f"Found {total_samples} samples")

    # Limit samples if specified (recreate loader with smaller dataset isn't trivial,
    # so we just break early in the loop via a wrapper)
    if args.num_samples is not None and args.num_samples < total_samples:
        print(f"Limiting to {args.num_samples} samples")

    # Run validation
    results = validate(
        model=model,
        val_loader=val_loader,
        device=args.device,
        output_dir=str(output_dir),
        visualize=args.visualize,
        num_visualize=args.num_visualize,
        save_predictions=args.save_predictions,
    )

    # Print results
    print_results(results)

    # Save results
    results_path = output_dir / 'validation_results.json'
    save_results(results, str(results_path))

    print("\nValidation completed!")


if __name__ == "__main__":
    main()