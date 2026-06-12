"""
Validate Timbre Encoder
========================

Evaluation script for computing:
    - Equal Error Rate (EER)
    - Speaker similarity scores
    - Cross-lingual performance

Usage:
    python validate.py --checkpoint checkpoints/best_model.pt --dataset voxceleb
    python validate.py --checkpoint checkpoints/best_model.pt --dataset arabic --cross_lingual
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from model.ecapa_tdnn import create_ecapa_tdnn

# Add ZVTVC root to path for shard dataset.
# Validate.py lives at <ZVTVC>/Timbre_Encoder/training/Validate.py
# so the project root is parent.parent.parent, NOT parent.parent.
zvtvc_root = Path(__file__).parent.parent.parent
if str(zvtvc_root) not in sys.path:
    sys.path.insert(0, str(zvtvc_root))
from Shard_dataset_unified import TimbreEncoderDataset

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate timbre encoder')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--shard_dir',
        type=str,
        required=True,
        help='Path to preprocessed shards'
    )

    parser.add_argument(
        '--model_config',
        type=str,
        default='configs/model_config.yaml',
        help='Path to model config'
    )
    
    parser.add_argument(
        '--num_trials',
        type=int,
        default=1000,
        help='Number of verification trials'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./validation_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda or cpu)'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    
    parser.add_argument(
        '--cross_lingual',
        action='store_true',
        help='Evaluate cross-lingual performance'
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(checkpoint_path: str, model_config: dict, device: str):
    """Load model from checkpoint."""
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")

    try:
        # Create model — handle both flat {ecapa_tdnn: ...} and nested {model: {ecapa_tdnn: ...}}
        if 'ecapa_tdnn' in model_config:
            ecapa_cfg = model_config['ecapa_tdnn']
        elif 'model' in model_config and 'ecapa_tdnn' in model_config['model']:
            ecapa_cfg = model_config['model']['ecapa_tdnn']
        else:
            # Fallback to the same hardcoded config train.py uses
            ecapa_cfg = {
                'input_dim': 80,
                'embedding_dim': 256,
                'input_conv_channels': 128,
                'attention_channels': 128,
            }
            logger.warning("No ecapa_tdnn config found in YAML — using train.py defaults")

        model = create_ecapa_tdnn(ecapa_cfg)
        logger.debug("Model architecture created")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.debug("Model weights loaded from checkpoint")

        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded and set to eval mode on device: {device}")

        iteration = checkpoint.get('iteration', 'unknown')
        logger.info(f"Checkpoint iteration: {iteration}")
        print(f"Loaded model from iteration {iteration}")

        return model

    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {str(e)}", exc_info=True)
        raise


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """
    Extract embeddings for all samples.

    Returns:
        embeddings: [N, D] array of embeddings
        speaker_ids: [N] array of speaker IDs
    """
    logger.info("Starting embedding extraction")
    all_embeddings = []
    all_speaker_ids = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
        mels = batch['mel'].to(device)
        speaker_ids = batch['speaker_id']

        logger.debug(f"Processing batch {batch_idx + 1}: {mels.shape[0]} samples")

        # Extract embeddings
        embeddings = model(mels)

        all_embeddings.append(embeddings.cpu())
        all_speaker_ids.append(speaker_ids)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    speaker_ids = torch.cat(all_speaker_ids, dim=0).numpy()

    logger.info(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    return embeddings, speaker_ids


def compute_similarity_matrix(embeddings):
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(embeddings_norm, embeddings_norm.T)
    return similarity


def generate_trials(speaker_ids, num_trials):
    trials = []
    num_samples = len(speaker_ids)
    
    # Generate positive pairs (same speaker)
    for _ in range(num_trials):
        # Find speaker with multiple samples
        unique_speakers, counts = np.unique(speaker_ids, return_counts=True)
        valid_speakers = unique_speakers[counts >= 2]
        
        if len(valid_speakers) == 0:
            break
        
        speaker = np.random.choice(valid_speakers)
        speaker_indices = np.where(speaker_ids == speaker)[0]
        
        if len(speaker_indices) >= 2:
            idx1, idx2 = np.random.choice(speaker_indices, size=2, replace=False)
            trials.append((idx1, idx2, 1))
    
    # Generate negative pairs (different speakers)
    for _ in range(num_trials):
        idx1, idx2 = np.random.choice(num_samples, size=2, replace=False)
        
        if speaker_ids[idx1] != speaker_ids[idx2]:
            trials.append((idx1, idx2, 0))
    
    return trials


def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # Find threshold where FPR = FNR
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    return eer, eer_threshold


def compute_metrics(embeddings, speaker_ids, num_trials=1000):
    """
    Compute verification metrics.

    Returns:
        dict with metrics
    """
    logger.info(f"Computing metrics on {len(embeddings)} samples")
    print(f"\nComputing metrics on {len(embeddings)} samples...")

    # Compute similarity matrix
    logger.debug("Computing similarity matrix")
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Generate trials
    logger.debug(f"Generating {num_trials} verification trials")
    trials = generate_trials(speaker_ids, num_trials)
    logger.info(f"Generated {len(trials)} trials")
    print(f"Generated {len(trials)} trials")

    # Extract scores and labels
    scores = []
    labels = []

    for idx1, idx2, is_same in trials:
        scores.append(similarity_matrix[idx1, idx2])
        labels.append(is_same)

    scores = np.array(scores)
    labels = np.array(labels)

    # Compute EER
    logger.debug("Computing Equal Error Rate (EER)")
    eer, threshold = compute_eer(scores, labels)

    # Compute accuracy at EER threshold
    predictions = (scores >= threshold).astype(int)
    accuracy = (predictions == labels).mean()

    # Compute average similarities
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    metrics = {
        'eer': eer,
        'threshold': threshold,
        'accuracy': accuracy,
        'positive_similarity_mean': positive_scores.mean(),
        'positive_similarity_std': positive_scores.std(),
        'negative_similarity_mean': negative_scores.mean(),
        'negative_similarity_std': negative_scores.std(),
        'num_trials': len(trials),
        'num_speakers': len(np.unique(speaker_ids)),
    }

    logger.info(f"Metrics computed - EER: {eer*100:.2f}%, Accuracy: {accuracy*100:.2f}%")

    return metrics, scores, labels


def plot_roc_curve(scores, labels, output_path):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc

    logger.debug("Plotting ROC curve")
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ROC curve to {output_path}")
    print(f"Saved ROC curve to {output_path}")


def plot_score_distribution(scores, labels, output_path):
    """Plot score distribution."""
    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]
    
    plt.figure(figsize=(10, 6))
    plt.hist(positive_scores, bins=50, alpha=0.5, label='Same speaker', color='green')
    plt.hist(negative_scores, bins=50, alpha=0.5, label='Different speaker', color='red')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved score distribution to {output_path}")


def main():
    """Main validation function."""
    args = parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configs
    print("Loading configurations...")
    model_config = load_config(args.model_config)
    
    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, model_config, args.device)
    
    # Create dataset
    # Create dataset from shards
    print(f"\nCreating dataset from shards...")

    shard_dir = args.shard_dir
    if not Path(shard_dir).exists():
        raise FileNotFoundError(f"Shard directory not found: {shard_dir}")

    dataset = TimbreEncoderDataset(
        shard_dir=shard_dir,
        split='val',
        lang=None,
        is_training=False,
    )

    print(f"Dataset: {len(dataset)} samples, {dataset.get_num_speakers()} speakers")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=TimbreEncoderDataset.collate_fn,
    )
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    embeddings, speaker_ids = extract_embeddings(model, dataloader, args.device)
    
    print(f"Extracted {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    # Compute metrics
    metrics, scores, labels = compute_metrics(
        embeddings,
        speaker_ids,
        num_trials=args.num_trials
    )
    
    # Print results
    print("\n" + "="*60)
    print("Validation Results")
    print("="*60)
    print(f"Shard dir: {args.shard_dir}")
    print(f"Number of speakers: {metrics['num_speakers']}")
    print(f"Number of trials: {metrics['num_trials']}")
    print(f"\nEqual Error Rate (EER): {metrics['eer']*100:.2f}%")
    print(f"Threshold at EER: {metrics['threshold']:.4f}")
    print(f"Accuracy at EER: {metrics['accuracy']*100:.2f}%")
    print(f"\nPositive pairs (same speaker):")
    print(f"  Mean similarity: {metrics['positive_similarity_mean']:.4f} ± {metrics['positive_similarity_std']:.4f}")
    print(f"\nNegative pairs (different speaker):")
    print(f"  Mean similarity: {metrics['negative_similarity_mean']:.4f} ± {metrics['negative_similarity_std']:.4f}")
    print("="*60)
    
    # Save results
    results_path = output_dir / 'timbre_val_results.yaml'
    with open(results_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"\nSaved results to {results_path}")
    
    # Plot ROC curve
    plot_roc_curve(scores, labels, output_dir / 'timbre_val_roc.png')
    
    # Plot score distribution
    plot_score_distribution(scores, labels, output_dir / 'timbre_val_scores.png')
    
    # Save embeddings (optional, for further analysis)
    embeddings_path = output_dir / 'timbre_val_embeddings.npz'
    np.savez(
        embeddings_path,
        embeddings=embeddings,
        speaker_ids=speaker_ids,
    )
    print(f"Saved embeddings to {embeddings_path}")
    
    print("\nValidation completed successfully!")


if __name__ == '__main__':
    main()