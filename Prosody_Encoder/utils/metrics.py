"""
Metrics for Prosody Encoder Evaluation
"""
import numpy as np
import torch
from scipy.stats import pearsonr
from typing import Dict, Tuple


def compute_f0_pcc(
    pred_f0: np.ndarray,
    target_f0: np.ndarray,
    voiced_mask: np.ndarray = None,
) -> float:
    # Compute F0 Pearson Correlation Coefficient
    if voiced_mask is not None:
        # Compute only on voiced regions
        pred_f0 = pred_f0[voiced_mask > 0.5]
        target_f0 = target_f0[voiced_mask > 0.5]

    if len(pred_f0) < 2:
        return 0.0

    # Guard: pearsonr returns NaN for constant arrays
    if np.std(pred_f0) < 1e-8 or np.std(target_f0) < 1e-8:
        return 0.0

    pcc, _ = pearsonr(pred_f0, target_f0)

    return float(pcc) if not np.isnan(pcc) else 0.0


def compute_f0_rmse(
    pred_f0: np.ndarray,
    target_f0: np.ndarray,
    voiced_mask: np.ndarray = None,
) -> float:
    """
    Compute F0 RMSE in normalized (whitened) space.

    NOTE: Features are whitened (zero-mean, unit-variance), not raw log-Hz.
    Converting to cents would require denormalizing first, which needs the
    per-utterance mean/std from the F0Extractor. We report RMSE directly
    in normalized space — lower is better, scale is comparable across utterances.
    """
    if voiced_mask is not None:
        pred_f0 = pred_f0[voiced_mask > 0.5]
        target_f0 = target_f0[voiced_mask > 0.5]

    if len(pred_f0) == 0:
        return 0.0

    rmse = np.sqrt(np.mean((pred_f0 - target_f0) ** 2))

    return rmse


def compute_energy_correlation(
    pred_energy: np.ndarray,
    target_energy: np.ndarray,
) -> float:
    if len(pred_energy) < 2:
        return 0.0

    if np.std(pred_energy) < 1e-8 or np.std(target_energy) < 1e-8:
        return 0.0

    corr, _ = pearsonr(pred_energy, target_energy)

    return float(corr) if not np.isnan(corr) else 0.0


def compute_energy_rmse(
    pred_energy: np.ndarray,
    target_energy: np.ndarray,
) -> float:
    """
    Compute energy RMSE in normalized (whitened) space.

    NOTE: Features are whitened, not raw log-energy. Converting to dB would
    require denormalizing first. We report RMSE directly in normalized space.
    """
    rmse = np.sqrt(np.mean((pred_energy - target_energy) ** 2))

    return rmse


def compute_voicing_decision_error(
    pred_voicing: np.ndarray,
    target_voicing: np.ndarray,
) -> float:
    # Threshold predictions
    pred_binary = (pred_voicing > 0.5).astype(float)
    target_binary = (target_voicing > 0.5).astype(float)

    # Compute error rate
    errors = np.abs(pred_binary - target_binary)
    error_rate = np.mean(errors)

    return error_rate


def compute_rhythm_correlation(
    pred_rhythm: np.ndarray,
    target_rhythm: np.ndarray,
) -> float:
    if len(pred_rhythm) < 2:
        return 0.0

    if np.std(pred_rhythm) < 1e-8 or np.std(target_rhythm) < 1e-8:
        return 0.0

    corr, _ = pearsonr(pred_rhythm, target_rhythm)

    return float(corr) if not np.isnan(corr) else 0.0


def compute_all_metrics(
    pred_features: np.ndarray,
    target_features: np.ndarray,
) -> Dict[str, float]:
    metrics = {}

    # Extract individual features
    pred_f0 = pred_features[0]
    pred_energy = pred_features[1]
    pred_voicing = pred_features[2]
    pred_rhythm = pred_features[3]

    target_f0 = target_features[0]
    target_energy = target_features[1]
    target_voicing = target_features[2]
    target_rhythm = target_features[3]

    # F0 metrics
    metrics["f0_pcc"] = compute_f0_pcc(pred_f0, target_f0, target_voicing)
    metrics["f0_rmse"] = compute_f0_rmse(pred_f0, target_f0, target_voicing)

    # Energy metrics
    metrics["energy_correlation"] = compute_energy_correlation(pred_energy, target_energy)
    metrics["energy_rmse"] = compute_energy_rmse(pred_energy, target_energy)

    # Voicing metrics
    metrics["voicing_error_rate"] = compute_voicing_decision_error(pred_voicing, target_voicing)

    # Rhythm metrics
    metrics["rhythm_correlation"] = compute_rhythm_correlation(pred_rhythm, target_rhythm)

    return metrics


class ProsodyMetrics:
    """Prosody metrics tracker"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.metrics = {
            "f0_pcc": [],
            "f0_rmse": [],
            "energy_correlation": [],
            "energy_rmse": [],
            "voicing_error_rate": [],
            "rhythm_correlation": [],
        }

    def update(self, pred_features: torch.Tensor, target_features: torch.Tensor):
        """
        Update metrics with batch

        Args:
            pred_features: Predicted features [B, 4, T]
            target_features: Target features [B, 4, T]
        """
        # Convert to numpy
        if isinstance(pred_features, torch.Tensor):
            pred_features = pred_features.cpu().numpy()
        if isinstance(target_features, torch.Tensor):
            target_features = target_features.cpu().numpy()

        # Compute metrics for each item in batch
        batch_size = pred_features.shape[0]
        for i in range(batch_size):
            item_metrics = compute_all_metrics(
                pred_features[i],
                target_features[i]
            )

            for key, value in item_metrics.items():
                self.metrics[key].append(value)

    def compute(self) -> Dict[str, float]:
        """
        Compute average metrics

        Returns:
            avg_metrics: Dictionary of average metrics
        """
        avg_metrics = {}

        for key, values in self.metrics.items():
            if len(values) > 0:
                avg_metrics[key] = np.mean(values)
            else:
                avg_metrics[key] = 0.0

        return avg_metrics


def test_metrics():
    """Test metrics"""
    print("Testing Prosody Metrics...")

    # Create dummy data
    target_features = np.random.randn(4, 100)
    pred_features = target_features + np.random.randn(4, 100) * 0.1

    # Compute metrics
    metrics = compute_all_metrics(pred_features, target_features)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test metrics tracker
    print("\nTesting Metrics Tracker...")
    tracker = ProsodyMetrics()

    # Add batch
    pred_batch = torch.randn(8, 4, 100)
    target_batch = pred_batch + torch.randn(8, 4, 100) * 0.1

    tracker.update(pred_batch, target_batch)

    avg_metrics = tracker.compute()
    print("Average metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_metrics()