"""
Visualization Utilities for Prosody Features
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


def plot_prosody_features(
    features: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 320,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    # Ensure shape is [4, T]
    if features.shape[0] != 4:
        features = features.T

    # Compute time axis
    frame_rate = sample_rate / hop_length
    time = np.arange(features.shape[1]) / frame_rate

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    # Plot F0
    axes[0].plot(time, features[0], linewidth=1)
    axes[0].set_ylabel("F0 (normalized)")
    axes[0].set_title("Prosody Features")
    axes[0].grid(True, alpha=0.3)

    # Plot Energy
    axes[1].plot(time, features[1], linewidth=1, color="orange")
    axes[1].set_ylabel("Energy (normalized)")
    axes[1].grid(True, alpha=0.3)

    # Plot Voicing
    axes[2].plot(time, features[2], linewidth=1, color="green")
    axes[2].set_ylabel("Voicing")
    axes[2].set_ylim([-0.1, 1.1])
    axes[2].grid(True, alpha=0.3)

    # Plot Rhythm
    axes[3].plot(time, features[3], linewidth=1, color="red")
    axes[3].set_ylabel("Rhythm")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_feature_comparison(
    pred_features: np.ndarray,
    target_features: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 320,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
):
    # Ensure shape is [4, T]
    if pred_features.shape[0] != 4:
        pred_features = pred_features.T
    if target_features.shape[0] != 4:
        target_features = target_features.T

    # Compute time axis
    frame_rate = sample_rate / hop_length
    time = np.arange(pred_features.shape[1]) / frame_rate

    # Feature names
    feature_names = ["F0", "Energy", "Voicing", "Rhythm"]
    colors = ["blue", "orange", "green", "red"]

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    for i, (name, color) in enumerate(zip(feature_names, colors)):
        # Plot target
        axes[i].plot(time, target_features[i],
                    linewidth=1.5, alpha=0.7,
                    label="Target", color=color)

        # Plot prediction
        axes[i].plot(time, pred_features[i],
                    linewidth=1.5, alpha=0.7,
                    label="Predicted", color=color,
                    linestyle="--")

        axes[i].set_ylabel(name)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.3)

    axes[0].set_title("Predicted vs Target Prosody Features")
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_training_curves(
    train_losses: list,
    val_losses: Optional[list] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
):
    fig, ax = plt.subplots(figsize=figsize)

    # Extract steps and losses
    train_steps = [loss_dict["step"] for loss_dict in train_losses]
    train_total = [loss_dict["total"] for loss_dict in train_losses]

    # Plot training loss
    ax.plot(train_steps, train_total, label="Training Loss", linewidth=2)

    # Plot validation loss if available
    if val_losses:
        val_steps = [loss_dict["step"] for loss_dict in val_losses]
        val_total = [loss_dict["total"] for loss_dict in val_losses]
        ax.plot(val_steps, val_total, label="Validation Loss",
               linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title("Training Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_refined_features(
    refined: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 320,
    max_channels: int = 8,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
):
    # Ensure shape is [C, T]
    if refined.shape[0] > refined.shape[1]:
        refined = refined.T

    # Limit channels
    num_channels = min(refined.shape[0], max_channels)

    # Compute time axis
    frame_rate = sample_rate / hop_length
    time = np.arange(refined.shape[1]) / frame_rate

    # Create subplots
    fig, axes = plt.subplots(num_channels, 1, figsize=figsize, sharex=True)

    if num_channels == 1:
        axes = [axes]

    for i in range(num_channels):
        axes[i].plot(time, refined[i], linewidth=1)
        axes[i].set_ylabel(f"Ch {i+1}")
        axes[i].grid(True, alpha=0.3)

    axes[0].set_title(f"Refined Features (showing {num_channels}/{refined.shape[0]} channels)")
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def plot_feature_heatmap(
    features: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 320,
    feature_names: Optional[list] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
):
    # Ensure shape is [C, T]
    if features.shape[0] > features.shape[1]:
        features = features.T

    # Compute time axis
    frame_rate = sample_rate / hop_length
    time = np.arange(features.shape[1]) / frame_rate

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(
        features,
        aspect="auto",
        origin="lower",
        extent=[0, time[-1], 0, features.shape[0]],
        cmap="viridis",
    )

    # Set labels
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Feature Channel")
    ax.set_title("Feature Heatmap")

    # Set y-ticks
    if feature_names:
        ax.set_yticks(np.arange(len(feature_names)) + 0.5)
        ax.set_yticklabels(feature_names)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Value")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    return fig


def test_visualization():
    """Test visualization functions"""
    print("Testing Visualization...")

    # Create dummy features
    features = np.random.randn(4, 200)
    refined = np.random.randn(32, 200)

    # Test plotting
    plot_prosody_features(features)
    plt.show()

    print("\nTest passed!")


if __name__ == "__main__":
    test_visualization()
