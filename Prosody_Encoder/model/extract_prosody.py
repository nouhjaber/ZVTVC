"""
Prosody Extraction Script
Extract prosody features from audio files using trained model
"""
import os
import sys
import yaml
import argparse
import torch
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import hashlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.prosody_encoder import ProsodyEncoder
from training.dataset import gather_audio_files


def load_config(config_path: str) -> dict:
    """Load YAML configuration"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load and preprocess audio file."""
    ext = Path(path).suffix.lower()
    if ext in ('.wav', '.flac'):
        audio, orig_sr = sf.read(path, dtype='float32')
    else:
        audio, orig_sr = librosa.load(path, sr=None, mono=True)
        audio = audio.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1 if audio.shape[1] < audio.shape[0] else 0)

    # Resample if needed
    if orig_sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)

    # Normalize
    peak = np.abs(audio).max()
    if peak > 1e-8:
        audio = audio / peak

    return audio.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Extract Prosody Features")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, uses only explicit features if not provided)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input audio file or directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for extracted features"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="numpy",
        choices=["numpy", "torch", "both"],
        help="Output format"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    sample_rate = config["features"]["sample_rate"]

    print(f"\n{'='*60}")
    print(f"Prosody Feature Extraction")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # Create model
    print("Creating model...")
    model = ProsodyEncoder(
        sample_rate=sample_rate,
        hop_length=config["features"]["hop_length"],
        frame_rate=config["features"]["frame_rate"],
        explicit_dim=config["model"]["explicit_dim"],
        refined_dim=config["model"]["refined_dim"],
        f0_method=config["features"]["f0"].get("method", "crepe"),
        f0_fmin=config["features"]["f0"]["fmin"],
        f0_fmax=config["features"]["f0"]["fmax"],
        rhythm_window_size=config["features"]["rhythm"]["window_size"],
        use_refinement=args.checkpoint is not None,
        output_format=config["inference"].get("output_format", "refined"),
    )

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from step {checkpoint['global_step']}")

    model.to(device)
    model.eval()

    # Gather input files
    if os.path.isfile(args.input):
        audio_paths = [args.input]
    else:
        audio_paths = gather_audio_files(args.input)

    print(f"\nFound {len(audio_paths)} audio files")

    if len(audio_paths) == 0:
        print("ERROR: No audio files found!")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Extract features
    print("\nExtracting features...")
    with torch.no_grad():
        for audio_path in audio_paths:
            try:
                # Load audio
                audio = load_audio(audio_path, sample_rate=sample_rate)

                # Extract prosody
                prosody = model.inference(audio)

                # Convert to numpy if needed
                if isinstance(prosody, torch.Tensor):
                    prosody_np = prosody.cpu().numpy()
                else:
                    prosody_np = prosody

                # Save features (collision-resistant filename)
                p = os.path.abspath(audio_path)
                p = os.path.normpath(p).replace(os.sep, "/")
                key = hashlib.md5(p.encode("utf-8")).hexdigest()
                output_path = os.path.join(args.output, key)

                if args.format in ["numpy", "both"]:
                    np.save(output_path + ".npy", prosody_np)
                    print(f"Saved: {output_path}.npy")

                if args.format in ["torch", "both"]:
                    torch.save(prosody, output_path + ".pt")
                    print(f"Saved: {output_path}.pt")

            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue

    print(f"\nFeature extraction completed!")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()