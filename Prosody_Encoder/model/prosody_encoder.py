"""
Prosody Encoder v1.1
Complete prosody encoder that extracts explicit features and refines them
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union

from model.f0_extractor import F0Extractor
from model.energy_extractor import EnergyExtractor
from model.voicing_detector import VoicingDetector
from model.rhythm_extractor import RhythmExtractor
from model.refinement_network import RefinementNetwork


class ProsodyEncoder(nn.Module):
    """
    Complete Prosody Encoder v1.1

    Pipeline:
    1. Extract explicit features: [f0, energy, voicing, rhythm]
    2. Normalize features (whitening)
    3. Refine features through light CNN
    4. Output: concatenated [explicit + refined] or just refined
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        frame_rate: int = 50,
        explicit_dim: int = 4,
        refined_dim: int = 32,
        f0_method: str = "crepe",
        f0_fmin: float = 50.0,
        f0_fmax: float = 600.0,
        rhythm_window_size: int = 11,
        use_refinement: bool = True,
        use_residual: bool = True,
        use_reconstruction_heads: bool = True,
        output_format: str = "refined",  # 'explicit', 'refined', 'combined'
    ):
        # Initialize Prosody Encoder
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_rate = frame_rate
        self.explicit_dim = explicit_dim
        self.refined_dim = refined_dim
        self.output_format = output_format
        self.use_refinement = use_refinement

        # Feature extractors (not trainable)
        self.f0_extractor = F0Extractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            method=f0_method,
            fmin=f0_fmin,
            fmax=f0_fmax,
            log_transform=True,
            whitening=True,
        )

        self.energy_extractor = EnergyExtractor(
            sample_rate=sample_rate,
            hop_length=hop_length,
            log_transform=True,
            whitening=True,
        )

        self.voicing_detector = VoicingDetector(
            sample_rate=sample_rate,
            hop_length=hop_length,
            method="from_f0",
        )

        self.rhythm_extractor = RhythmExtractor(
            window_size=rhythm_window_size,
            method="local_voicing_rate",
        )

        # Refinement network (trainable)
        if self.use_refinement:
            self.refinement_network = RefinementNetwork(
                explicit_dim=explicit_dim,
                refined_dim=refined_dim,
                use_residual=use_residual,
                use_reconstruction_heads=use_reconstruction_heads,
            )

    def extract_explicit_features(
        self,
        audio: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        # Extract explicit prosody features from audio
        # Handle batched or single audio
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        is_batched = audio.ndim == 2
        if not is_batched:
            audio = audio[np.newaxis, :]

        batch_size = audio.shape[0]
        features_list = []

        for i in range(batch_size):
            audio_i = audio[i]

            # Extract F0 (normalized)
            f0_norm, voiced_mask = self.f0_extractor(audio_i)

            # Extract energy (normalized)
            energy_norm = self.energy_extractor(audio_i)

            # Extract voicing
            voicing = self.voicing_detector(f0=f0_norm, voiced_flag=voiced_mask)

            # Extract rhythm
            rhythm = self.rhythm_extractor(voicing)

            # Ensure all features have the same length
            min_length = min(len(f0_norm), len(energy_norm), len(voicing), len(rhythm))
            f0_norm = f0_norm[:min_length]
            energy_norm = energy_norm[:min_length]
            voicing = voicing[:min_length]
            rhythm = rhythm[:min_length]

            # Stack features [4, T]
            features = np.stack([f0_norm, energy_norm, voicing, rhythm], axis=0)
            features_list.append(features)

        # Stack batch [B, 4, T]
        explicit_features = np.stack(features_list, axis=0)

        # Convert to torch tensor
        explicit_features = torch.from_numpy(explicit_features).float()

        if not is_batched:
            explicit_features = explicit_features.squeeze(0)

        return explicit_features

    def forward(
        self,
        audio: Optional[Union[np.ndarray, torch.Tensor]] = None,
        explicit_features: Optional[torch.Tensor] = None,
        return_reconstructions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        # Extract explicit features if not provided
        if explicit_features is None:
            if audio is None:
                raise ValueError("Either audio or explicit_features must be provided")
            explicit_features = self.extract_explicit_features(audio)

        # Ensure correct device
        if isinstance(explicit_features, torch.Tensor):
            if self.use_refinement:
                device = next(self.refinement_network.parameters()).device
                explicit_features = explicit_features.to(device)

        # Ensure batch dimension [B, 4, T]
        if explicit_features.dim() == 2:
            explicit_features = explicit_features.unsqueeze(0)
            was_unbatched = True
        else:
            was_unbatched = False

        # Refine features
        refined = None
        reconstructions = None

        if self.use_refinement:
            refined, reconstructions = self.refinement_network(
                explicit_features,
                return_reconstructions=return_reconstructions
            )

        # Format output
        if self.output_format == "explicit":
            output = explicit_features
        elif self.output_format == "refined":
            output = refined if refined is not None else explicit_features
        elif self.output_format == "combined":
            if refined is not None:
                output = torch.cat([explicit_features, refined], dim=1)  # [B, 36, T]
            else:
                output = explicit_features
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")

        # Remove batch dimension if input was unbatched
        if was_unbatched:
            output = output.squeeze(0)
            explicit_features = explicit_features.squeeze(0)

        return output, explicit_features, reconstructions

    def inference(
        self,
        audio: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            output, _, _ = self.forward(audio, return_reconstructions=False)
        return output

    def get_output_dim(self) -> int:
        """Get output dimension based on format"""
        if self.output_format == "explicit":
            return self.explicit_dim
        elif self.output_format == "refined":
            return self.refined_dim
        elif self.output_format == "combined":
            return self.explicit_dim + self.refined_dim
        else:
            raise ValueError(f"Unknown output format: {self.output_format}")


def test_prosody_encoder():
    """Test the prosody encoder"""
    print("Testing Prosody Encoder v1.1...")

    # Create model
    model = ProsodyEncoder(
        sample_rate=16000,
        hop_length=320,
        frame_rate=50,
        output_format="refined",
    )

    print(f"Output dimension: {model.get_output_dim()}")

    # Create dummy audio (3 seconds)
    audio = np.random.randn(16000 * 3).astype(np.float32)

    # Test inference
    output = model.inference(audio)
    print(f"Audio shape: {audio.shape}")
    print(f"Output shape: {output.shape}")

    # Test with batch
    audio_batch = np.random.randn(4, 16000 * 2).astype(np.float32)
    output_batch = model.inference(audio_batch)
    print(f"\nBatch audio shape: {audio_batch.shape}")
    print(f"Batch output shape: {output_batch.shape}")

    # Test training mode with reconstructions
    model.train()
    output, explicit, reconstructions = model.forward(
        audio_batch,
        return_reconstructions=True
    )
    print(f"\nTraining mode:")
    print(f"Output shape: {output.shape}")
    print(f"Explicit shape: {explicit.shape}")
    print(f"Reconstructions: {list(reconstructions.keys())}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_prosody_encoder()