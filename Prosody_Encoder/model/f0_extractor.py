"""
F0 (Fundamental Frequency) Extractor
Supports both PYIN and CREPE methods with whitening normalization
"""
import torch
import numpy as np
import librosa
from typing import Optional, Tuple


class F0Extractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        method: str = "pyin",
        fmin: float = 50.0,
        fmax: float = 600.0,
        log_transform: bool = True,
        whitening: bool = True,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        self.log_transform = log_transform
        self.whitening = whitening

        # Statistics for whitening (computed per utterance or dataset)
        self.mean = None
        self.std = None

    def extract(
        self,
        audio: np.ndarray,
        interpolate: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.method == "pyin":
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                sr=self.sample_rate,
                fmin=self.fmin,
                fmax=self.fmax,
                hop_length=self.hop_length,
                frame_length=self.hop_length * 4,
                fill_na=None
            )
            # For PYIN, voiced_mask is based on NaN values
            voiced_mask = ~np.isnan(f0)
    
        elif self.method == "crepe":
            # CREPE requires torchcrepe package
            try:
                import torchcrepe
                
                # CREPE expects audio in [-1, 1]
                max_abs = np.max(np.abs(audio))
                audio_norm = audio / max_abs if max_abs > 0 else audio
                audio_torch = torch.from_numpy(audio_norm).float().unsqueeze(0)
    
                # Extract F0
                f0_torch, confidence = torchcrepe.predict(
                    audio_torch,
                    self.sample_rate,
                    self.hop_length,
                    self.fmin,
                    self.fmax,
                    model='tiny',
                    return_periodicity=True,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
    
                f0 = f0_torch.squeeze().cpu().numpy()
                confidence_np = confidence.squeeze().cpu().numpy()
                
                # For CREPE, voiced_mask is based on confidence threshold
                voiced_mask = confidence_np > 0.5
                
                # Mark low-confidence regions as NaN for consistent handling
                f0[~voiced_mask] = np.nan
    
            except ImportError:
                raise ImportError(
                    "CREPE method requires 'torchcrepe' package. "
                    "Install with: pip install torchcrepe"
                )
        else:
            raise ValueError(f"Unknown F0 extraction method: {self.method}")
    
        # Handle unvoiced regions
        if interpolate and np.any(voiced_mask):
            # Linear interpolation for unvoiced regions
            f0 = self._interpolate_f0(f0, voiced_mask)
        else:
            # Fill NaN with median of voiced regions
            if np.any(voiced_mask):
                f0[~voiced_mask] = np.median(f0[voiced_mask])
            else:  
                f0 = np.full_like(f0, 100.0)  # Default F0 if all unvoiced
    
        return f0, voiced_mask.astype(np.float32)


    def _interpolate_f0(self, f0: np.ndarray, voiced_mask: np.ndarray) -> np.ndarray:
        """Interpolate F0 in unvoiced regions"""
        f0_interp = f0.copy()

        if np.sum(voiced_mask) < 2:
            # Not enough voiced frames for interpolation
            return f0_interp

        # Get indices of voiced frames
        voiced_indices = np.where(voiced_mask)[0]

        # Interpolate unvoiced frames
        all_indices = np.arange(len(f0))
        f0_interp = np.interp(
            all_indices,
            voiced_indices,
            f0[voiced_mask],
        )

        return f0_interp

    def normalize(
        self,
        f0: np.ndarray,
        voiced_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Normalize F0 with log-transform and whitening.
        Whitens over ALL frames to match preprocess_unified_shards.py.
        """
        f0_norm = f0.copy()

        # Log transform
        if self.log_transform:
            f0_norm = np.log(f0_norm + 1e-8)

        # Whitening over ALL frames — matches preprocessor
        if self.whitening:
            mean = np.mean(f0_norm)
            std = np.std(f0_norm) + 1e-8
            f0_norm = (f0_norm - mean) / std

        return f0_norm

    def denormalize(self, f0_norm: np.ndarray) -> np.ndarray:
        """
        Denormalize F0 back to Hz
        """
        f0 = f0_norm.copy()

        # Reverse whitening
        if self.whitening and self.mean is not None and self.std is not None:
            f0 = f0 * self.std + self.mean

        # Reverse log transform
        if self.log_transform:
            f0 = np.exp(f0)

        return f0

    def __call__(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and normalize F0"""
        f0, voiced_mask = self.extract(audio)
        f0_norm = self.normalize(f0, voiced_mask)
        return f0_norm, voiced_mask