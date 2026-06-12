"""
Energy Extractor
Extracts RMS energy from audio with log-transform and whitening
"""
from typing import Optional, Union, Tuple

import numpy as np
import torch
import librosa


ArrayLike = Union[np.ndarray, torch.Tensor]


class EnergyExtractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        n_fft: int = 1024,
        n_mels: int = 80,
        epsilon: float = 1e-5,
        log_transform: bool = True,
        whitening: bool = True,
    ):
        # Initialize Energy extractor
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.epsilon = epsilon
        self.log_transform = log_transform
        self.whitening = whitening

        # Statistics for whitening
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def _to_numpy(
        self,
        audio: ArrayLike,
    ) -> Tuple[np.ndarray, Optional[torch.device]]:
        """
        Convert input audio to numpy array.

        Returns:
            audio_np: numpy array [L]
            device: original device if input was a tensor, else None
        """
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32), None
        elif isinstance(audio, torch.Tensor):
            device = audio.device
            audio_np = audio.detach().cpu().numpy().astype(np.float32)
            return audio_np, device
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

    def _from_numpy(
        self,
        x: np.ndarray,
        device: Optional[torch.device],
    ) -> ArrayLike:
        """
        Convert numpy result back to original type.

        If device is None -> return numpy array.
        If device is a torch device -> return tensor on that device.
        """
        x = x.astype(np.float32)
        if device is None:
            return x
        else:
            return torch.from_numpy(x).to(device)

    def extract(self, audio: ArrayLike) -> ArrayLike:
        # Extract RMS energy from audio
        audio_np, device = self._to_numpy(audio)

        # Use librosa.feature.rms — matches preprocess_unified_shards.py
        energy_np = librosa.feature.rms(
            y=audio_np,
            hop_length=self.hop_length,
            frame_length=self.hop_length * 4,
        )[0]

        return self._from_numpy(energy_np, device)

    def normalize(self, energy: ArrayLike) -> ArrayLike:
        """
        Normalize energy with log-transform and whitening.
        Whitens over ALL frames to match preprocess_unified_shards.py.
        """
        energy_np, device = self._to_numpy(energy)

        energy_norm = energy_np.copy()

        if self.log_transform:
            energy_norm = np.log(energy_norm + self.epsilon)

        if self.whitening:
            # Whiten over ALL frames — matches preprocessor
            mean = float(np.mean(energy_norm))
            std = float(np.std(energy_norm) + self.epsilon)
            energy_norm = (energy_norm - mean) / std

        return self._from_numpy(energy_norm, device)

    def denormalize(self, energy_norm: ArrayLike) -> ArrayLike:
        """
        Denormalize energy back to original scale

        Args:
            energy_norm: Normalized energy [T] (np.ndarray or torch.Tensor)

        Returns:
            energy: Energy in original scale [T] (same type as input)
        """
        energy_norm_np, device = self._to_numpy(energy_norm)

        energy_np = energy_norm_np.copy()

        if self.whitening and self.mean is not None and self.std is not None:
            energy_np = energy_np * self.std + self.mean

        if self.log_transform:
            energy_np = np.exp(energy_np) - self.epsilon

        return self._from_numpy(energy_np, device)

    def __call__(self, audio: ArrayLike) -> ArrayLike:
        """Extract and normalize energy"""
        energy = self.extract(audio)
        energy_norm = self.normalize(energy)
        return energy_norm