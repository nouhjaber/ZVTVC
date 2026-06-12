"""
Voicing Detector
Detects voiced/unvoiced frames from F0 or audio
"""

from typing import Optional, Tuple, Union

import numpy as np
import torch
import librosa


ArrayLike = Union[np.ndarray, torch.Tensor]


class VoicingDetector:
    def __init__(
        self,
        sample_rate: int = 16000,
        hop_length: int = 320,
        threshold: float = 0.5,
        method: str = "from_f0",  # 'from_f0' or 'from_audio'
    ):
        # Initialize Voicing detector
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.threshold = threshold
        self.method = method

    def _to_numpy(
        self,
        x: ArrayLike,
    ) -> Tuple[np.ndarray, Optional[torch.device]]:
        """Convert to numpy, keeping original device if x is a tensor."""
        if isinstance(x, np.ndarray):
            return x, None
        elif isinstance(x, torch.Tensor):
            device = x.device
            x_np = x.detach().cpu().numpy()
            return x_np, device
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    def _from_numpy(
        self,
        x_np: np.ndarray,
        device: Optional[torch.device],
    ) -> ArrayLike:
        """Convert numpy result back to original type (np or tensor)."""
        x_np = x_np.astype(np.float32)
        if device is None:
            return x_np
        else:
            return torch.from_numpy(x_np).to(device)

    def detect_from_f0(
        self,
        f0: ArrayLike,
        voiced_flag: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        # Detect voicing from F0 contour
        if voiced_flag is not None:
            vf_np, device = self._to_numpy(voiced_flag)
            return self._from_numpy(vf_np.astype(np.float32), device)

        f0_np, device = self._to_numpy(f0)

        voicing_np = np.logical_and(~np.isnan(f0_np), f0_np > 0).astype(np.float32)

        return self._from_numpy(voicing_np, device)

    def detect_from_audio(self, audio: ArrayLike) -> ArrayLike:
        """
        Detect voicing directly from audio using spectral features
        """
        audio_np, device = self._to_numpy(audio)

        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio_np,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=2048,
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            audio_np,
            frame_length=self.hop_length * 4,
            hop_length=self.hop_length,
        )[0]

        centroid_norm = (spectral_centroid - np.mean(spectral_centroid)) / (
            np.std(spectral_centroid) + 1e-8
        )
        zcr_norm = (zcr - np.mean(zcr)) / (np.std(zcr) + 1e-8)

        voicing_score = centroid_norm - zcr_norm
        voicing_np = (voicing_score > self.threshold).astype(np.float32)

        return self._from_numpy(voicing_np, device)

    def __call__(
        self,
        audio: Optional[ArrayLike] = None,
        f0: Optional[ArrayLike] = None,
        voiced_flag: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Detect voicing from audio or F0."""
        if self.method == "from_f0" and f0 is not None:
            return self.detect_from_f0(f0, voiced_flag)
        elif self.method == "from_audio" and audio is not None:
            return self.detect_from_audio(audio)
        else:
            raise ValueError("Invalid method or missing input")