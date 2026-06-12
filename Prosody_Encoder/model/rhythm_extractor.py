"""
Rhythm Extractor (v1.1 feature)
Computes local voicing rate to capture rhythm patterns
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple


ArrayLike = Union[np.ndarray, torch.Tensor]


class RhythmExtractor:
    def __init__(
        self,
        window_size: int = 11,
        method: str = "local_voicing_rate",
    ):
        # Initialize Rhythm extractor
        self.window_size = window_size
        self.method = method

        # Ensure window size is odd for symmetric padding
        if self.window_size % 2 == 0:
            self.window_size += 1

    def _to_tensor(
        self,
        x: ArrayLike,
    ) -> Tuple[torch.Tensor, Optional[torch.device], bool]:
        """
        Convert input to torch.Tensor.
        """
        if isinstance(x, np.ndarray):
            x_torch = torch.from_numpy(x).float()
            return x_torch, None, True
        elif isinstance(x, torch.Tensor):
            return x.float(), x.device, False
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

    def _from_tensor(
        self,
        x_torch: torch.Tensor,
        device: Optional[torch.device],
        is_numpy: bool,
    ) -> ArrayLike:
        """
        Convert tensor result back to original type.

        If original was numpy -> return np.ndarray.
        If original was tensor -> return tensor on same device.
        """
        if is_numpy:
            return x_torch.detach().cpu().numpy()
        else:
            if device is None:
                return x_torch
            return x_torch.to(device)

    def extract(self, voicing: ArrayLike) -> ArrayLike:
        """
        Extract rhythm feature from voicing contour
        """
        if self.method == "local_voicing_rate":
            return self._local_voicing_rate(voicing)
        else:
            raise ValueError(f"Unknown rhythm extraction method: {self.method}")

    def _local_voicing_rate(self, voicing: ArrayLike) -> ArrayLike:
        """
        Compute local voicing rate using sliding window

        High rhythm value = more voiced = slower speech (vowels, sustained sounds)
        Low rhythm value = less voiced = faster speech (consonants, transitions)
        """
        voicing_torch, device, is_numpy = self._to_tensor(voicing)

        # [1, 1, T]
        voicing_torch = voicing_torch.unsqueeze(0).unsqueeze(0)

        kernel = torch.ones(1, 1, self.window_size, device=voicing_torch.device) / self.window_size

        padding = self.window_size // 2
        rhythm_torch = F.conv1d(voicing_torch, kernel, padding=padding)

        # [T]
        rhythm_torch = rhythm_torch.squeeze(0).squeeze(0)

        return self._from_tensor(rhythm_torch, device, is_numpy)

    def extract_temporal_derivatives(self, rhythm: ArrayLike) -> ArrayLike:
        """
        Compute first-order temporal derivative (optional feature)
        Captures rate of change in rhythm patterns

        """
        rhythm_torch, device, is_numpy = self._to_tensor(rhythm)

        # Use torch gradient along last dimension
        rhythm_derivative = torch.gradient(rhythm_torch)[0]

        return self._from_tensor(rhythm_derivative, device, is_numpy)

    def __call__(self, voicing: ArrayLike) -> ArrayLike:
        """Extract rhythm feature"""
        return self.extract(voicing)