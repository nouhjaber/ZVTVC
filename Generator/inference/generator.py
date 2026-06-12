"""
Mel-Spectrogram Generator
Complete inference pipeline for generating mel-spectrograms from conditions
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from .sampler import get_sampler, ODESampler


class MelGenerator:
    """
    Mel-spectrogram generator using trained UNet model.

    Generates mel-spectrograms from content, prosody, and timbre features
    using conditional flow matching.
    """

    def __init__(
        self,
        model: nn.Module,
        sampler_method: str = 'euler',
        num_steps: int = 10,
        use_cfg: bool = False,
        cfg_scale: float = 1.5,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # Setup sampler
        self.sampler = get_sampler(sampler_method, num_steps)
        self.num_steps = num_steps

        # Classifier-free guidance
        self.use_cfg = use_cfg
        self.cfg_scale = cfg_scale

    @torch.no_grad()
    def generate(
        self,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor,
        num_frames: Optional[int] = None
    ) -> torch.Tensor:
        # Move to device
        content = content.to(self.device)
        prosody = prosody.to(self.device)
        timbre = timbre.to(self.device)

        # Determine shape
        batch_size = content.shape[0]
        if num_frames is None:
            num_frames = content.shape[-1]

        shape = (batch_size, 80, num_frames)

        # Generate with or without CFG
        if self.use_cfg and self.cfg_scale != 1.0:
            mel = self._generate_with_cfg(shape, content, prosody, timbre)
        else:
            mel = self.sampler.sample(
                model=self.model,
                shape=shape,
                content=content,
                prosody=prosody,
                timbre=timbre,
                device=self.device
            )

        return mel

    def _generate_with_cfg(
        self,
        shape: tuple,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate with classifier-free guidance.

        v_guided = v_uncond + s * (v_cond - v_uncond)
        """
        device = self.device
        batch_size = shape[0]

        # Initialize with noise
        x = torch.randn(shape, device=device)

        # Time step
        dt = 1.0 / self.num_steps

        # Prepare unconditional features (zeros)
        content_uncond = torch.zeros_like(content)
        prosody_uncond = torch.zeros_like(prosody)
        timbre_uncond = torch.zeros_like(timbre)

        # Solve ODE with guidance
        for i in range(self.num_steps):
            t = i / self.num_steps
            t_tensor = torch.full((batch_size,), t, device=device)

            # Conditional prediction
            v_cond = self.model(x, t_tensor, content, prosody, timbre)

            # Unconditional prediction
            v_uncond = self.model(x, t_tensor, content_uncond, prosody_uncond, timbre_uncond)

            # Apply guidance
            v_guided = v_uncond + self.cfg_scale * (v_cond - v_uncond)

            # Euler step with guided velocity
            x = x + dt * v_guided

        return x

    @torch.no_grad()
    def generate_batch(
        self,
        content_list: list,
        prosody_list: list,
        timbre_list: list
    ) -> list:
        results = []

        for content, prosody, timbre in zip(content_list, prosody_list, timbre_list):
            # Add batch dimension if needed
            if content.dim() == 2:
                content = content.unsqueeze(0)
            if prosody.dim() == 2:
                prosody = prosody.unsqueeze(0)
            if timbre.dim() == 1:
                timbre = timbre.unsqueeze(0)

            # Generate
            mel = self.generate(content, prosody, timbre)

            # Remove batch dimension
            mel = mel.squeeze(0)

            results.append(mel)

        return results

    def set_num_steps(self, num_steps: int):
        """Change number of sampling steps"""
        self.num_steps = num_steps
        self.sampler.num_steps = num_steps

    def set_cfg_scale(self, cfg_scale: float):
        """Change CFG guidance scale"""
        self.cfg_scale = cfg_scale

    def enable_cfg(self, enabled: bool = True):
        """Enable/disable classifier-free guidance"""
        self.use_cfg = enabled

    @property
    def estimated_latency_ms(self) -> float:
        # Rough estimate (T4 baseline)
        ms_per_step = 7.0

        if 'cuda' in str(self.device):
            # Try to detect GPU type (rough heuristic)
            try:
                gpu_name = torch.cuda.get_device_name(self.device)
                if 'V100' in gpu_name:
                    ms_per_step = 4.0
                elif 'A100' in gpu_name:
                    ms_per_step = 3.0
            except:
                pass

        total_latency = self.num_steps * ms_per_step

        # Add overhead
        overhead = 10.0  # ms
        total_latency += overhead

        return total_latency

    def __repr__(self):
        return (
            f"MelGenerator(\n"
            f"  sampler={self.sampler.__class__.__name__},\n"
            f"  num_steps={self.num_steps},\n"
            f"  use_cfg={self.use_cfg},\n"
            f"  cfg_scale={self.cfg_scale},\n"
            f"  device={self.device},\n"
            f"  estimated_latency={self.estimated_latency_ms:.1f}ms\n"
            f")"
        )


def load_generator(
    checkpoint_path: str,
    model_class: nn.Module,
    device: str = 'cuda',
    use_ema: bool = True,
    **generator_kwargs
) -> MelGenerator:
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model weights
    if use_ema and 'ema_state_dict' in checkpoint:
        # Load EMA weights
        ema_state = checkpoint['ema_state_dict']['shadow']
        model_class.load_state_dict(ema_state)
        print("Loaded EMA weights")
    else:
        # Load regular weights
        model_class.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")

    # Create generator
    generator = MelGenerator(
        model=model_class,
        device=device,
        **generator_kwargs
    )

    print(f"Loaded generator from iteration {checkpoint['iteration']}")
    print(generator)

    return generator
