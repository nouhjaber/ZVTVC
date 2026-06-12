import torch
import torch.nn as nn
import copy
from typing import Optional


class EMATeacher(nn.Module):
    """
    Exponential Moving Average (EMA) Teacher for self-distillation.

    Formula: θ_ema = α · θ_ema + (1-α) · θ_student

    where α (alpha) is the decay rate:
    - Stage 0: α = 0.99 (fast tracking, model changing quickly)
    - Stage 1: α = 0.995 (moderate tracking)
    - Stage 2: α = 0.999 (slow, stable tracking)
    """

    def __init__(
        self,
        student_encoder: nn.Module,
        alpha: float = 0.999,
        update_after_step: int = 0,
        device: str = 'cuda'
    ):
        super().__init__()

        # Create deep copy of student
        self.encoder = copy.deepcopy(student_encoder)
        self.encoder = self.encoder.to(device)
        self.encoder.eval()

        # Freeze EMA model (no gradient computation)
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.alpha = alpha
        self.update_after_step = update_after_step
        self.num_updates = 0
        self.enabled = False
        self.device = device

    @torch.no_grad()
    def update(self, student_encoder: nn.Module):
        if not self.enabled:
            return

        self.num_updates += 1

        # Skip updates until update_after_step
        if self.num_updates < self.update_after_step:
            return

        # Update each parameter: ema = α * ema + (1-α) * student
        for ema_param, student_param in zip(
            self.encoder.parameters(),
            student_encoder.parameters()
        ):
            ema_param.data.mul_(self.alpha).add_(
                student_param.data.to(self.device),
                alpha=1.0 - self.alpha
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x.to(self.device))

    def enable(self):
        self.enabled = True
        print(f" EMA teacher enabled (α={self.alpha})")

    def disable(self):
        self.enabled = False
        print(" EMA teacher disabled")

    def set_alpha(self, alpha: float):
        old_alpha = self.alpha
        self.alpha = alpha
        print(f"[OK] EMA alpha updated: {old_alpha:.4f} -> {alpha:.4f}")

    def reset(self, student_encoder: nn.Module):
        print("[RESET] Resetting EMA teacher to match student...")
        self.encoder = copy.deepcopy(student_encoder)
        self.encoder = self.encoder.to(self.device)
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False

        self.num_updates = 0
        print("EMA reset complete")

    def get_alpha(self) -> float:
        return self.alpha

    def get_num_updates(self) -> int:
        return self.num_updates

    def is_enabled(self) -> bool:
        return self.enabled

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Override state_dict to match nn.Module signature"""
        return {
            'encoder': self.encoder.state_dict(),
            'alpha': self.alpha,
            'num_updates': self.num_updates,
            'enabled': self.enabled,
            'update_after_step': self.update_after_step
        }

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict including metadata"""
        self.encoder.load_state_dict(state_dict['encoder'])
        self.alpha = state_dict.get('alpha', 0.999)
        self.num_updates = state_dict.get('num_updates', 0)
        self.enabled = state_dict.get('enabled', False)
        self.update_after_step = state_dict.get('update_after_step', 0)


class EMAScheduler:
    """
    Scheduler for EMA alpha values across training stages.

    Automatically adjusts alpha based on training stage:
    - Stage 0: Fast tracking (α=0.99)
    - Stage 1: Moderate tracking (α=0.995)
    - Stage 2: Slow tracking (α=0.999)
    """

    def __init__(self, ema_teacher: EMATeacher):
        """
        Args:
            ema_teacher: EMA teacher to schedule
        """
        self.ema_teacher = ema_teacher

        # Stage-specific alpha values
        self.alpha_schedule = {
            0: 0.99,    # Stage 0: Fast
            1: 0.995,   # Stage 1: Moderate
            2: 0.999    # Stage 2: Slow
        }

        # Enable iteration schedule
        self.enable_schedule = {
            0: 10000,  # Enable after 10k iterations in stage 0
            1: 0,      # Enable immediately in stage 1
            2: 0       # Enable immediately in stage 2
        }

    def set_stage(self, stage: int, iteration: int = 0):
        """
        Update EMA for new training stage.

        Args:
            stage: Training stage (0, 1, or 2)
            iteration: Current iteration within stage
        """
        # Get alpha for this stage
        alpha = self.alpha_schedule.get(stage, 0.999)
        self.ema_teacher.set_alpha(alpha)

        # Check if should be enabled
        enable_after = self.enable_schedule.get(stage, 0)

        if iteration >= enable_after:
            if not self.ema_teacher.is_enabled():
                self.ema_teacher.enable()
        else:
            if self.ema_teacher.is_enabled():
                self.ema_teacher.disable()

    def step(self, iteration: int, stage: int):
        """
        Update EMA teacher for current iteration.

        Args:
            iteration: Current iteration within stage
            stage: Current stage
        """
        # Check if should enable based on iteration
        enable_after = self.enable_schedule.get(stage, 0)

        if iteration == enable_after and not self.ema_teacher.is_enabled():
            self.ema_teacher.enable()


def create_ema_teacher(
    student_encoder: nn.Module,
    stage: int = 0,
    device: str = 'cuda',
    auto_enable: bool = False
) -> EMATeacher:
    """
    Factory function to create EMA teacher for specific stage.

    Args:
        student_encoder: Student encoder to track
        stage: Training stage (0, 1, or 2)
        device: Device to use
        auto_enable: Whether to enable immediately

    Returns:
        EMATeacher instance
    """
    # Get alpha for stage
    alpha_values = {0: 0.99, 1: 0.995, 2: 0.999}
    alpha = alpha_values.get(stage, 0.999)

    # Get enable iteration for stage
    enable_after = {0: 10000, 1: 0, 2: 0}
    update_after = enable_after.get(stage, 0)

    # Create teacher
    teacher = EMATeacher(
        student_encoder=student_encoder,
        alpha=alpha,
        update_after_step=update_after,
        device=device
    )

    if auto_enable:
        teacher.enable()

    return teacher


# Recommended alpha values for different scenarios
EMA_ALPHA_VALUES = {
    'very_fast': 0.9,      # Very fast tracking (not recommended)
    'fast': 0.99,          # Fast tracking (stage 0)
    'moderate': 0.995,     # Moderate tracking (stage 1)
    'slow': 0.999,         # Slow tracking (stage 2, recommended)
    'very_slow': 0.9999    # Very slow tracking (for stable models)
}


def get_recommended_alpha(stage: int) -> float:
    alpha_map = {
        0: EMA_ALPHA_VALUES['fast'],
        1: EMA_ALPHA_VALUES['moderate'],
        2: EMA_ALPHA_VALUES['slow']
    }
    return alpha_map.get(stage, EMA_ALPHA_VALUES['slow'])
