import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .whisper_teacher import WhisperTeacher, create_whisper_teacher
from .mhubert_teacher import MHubertTeacher, create_mhubert_teacher
from .ema_teacher import EMATeacher, create_ema_teacher, EMAScheduler


class TeacherManager(nn.Module):
    """
    Unified manager for all teacher models.

    Coordinates three teachers:
    - Whisper: Content-focused, multilingual speech understanding
    - mHuBERT: Self-supervised speech representations
    - EMA: Temporal consistency via slow-moving student copy

    Combines their outputs with learnable or fixed weights.
    """

    def __init__(
        self,
        student_encoder: nn.Module,
        whisper_model_name: str = "openai/whisper-large-v3",
        mhubert_model_name: str = "utter-project/mHuBERT-147",
        stage: int = 0,
        device: str = 'cuda',
        use_learnable_weights: bool = False,
        auto_load: bool = True
    ):
        super().__init__()

        self.device = device
        self.stage = stage

        # Initialize teachers
        print("=" * 60)
        print("Initializing Teacher Manager")
        print("=" * 60)

        # 1. Whisper Teacher
        self.whisper = create_whisper_teacher(
            model_name=whisper_model_name,
            device=device,
            auto_load=auto_load
        )

        # 2. mHuBERT Teacher
        self.mhubert = create_mhubert_teacher(
            model_name=mhubert_model_name,
            device=device,
            auto_load=auto_load
        )

        # 3. EMA Teacher
        self.ema = create_ema_teacher(
            student_encoder=student_encoder,
            stage=stage,
            device=device,
            auto_enable=False  # Will be enabled by scheduler
        )

        # EMA Scheduler
        self.ema_scheduler = EMAScheduler(self.ema)
        self.ema_scheduler.set_stage(stage)

        # Teacher combination weights
        self.use_learnable_weights = use_learnable_weights

        if use_learnable_weights:
            # Learnable weights (will be normalized with softmax)
            self.weight_whisper = nn.Parameter(torch.tensor(1.0))
            self.weight_mhubert = nn.Parameter(torch.tensor(0.8))
            self.weight_ema = nn.Parameter(torch.tensor(0.6))
        else:
            # Fixed weights as per config
            self.register_buffer('weight_whisper', torch.tensor(0.5))
            self.register_buffer('weight_mhubert', torch.tensor(0.3))
            self.register_buffer('weight_ema', torch.tensor(0.2))

        print("=" * 60)
        print("Teacher Manager Initialized")
        print(f"  Whisper: {'Loaded' if self.whisper.is_model_loaded() else 'Not Loaded'}")
        print(f"  mHuBERT: {'Loaded' if self.mhubert.is_model_loaded() else 'Not Loaded'}")
        print(f"  EMA: {'Enabled' if self.ema.is_enabled() else 'Disabled'}")
        print(f"  Combination weights: {'Learnable' if use_learnable_weights else 'Fixed'}")
        print("=" * 60)

    def extract_all_features(
        self,
        mel_spec: torch.Tensor,
        waveform: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        features = {}

        # Whisper features
        if self.whisper.is_model_loaded():
            features['whisper'] = self.whisper(mel_spec)
        else:
            # Return zeros if not loaded
            B, _, T = mel_spec.shape
            features['whisper'] = torch.zeros(
                B, self.whisper.get_output_dim(), T,
                device=mel_spec.device
            )

        # mHuBERT features
        if self.mhubert.is_model_loaded() and waveform is not None:
            features['mhubert'] = self.mhubert(waveform)
        else:
            # Return zeros if not loaded or no waveform
            B, _, T = mel_spec.shape
            # Approximate time dimension for mHuBERT
            T_hubert = T // 2 if waveform is None else waveform.shape[1] // 320
            features['mhubert'] = torch.zeros(
                B, self.mhubert.get_output_dim(), T_hubert,
                device=mel_spec.device
            )

        # EMA features
        if self.ema.is_enabled():
            features['ema'] = self.ema(mel_spec)
        else:
            # Return zeros if not enabled
            B, _, T = mel_spec.shape
            features['ema'] = torch.zeros(
                B, 512, T,
                device=mel_spec.device
            )

        return features

    def get_combined_features(
        self,
        teacher_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:

        if self.use_learnable_weights:
            # Normalize weights with softmax
            weights = torch.softmax(
                torch.stack([
                    self.weight_whisper,
                    self.weight_mhubert,
                    self.weight_ema
                ]),
                dim=0
            )
            w_whisper, w_mhubert, w_ema = weights
        else:
            # Use fixed weights
            w_whisper = self.weight_whisper
            w_mhubert = self.weight_mhubert
            w_ema = self.weight_ema

        # NOTE: This is a simplified example
        # In practice, features need to be projected to same dimension first
        # See distillation.py for actual implementation

        return {
            'weights': {
                'whisper': w_whisper.item(),
                'mhubert': w_mhubert.item(),
                'ema': w_ema.item()
            }
        }

    def update_ema(self, student_encoder: nn.Module):
        self.ema.update(student_encoder)

    def set_stage(self, stage: int, iteration: int = 0):
        """
        Update all teachers for new training stage.

        """
        self.stage = stage

        # Update EMA scheduler
        self.ema_scheduler.set_stage(stage, iteration)

        print(f"\n{'='*60}")
        print(f"Teacher Manager: Stage {stage}")
        print(f"  EMA Alpha: {self.ema.get_alpha():.4f}")
        print(f"  EMA Enabled: {self.ema.is_enabled()}")
        print(f"{'='*60}\n")

    def step_ema_scheduler(self, iteration: int):
        self.ema_scheduler.step(iteration, self.stage)

    def enable_teacher(self, teacher_name: str):
        if teacher_name == 'ema':
            self.ema.enable()
        else:
            print(f"Teacher '{teacher_name}' is always enabled if loaded")

    def disable_teacher(self, teacher_name: str):
        if teacher_name == 'ema':
            self.ema.disable()
        else:
            print(f"Cannot disable '{teacher_name}' - unload model instead")

    def get_teacher_status(self) -> Dict[str, bool]:
        return {
            'whisper': self.whisper.is_model_loaded(),
            'mhubert': self.mhubert.is_model_loaded(),
            'ema': self.ema.is_enabled()
        }

    def get_teacher_dims(self) -> Dict[str, int]:
        return {
            'whisper': self.whisper.get_output_dim(),
            'mhubert': self.mhubert.get_output_dim(),
            'ema': 512  # EMA outputs student dimension
        }

    def get_combination_weights(self) -> Dict[str, float]:
        if self.use_learnable_weights:
            weights = torch.softmax(
                torch.stack([
                    self.weight_whisper,
                    self.weight_mhubert,
                    self.weight_ema
                ]),
                dim=0
            )
            return {
                'whisper': weights[0].item(),
                'mhubert': weights[1].item(),
                'ema': weights[2].item()
            }
        else:
            return {
                'whisper': self.weight_whisper.item(),
                'mhubert': self.weight_mhubert.item(),
                'ema': self.weight_ema.item()
            }

    def state_dict(self):
        state = {
            'ema': self.ema.state_dict(),
            'stage': self.stage,
            'use_learnable_weights': self.use_learnable_weights
        }

        if self.use_learnable_weights:
            state['weight_whisper'] = self.weight_whisper
            state['weight_mhubert'] = self.weight_mhubert
            state['weight_ema'] = self.weight_ema

        return state

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict['ema'])
        self.stage = state_dict.get('stage', 0)
        self.use_learnable_weights = state_dict.get('use_learnable_weights', False)

        if self.use_learnable_weights and 'weight_whisper' in state_dict:
            self.weight_whisper.data = state_dict['weight_whisper']
            self.weight_mhubert.data = state_dict['weight_mhubert']
            self.weight_ema.data = state_dict['weight_ema']

    def print_summary(self):
        print("\n" + "=" * 60)
        print("TEACHER MANAGER SUMMARY")
        print("=" * 60)

        # Status
        status = self.get_teacher_status()
        print("\nTeacher Status:")
        for name, active in status.items():
            status_str = "[Active]" if active else "[Inactive]"
            print(f"  {name:12s}: {status_str}")

        # Dimensions
        dims = self.get_teacher_dims()
        print("\nOutput Dimensions:")
        for name, dim in dims.items():
            print(f"  {name:12s}: {dim}")

        # Weights
        weights = self.get_combination_weights()
        print("\nCombination Weights:")
        for name, weight in weights.items():
            print(f"  {name:12s}: {weight:.4f}")

        # EMA specific
        print("\nEMA Details:")
        print(f"  Alpha: {self.ema.get_alpha():.4f}")
        print(f"  Updates: {self.ema.get_num_updates()}")
        print(f"  Enabled: {self.ema.is_enabled()}")

        print("=" * 60 + "\n")


def create_teacher_manager(
    student_encoder: nn.Module,
    whisper_model: str = "openai/whisper-large-v3",
    mhubert_model: str = "utter-project/mHuBERT-147",
    stage: int = 0,
    device: str = 'cuda',
    use_learnable_weights: bool = False,
    auto_load: bool = True
) -> TeacherManager:
    return TeacherManager(
        student_encoder=student_encoder,
        whisper_model_name=whisper_model,
        mhubert_model_name=mhubert_model,
        stage=stage,
        device=device,
        use_learnable_weights=use_learnable_weights,
        auto_load=auto_load
    )