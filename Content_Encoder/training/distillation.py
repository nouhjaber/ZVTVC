import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import logging

# Import EMATeacher from teachers module
from teachers.ema_teacher import EMATeacher

logger = logging.getLogger(__name__)


class ProjectionLayer(nn.Module):
    """
    Projects student features to teacher feature dimension.
    
    student_dim (512) -> teacher_dim (varies by model)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        
        logger.info(f"[ProjectionLayer] Created: {in_dim} -> {out_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, in_dim, T]
        # Transpose for linear layer [B, in_dim, T] -> [B, T, in_dim]
        x = x.transpose(1, 2)

        # Linear and activation
        x = self.linear(x)  # [B, T, out_dim]
        x = self.activation(x)

        # Transpose back [B, T, out_dim] -> [B, out_dim, T]
        x = x.transpose(1, 2)

        return x


class DistillationLoss(nn.Module):
    """
    Multi-teacher distillation loss.
    
    FIXED: Projection layers are created dynamically based on actual teacher dimensions.
    """

    def __init__(
        self,
        student_dim: int = 512,
        whisper_dim: int = None,  # Will be set from teacher
        mhubert_dim: int = None,  # Will be set from teacher
        whisper_weight: float = 0.5,
        mhubert_weight: float = 0.3,
        ema_weight: float = 0.2,
        ema_start_iter: int = 10000,
        teacher_frequency: int = 1  # NEW: Run teachers every N iterations
    ):
        super().__init__()

        self.student_dim = student_dim
        
        # Store target dimensions (will be updated when teachers are loaded)
        self._whisper_dim = whisper_dim
        self._mhubert_dim = mhubert_dim

        # Projection layers (initialized as None, created when teachers are loaded)
        self.proj_whisper = None
        self.proj_mhubert = None

        # Weights
        self.whisper_weight = whisper_weight
        self.mhubert_weight = mhubert_weight
        self.ema_weight = ema_weight

        # EMA configuration
        self.ema_start_iter = ema_start_iter
        
        # NEW: Teacher frequency control
        self.teacher_frequency = teacher_frequency

        # Teacher models (to be loaded separately)
        self.whisper_model = None
        self.mhubert_model = None
        self.ema_teacher = None
        
        # Track if projections have been initialized
        self._projections_initialized = False

        logger.info(f"[DistillationLoss] Initialized - student_dim={student_dim}")
        logger.info(f"[DistillationLoss] Weights - Whisper:{whisper_weight}, mHuBERT:{mhubert_weight}, EMA:{ema_weight}")
        logger.info(f"[DistillationLoss] EMA starts at iteration {ema_start_iter}")
        logger.info(f"[DistillationLoss] Teacher frequency: every {teacher_frequency} iterations")
        logger.info(f"[DistillationLoss] Projection layers will be created when teachers are loaded")

    def load_teachers(
        self,
        whisper_model: nn.Module,
        mhubert_model: nn.Module,
        ema_teacher: EMATeacher
    ):
        """
        Load pre-trained teacher models and create projection layers.
        
        FIXED: Gets actual output dimensions from teachers.
        """
        self.whisper_model = whisper_model
        self.mhubert_model = mhubert_model
        self.ema_teacher = ema_teacher

        # Get actual dimensions from teachers
        if whisper_model is not None:
            self._whisper_dim = whisper_model.get_output_dim()
            logger.info(f"[DistillationLoss] Whisper output dim: {self._whisper_dim}")
        
        if mhubert_model is not None:
            self._mhubert_dim = mhubert_model.get_output_dim()
            logger.info(f"[DistillationLoss] mHuBERT output dim: {self._mhubert_dim}")

        # Create projection layers with correct dimensions
        self._create_projection_layers()

        # Freeze teacher models
        if self.whisper_model is not None:
            for param in self.whisper_model.parameters():
                param.requires_grad = False
            self.whisper_model.eval()

        if self.mhubert_model is not None:
            for param in self.mhubert_model.parameters():
                param.requires_grad = False
            self.mhubert_model.eval()
            
        logger.info(f"[DistillationLoss] Teachers loaded successfully")

    def _create_projection_layers(self):
        """Create projection layers based on actual teacher dimensions."""
        # Try to get device from existing parameters, fallback to stored device info
        try:
            device = next(self.parameters()).device
        except StopIteration:
            # No parameters yet - check if whisper_model has a device
            if self.whisper_model is not None:
                try:
                    device = next(self.whisper_model.parameters()).device
                except StopIteration:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create Whisper projection
        if self._whisper_dim is not None:
            self.proj_whisper = ProjectionLayer(self.student_dim, self._whisper_dim)
            self.proj_whisper = self.proj_whisper.to(device)
            logger.info(f"[DistillationLoss] Created Whisper projection: {self.student_dim} -> {self._whisper_dim}")
        
        # Create mHuBERT projection
        if self._mhubert_dim is not None:
            self.proj_mhubert = ProjectionLayer(self.student_dim, self._mhubert_dim)
            self.proj_mhubert = self.proj_mhubert.to(device)
            logger.info(f"[DistillationLoss] Created mHuBERT projection: {self.student_dim} -> {self._mhubert_dim}")
        
        self._projections_initialized = True

    def to(self, device):
        """Override to() to also move dynamically created projection layers."""
        super().to(device)
        if self.proj_whisper is not None:
            self.proj_whisper = self.proj_whisper.to(device)
        if self.proj_mhubert is not None:
            self.proj_mhubert = self.proj_mhubert.to(device)
        return self

    def forward(
        self,
        z_c_student: torch.Tensor,
        audio_or_mel: torch.Tensor,
        iteration: int,
        waveform: torch.Tensor = None  # NEW: Pass raw waveform for mHuBERT
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss from all teachers."""
        logger.debug(f"[DistillationLoss] Forward - student: {z_c_student.shape}, iteration={iteration}")
        losses = {}
        total_loss = torch.tensor(0.0, device=z_c_student.device, dtype=z_c_student.dtype)
        
        # NEW: Skip teacher computation if not on frequency schedule
        skip_teachers = (iteration % self.teacher_frequency != 0)
        if skip_teachers:
            logger.debug(f"[DistillationLoss] Skipping teachers (iteration {iteration} not divisible by {self.teacher_frequency})")
            losses['whisper'] = torch.tensor(0.0, device=z_c_student.device)
            losses['mhubert'] = torch.tensor(0.0, device=z_c_student.device)
            losses['ema'] = torch.tensor(0.0, device=z_c_student.device)
            losses['total'] = torch.tensor(0.0, device=z_c_student.device)
            return losses

        # 1. Whisper distillation
        if self.whisper_model is not None and self.proj_whisper is not None:
            with torch.no_grad():
                # Extract Whisper features (frozen)
                z_whisper = self.extract_whisper_features(audio_or_mel)
                
            logger.debug(f"[DistillationLoss] Whisper features: {z_whisper.shape}")

            # Project student features to Whisper dimension
            z_student_proj = self.proj_whisper(z_c_student)
            logger.debug(f"[DistillationLoss] Student projected: {z_student_proj.shape}")

            # Align lengths if needed
            z_student_proj, z_whisper = self.align_features(z_student_proj, z_whisper)

            # Compute MSE loss
            loss_whisper = F.mse_loss(z_student_proj, z_whisper)
            losses['whisper'] = loss_whisper
            total_loss += self.whisper_weight * loss_whisper
            logger.debug(f"[DistillationLoss] Whisper loss: {loss_whisper.item():.4f}")
        else:
            losses['whisper'] = torch.tensor(0.0, device=z_c_student.device)
            if self.whisper_model is None:
                logger.debug(f"[DistillationLoss] Whisper model not loaded")
            elif self.proj_whisper is None:
                logger.debug(f"[DistillationLoss] Whisper projection not initialized")

        # 2. mHuBERT distillation (requires raw waveform)
        if self.mhubert_model is not None and self.proj_mhubert is not None and waveform is not None:
            with torch.no_grad():
                # Extract mHuBERT features from raw waveform (frozen)
                z_mhubert = self.mhubert_model(waveform)

            logger.debug(f"[DistillationLoss] mHuBERT features: {z_mhubert.shape}")

            # Project student features to mHuBERT dimension
            z_student_proj_mhubert = self.proj_mhubert(z_c_student)
            logger.debug(f"[DistillationLoss] Student projected for mHuBERT: {z_student_proj_mhubert.shape}")

            # Align lengths if needed
            z_student_proj_mhubert, z_mhubert = self.align_features(z_student_proj_mhubert, z_mhubert)

            # Compute MSE loss
            loss_mhubert = F.mse_loss(z_student_proj_mhubert, z_mhubert)
            losses['mhubert'] = loss_mhubert
            total_loss += self.mhubert_weight * loss_mhubert
            logger.debug(f"[DistillationLoss] mHuBERT loss: {loss_mhubert.item():.4f}")
        else:
            losses['mhubert'] = torch.tensor(0.0, device=z_c_student.device)
            if self.mhubert_model is None:
                logger.debug(f"[DistillationLoss] mHuBERT model not loaded")
            elif self.proj_mhubert is None:
                logger.debug(f"[DistillationLoss] mHuBERT projection not initialized")
            elif waveform is None:
                logger.debug(f"[DistillationLoss] No waveform provided for mHuBERT")

        # 3. EMA distillation (only after ema_start_iter)
        if self.ema_teacher is not None and iteration >= self.ema_start_iter:
            with torch.no_grad():
                # Extract EMA features (frozen, no grad)
                z_ema = self.ema_teacher(audio_or_mel)

            # No projection needed (same dimension as student)
            # Align lengths if needed
            z_c_student_aligned, z_ema = self.align_features(z_c_student, z_ema)

            # Compute MSE loss
            loss_ema = F.mse_loss(z_c_student_aligned, z_ema)
            losses['ema'] = loss_ema
            total_loss += self.ema_weight * loss_ema
            logger.debug(f"[DistillationLoss] EMA loss: {loss_ema.item():.4f} (iteration={iteration})")
        else:
            losses['ema'] = torch.tensor(0.0, device=z_c_student.device)
            if self.ema_teacher is None:
                logger.debug(f"[DistillationLoss] EMA teacher not loaded")
            else:
                logger.debug(f"[DistillationLoss] EMA DISABLED (iteration {iteration} < {self.ema_start_iter})")

        losses['total'] = total_loss
        logger.debug(f"[DistillationLoss] Total distillation loss: {total_loss.item():.4f}")

        return losses

    def extract_whisper_features(self, audio_or_mel: torch.Tensor) -> torch.Tensor:
        """Extract features from Whisper teacher."""
        if self.whisper_model is None:
            batch_size = audio_or_mel.shape[0]
            time_steps = audio_or_mel.shape[2] if len(audio_or_mel.shape) == 3 else 100
            dim = self._whisper_dim or 1280
            return torch.zeros(batch_size, dim, time_steps, device=audio_or_mel.device)

        try:
            with torch.no_grad():
                features = self.whisper_model.forward(audio_or_mel)
            return features

        except Exception as e:
            print(f"Warning: Whisper feature extraction failed: {e}")
            batch_size = audio_or_mel.shape[0]
            time_steps = audio_or_mel.shape[2]
            dim = self._whisper_dim or 1280
            return torch.zeros(batch_size, dim, time_steps, device=audio_or_mel.device)

    def extract_mhubert_features(self, audio_or_mel: torch.Tensor) -> torch.Tensor:
        """Extract features from mHuBERT teacher."""
        if self.mhubert_model is None:
            batch_size = audio_or_mel.shape[0]
            time_steps = audio_or_mel.shape[2] if len(audio_or_mel.shape) == 3 else 100
            dim = self._mhubert_dim or 768
            return torch.zeros(batch_size, dim, time_steps, device=audio_or_mel.device)

        try:
            with torch.no_grad():
                features = self.mhubert_model.forward(audio_or_mel)
            return features

        except Exception as e:
            print(f"Warning: mHuBERT feature extraction failed: {e}")
            batch_size = audio_or_mel.shape[0]
            time_steps = audio_or_mel.shape[2]
            dim = self._mhubert_dim or 768
            return torch.zeros(batch_size, dim, time_steps, device=audio_or_mel.device)

    def align_features(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> tuple:
        """Align two feature tensors to the same temporal length."""
        # Check for empty tensors
        if feat1.shape[2] == 0 or feat2.shape[2] == 0:
            batch_size = feat1.shape[0]
            dim1 = feat1.shape[1]
            dim2 = feat2.shape[1]
            max_len = max(feat1.shape[2], feat2.shape[2], 1)
            feat1 = torch.zeros(batch_size, dim1, max_len, device=feat1.device)
            feat2 = torch.zeros(batch_size, dim2, max_len, device=feat2.device)
            return feat1, feat2

        if feat1.shape[2] == feat2.shape[2]:
            return feat1, feat2

        # Get target length (use shorter one to avoid extrapolation)
        target_len = min(feat1.shape[2], feat2.shape[2])

        # Interpolate both to target length
        if feat1.shape[2] != target_len and target_len > 0:
            feat1 = F.interpolate(
                feat1,
                size=target_len,
                mode='linear',
                align_corners=False
            )

        if feat2.shape[2] != target_len and target_len > 0:
            feat2 = F.interpolate(
                feat2,
                size=target_len,
                mode='linear',
                align_corners=False
            )

        return feat1, feat2
    
    def get_projection_info(self) -> Dict[str, str]:
        """Get info about projection layers for debugging."""
        info = {
            'whisper_proj': f"{self.student_dim} -> {self._whisper_dim}" if self.proj_whisper else "Not initialized",
            'mhubert_proj': f"{self.student_dim} -> {self._mhubert_dim}" if self.proj_mhubert else "Not initialized",
        }
        return info


# Keep these for backward compatibility
class WhisperFeatureExtractor:
    """Deprecated: Use WhisperTeacher instead."""
    def __init__(self, model_name: str = "openai/whisper-large-v3", device: str = 'cuda'):
        print("WARNING: WhisperFeatureExtractor is deprecated. Use WhisperTeacher from teachers/ instead.")
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        try:
            from transformers import WhisperModel
            print(f"Loading Whisper model: {self.model_name}")
            self.model = WhisperModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[OK] Whisper model loaded")
        except Exception as e:
            print(f"[FAIL] Failed to load Whisper model: {e}")
            self.model = None

    def extract_features(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            return torch.zeros(mel_spec.shape[0], 1280, mel_spec.shape[2], device=mel_spec.device)
        with torch.no_grad():
            encoder_outputs = self.model.encoder(mel_spec.to(self.device), return_dict=True)
            features = encoder_outputs.last_hidden_state.transpose(1, 2)
        return features


class MHubertFeatureExtractor:
    """Deprecated: Use MHubertTeacher instead."""
    def __init__(self, model_name: str = "facebook/mhubert-base-25langs", device: str = 'cuda'):
        print("WARNING: MHubertFeatureExtractor is deprecated. Use MHubertTeacher from teachers/ instead.")
        self.model_name = model_name
        self.device = device
        self.model = None

    def load_model(self):
        try:
            from transformers import HubertModel
            print(f"Loading mHuBERT model: {self.model_name}")
            self.model = HubertModel.from_pretrained(self.model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"[OK] mHuBERT model loaded")
        except Exception as e:
            print(f"[FAIL] Failed to load mHuBERT model: {e}")
            self.model = None

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            time_steps = waveform.shape[1] // 320
            return torch.zeros(waveform.shape[0], 768, time_steps, device=waveform.device)
        with torch.no_grad():
            outputs = self.model(waveform.to(self.device), return_dict=True)
            features = outputs.last_hidden_state.transpose(1, 2)
        return features