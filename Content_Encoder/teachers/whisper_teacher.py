"""
Whisper Teacher Model for Knowledge Distillation
OPTIMIZED: Uses fp16, handles 80-dim mel by projecting to Whisper's 128-dim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WhisperTeacher(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: str = 'cuda',
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.model = None
        self.feature_extractor = None  # Whisper's own mel extractor
        # Dynamic output_dim based on model
        WHISPER_DIMS = {
            'openai/whisper-tiny': 384,
            'openai/whisper-base': 512,
            'openai/whisper-small': 768,
            'openai/whisper-medium': 1024,
            'openai/whisper-large-v3': 1280,
        }
        self.output_dim = WHISPER_DIMS.get(model_name, 1280)
        self.is_loaded = False
        self._whisper_n_mels = 128  # Whisper expects 128-dim mel

    def load_model(self):
        try:
            from transformers import WhisperModel, WhisperFeatureExtractor

            print(f"Loading Whisper teacher: {self.model_name}")
            print("  This may take a few minutes on first run...")

            self.model = WhisperModel.from_pretrained(
                self.model_name,
                use_safetensors=True,
                local_files_only=False
            )
            self.model = self.model.to(self.device)
            
            # Load Whisper's own feature extractor for computing correct mel
            try:
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    self.model_name
                )
                self._whisper_n_mels = self.feature_extractor.feature_size
                print(f"  [OK] Whisper feature extractor loaded (n_mels={self._whisper_n_mels})")
            except Exception as e:
                print(f"  [WARN] Could not load feature extractor: {e}")
                self.feature_extractor = None
            
            # Convert to half precision for 2x speedup
            if self.device != 'cpu':
                self.model = self.model.half()
                print(f"  [OPTIMIZED] Using half precision (fp16)")
            
            self.model.eval()
            actual_dim = self.model.config.d_model
            if actual_dim != self.output_dim:
                print(f"[WARNING] Updating output_dim: {self.output_dim} -> {actual_dim}")
                self.output_dim = actual_dim

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            self.is_loaded = True
            print(f"[OK] Whisper teacher loaded successfully")
            print(f"  Output dimension: {self.output_dim}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except ImportError:
            print("[FAIL] Failed to import transformers")
            self.is_loaded = False

        except Exception as e:
            print(f"[FAIL] Failed to load Whisper: {e}")
            self.is_loaded = False

    def _prepare_mel_for_whisper(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert 80-dim mel to Whisper-compatible input.
        Whisper expects [B, 128, 3000] (or model-specific n_mels).
        We receive [B, 80, T] from our mel transform.
        
        Strategy: Use a linear projection from 80 -> 128 mel bins,
        then pad/truncate to 3000 frames.
        """
        batch_size, n_mels, time_steps = mel_spec.shape
        
        if n_mels == self._whisper_n_mels:
            # Already correct dimension, just handle length
            mel = mel_spec
        elif n_mels < self._whisper_n_mels:
            # Project 80 -> 128 using interpolation along mel axis
            # [B, 80, T] -> [B, 128, T]
            mel = F.interpolate(
                mel_spec.transpose(1, 2),  # [B, T, 80]
                size=self._whisper_n_mels,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [B, 128, T]
        else:
            # Truncate if somehow larger
            mel = mel_spec[:, :self._whisper_n_mels, :]
        
        # Whisper REQUIRES exactly 3000 frames
        target_length = 3000
        if mel.shape[2] < target_length:
            pad_length = target_length - mel.shape[2]
            mel = F.pad(mel, (0, pad_length), mode='constant', value=0)
        else:
            mel = mel[:, :, :target_length]
        
        return mel

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded or self.model is None:
            batch_size, n_mels, time_steps = mel_spec.shape
            return torch.zeros(batch_size, self.output_dim, time_steps, device=mel_spec.device)

        with torch.no_grad():
            try:
                batch_size, n_mels, time_steps = mel_spec.shape
                original_length = time_steps

                # Prepare mel for Whisper (80 -> 128 dims, pad to 3000)
                input_mel = self._prepare_mel_for_whisper(mel_spec)
                
                # Move to device and convert precision
                input_mel = input_mel.to(self.device)
                if next(self.model.parameters()).dtype == torch.float16:
                    input_mel = input_mel.half()

                # Pass through encoder
                encoder_outputs = self.model.encoder(
                    input_mel,
                    return_dict=True
                )

                # Get hidden states and convert back to float32
                features = encoder_outputs.last_hidden_state.float()
                features = features.transpose(1, 2)

                # Interpolate back to original length
                if features.shape[2] != original_length:
                    features = F.interpolate(
                        features,
                        size=original_length,
                        mode='linear',
                        align_corners=False
                    )

                return features.to(mel_spec.device)

            except Exception as e:
                print(f"Warning: Whisper forward pass failed: {e}")
                batch_size, n_mels, time_steps = mel_spec.shape
                return torch.zeros(batch_size, self.output_dim, time_steps, device=mel_spec.device)

    def extract_features(self, mel_spec: torch.Tensor) -> torch.Tensor:
        return self.forward(mel_spec)

    def get_output_dim(self) -> int:
        return self.output_dim

    def is_model_loaded(self) -> bool:
        return self.is_loaded


def create_whisper_teacher(
    model_name: str = "openai/whisper-large-v3",
    device: str = 'cuda',
    auto_load: bool = True
) -> WhisperTeacher:
    teacher = WhisperTeacher(model_name, device)
    if auto_load:
        teacher.load_model()
    return teacher


WHISPER_MODELS = {
    'tiny': 'openai/whisper-tiny',
    'base': 'openai/whisper-base',
    'small': 'openai/whisper-small',
    'medium': 'openai/whisper-medium',
    'large': 'openai/whisper-large-v2',
    'large-v3': 'openai/whisper-large-v3'
}


def get_whisper_model_name(model_size: str) -> str:
    return WHISPER_MODELS.get(model_size, WHISPER_MODELS['large-v3'])