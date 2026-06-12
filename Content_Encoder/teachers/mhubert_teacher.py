"""
mHuBERT Teacher Model for Knowledge Distillation
OPTIMIZED: Uses fp16, properly handles waveform input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MHubertTeacher(nn.Module):
    """
    mHuBERT teacher model wrapper for knowledge distillation.
    FIXED: Properly handles waveform input, uses half precision.
    """

    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        device: str = 'cuda',
        sample_rate: int = 16000,
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.model = None
        self.output_dim = 768  # HuBERT-base hidden size
        self.is_loaded = False

    def load_model(self):
        try:
            print(f"Loading mHuBERT teacher: {self.model_name}")
            print("  This may take a few minutes on first run...")

            # Try HubertModel first, then Wav2Vec2Model as fallback
            # (mHuBERT variants often use Wav2Vec2 architecture internally)
            model_loaded = False
            
            try:
                from transformers import HubertModel
                self.model = HubertModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    local_files_only=False
                )
                model_loaded = True
            except Exception as e1:
                print(f"  [INFO] HubertModel failed, trying Wav2Vec2Model...")
                try:
                    from transformers import Wav2Vec2Model
                    self.model = Wav2Vec2Model.from_pretrained(
                        self.model_name,
                        use_safetensors=True,
                        local_files_only=False
                    )
                    model_loaded = True
                except Exception as e2:
                    raise RuntimeError(
                        f"Both HubertModel and Wav2Vec2Model failed:\n"
                        f"  HubertModel: {e1}\n"
                        f"  Wav2Vec2Model: {e2}"
                    )

            self.model = self.model.to(self.device)
            
            # Update output_dim from actual model config
            if hasattr(self.model.config, 'hidden_size'):
                actual_dim = self.model.config.hidden_size
                if actual_dim != self.output_dim:
                    print(f"  [INFO] Updating output_dim: {self.output_dim} -> {actual_dim}")
                    self.output_dim = actual_dim
            
            # Convert to half precision for 2x speedup
            if self.device != 'cpu':
                self.model = self.model.half()
                print(f"  [OPTIMIZED] Using half precision (fp16)")
            
            self.model.eval()

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            self.is_loaded = True
            print(f"[OK] mHuBERT teacher loaded successfully")
            print(f"  Output dimension: {self.output_dim}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except ImportError:
            print("[FAIL] Failed to import transformers")
            self.is_loaded = False

        except Exception as e:
            print(f"[FAIL] Failed to load mHuBERT: {e}")
            self.is_loaded = False

    def forward(self, audio_or_mel: torch.Tensor, waveform: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass. Can accept either:
        - waveform: Raw audio [B, num_samples]
        - audio_or_mel: If 3D [B, 80, T], treats as mel and returns zeros
                       If 2D [B, samples], treats as waveform
        """
        if not self.is_loaded or self.model is None:
            # Return zeros
            if audio_or_mel.dim() == 3:
                batch_size, _, time_steps = audio_or_mel.shape
            else:
                batch_size = audio_or_mel.shape[0]
                time_steps = audio_or_mel.shape[1] // 320
            return torch.zeros(batch_size, self.output_dim, max(time_steps, 1), device=audio_or_mel.device)

        with torch.no_grad():
            try:
                # Determine if input is waveform or mel
                if waveform is not None:
                    # Explicit waveform provided
                    input_wav = waveform
                    target_time = waveform.shape[1] // 320  # Approx output frames
                elif audio_or_mel.dim() == 2 and audio_or_mel.shape[1] > 1000:
                    # 2D tensor with >1000 samples = likely waveform
                    input_wav = audio_or_mel
                    target_time = audio_or_mel.shape[1] // 320
                else:
                    # 3D tensor or short 2D = mel spectrogram, can't process
                    # Return zeros with correct shape
                    if audio_or_mel.dim() == 3:
                        batch_size, _, time_steps = audio_or_mel.shape
                    else:
                        batch_size = audio_or_mel.shape[0]
                        time_steps = 100
                    return torch.zeros(batch_size, self.output_dim, time_steps, device=audio_or_mel.device)

                # Process waveform
                batch_size = input_wav.shape[0]
                
                # Convert to half precision if model is half
                input_wav = input_wav.to(self.device)
                if next(self.model.parameters()).dtype == torch.float16:
                    input_wav = input_wav.half()

                # Pass through model
                outputs = self.model(
                    input_wav,
                    return_dict=True
                )

                # Get hidden states: [B, T, 768] and convert to float32
                features = outputs.last_hidden_state.float()

                # Transpose to [B, 768, T]
                features = features.transpose(1, 2)

                return features.to(audio_or_mel.device)

            except Exception as e:
                print(f"Warning: mHuBERT forward pass failed: {e}")
                if audio_or_mel.dim() == 3:
                    batch_size, _, time_steps = audio_or_mel.shape
                else:
                    batch_size = audio_or_mel.shape[0]
                    time_steps = audio_or_mel.shape[1] // 320 if audio_or_mel.dim() == 2 else 100
                return torch.zeros(batch_size, self.output_dim, max(time_steps, 1), device=audio_or_mel.device)

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.forward(waveform)

    def get_output_dim(self) -> int:
        return self.output_dim

    def is_model_loaded(self) -> bool:
        return self.is_loaded


def create_mhubert_teacher(
    model_name: str = "facebook/hubert-base-ls960",
    device: str = 'cuda',
    auto_load: bool = True
) -> MHubertTeacher:
    teacher = MHubertTeacher(model_name, device)
    if auto_load:
        teacher.load_model()
    return teacher


HUBERT_MODELS = {
    'mhubert-base': 'utter-project/mHuBERT-147',
    'mhubert-147': 'utter-project/mHuBERT-147',
    'hubert-base': 'facebook/hubert-base-ls960',
    'hubert-large': 'facebook/hubert-large-ll60k',
    'wav2vec2-xls-r-300m': 'facebook/wav2vec2-xls-r-300m',
}


def get_hubert_model_name(model_size: str) -> str:
    return HUBERT_MODELS.get(model_size, HUBERT_MODELS['hubert-base'])