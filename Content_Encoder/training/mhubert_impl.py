"""
mHuBERT Teacher Model for Knowledge Distillation

FIXED: Dynamic output_dim based on actual model loaded
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# HuBERT model hidden dimensions
HUBERT_HIDDEN_DIMS = {
    'facebook/hubert-base-ls960': 768,
    'facebook/hubert-large-ll60k': 1024,
    'facebook/hubert-xlarge-ll60k': 1280,
    'facebook/mhubert-base-25langs': 768,  # Multilingual HuBERT
    'facebook/wav2vec2-base': 768,
    'facebook/wav2vec2-large': 1024,
    'facebook/wav2vec2-xls-r-300m': 1024,
    'facebook/wav2vec2-xls-r-1b': 1280,
    'facebook/wav2vec2-xls-r-2b': 1920,
}


class MHubertTeacher(nn.Module):
    """
    mHuBERT teacher model wrapper for knowledge distillation.

    Uses pre-trained mHuBERT (Multilingual HuBERT) to extract
    self-supervised speech representations.
    """

    def __init__(
        self,
        model_name: str = "facebook/mhubert-base-25langs",
        device: str = 'cuda',
        sample_rate: int = 16000,
        freeze: bool = True
    ):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.sample_rate = sample_rate
        self.model = None
        self.is_loaded = False
        
        # FIXED: Set output_dim based on model name (can be updated after loading)
        self.output_dim = HUBERT_HIDDEN_DIMS.get(model_name, 768)
        print(f"[MHubertTeacher] Initialized for {model_name}, expected output_dim={self.output_dim}")

    def load_model(self):
        try:
            from transformers import HubertModel, Wav2Vec2Model
            
            print(f"Loading mHuBERT teacher: {self.model_name}")
            print("  This may take a few minutes on first run...")

            # Determine which model class to use
            if 'wav2vec2' in self.model_name.lower():
                self.model = Wav2Vec2Model.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    local_files_only=False
                )
            else:
                self.model = HubertModel.from_pretrained(
                    self.model_name,
                    use_safetensors=True,
                    local_files_only=False
                )
            
            self.model = self.model.to(self.device)
            self.model.eval()

            # Freeze all parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # FIXED: Get actual output dimension from loaded model config
            actual_dim = self.model.config.hidden_size
            if actual_dim != self.output_dim:
                print(f"[MHubertTeacher] WARNING: Updating output_dim from {self.output_dim} to {actual_dim}")
                self.output_dim = actual_dim

            self.is_loaded = True
            print(f"[OK] mHuBERT teacher loaded successfully")
            print(f"  Model: {self.model_name}")
            print(f"  Output dimension: {self.output_dim}")
            print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        except ImportError:
            print("[FAIL] Failed to import transformers")
            print("   Install with: pip install transformers")
            self.is_loaded = False

        except Exception as e:
            print(f"[FAIL] Failed to load mHuBERT: {e}")
            self.is_loaded = False

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from waveform.
        
        Args:
            waveform: Raw audio [B, num_samples] or mel-spectrogram [B, 80, T]
            
        Returns:
            Features [B, hidden_dim, T]
        """
        if not self.is_loaded or self.model is None:
            # Return zeros if model not loaded
            batch_size = waveform.shape[0]
            if waveform.dim() == 2:
                time_steps = waveform.shape[1] // 320  # Approximate
            else:
                time_steps = waveform.shape[-1]
            return torch.zeros(batch_size, self.output_dim, time_steps, device=waveform.device)

        with torch.no_grad():
            try:
                # Check if input is mel-spectrogram instead of waveform
                if waveform.dim() > 2 or (waveform.dim() == 2 and waveform.shape[1] < 1000):
                    # Likely mel-spectrogram [B, 80, T], return zeros
                    # mHuBERT requires raw audio waveform, not mel-spectrogram
                    batch_size = waveform.shape[0]
                    time_steps = waveform.shape[-1]
                    return torch.zeros(batch_size, self.output_dim, time_steps, device=waveform.device)

                # Ensure 2D input [B, num_samples]
                if waveform.dim() > 2:
                    waveform = waveform.squeeze()

                # Pass through model
                outputs = self.model(
                    waveform.to(self.device),
                    return_dict=True
                )

                # Get hidden states: [B, T, hidden_dim]
                features = outputs.last_hidden_state

                # Transpose to [B, hidden_dim, T]
                features = features.transpose(1, 2)

                return features.to(waveform.device)

            except Exception as e:
                print(f"Warning: mHuBERT forward pass failed: {e}")
                batch_size = waveform.shape[0]
                if waveform.dim() == 2:
                    time_steps = waveform.shape[1] // 320
                else:
                    time_steps = waveform.shape[-1]
                return torch.zeros(batch_size, self.output_dim, time_steps, device=waveform.device)

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convenience method for feature extraction."""
        return self.forward(waveform)

    def extract_from_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Note: mHuBERT requires waveform input, not mel-spectrogram.
        Returns zeros and prints a warning.
        """
        print("Warning: mHuBERT requires waveform input, not mel-spectrogram")
        print("  Returning zeros. Pass waveform directly for best results.")

        batch_size, n_mels, time_steps = mel_spec.shape
        out_time = time_steps // 2  # Rough approximation
        return torch.zeros(batch_size, self.output_dim, out_time, device=mel_spec.device)

    def get_output_dim(self) -> int:
        """Get output feature dimension."""
        return self.output_dim

    def is_model_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.is_loaded

    def preprocess_waveform(self, waveform: torch.Tensor, current_sr: int = None) -> torch.Tensor:
        """
        Preprocess waveform for mHuBERT.

        Ensures:
        - 16kHz sample rate
        - Normalized amplitude
        - Correct shape
        """
        # Add batch dimension if needed
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # Resample if needed
        if current_sr is not None and current_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=current_sr,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # Normalize amplitude to [-1, 1]
        max_val = torch.abs(waveform).max()
        if max_val > 1.0:
            waveform = waveform / max_val

        return waveform


def create_mhubert_teacher(
    model_name: str = "facebook/mhubert-base-25langs",
    device: str = 'cuda',
    auto_load: bool = True
) -> MHubertTeacher:
    """
    Factory function to create MHubertTeacher.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        auto_load: Whether to load model immediately
        
    Returns:
        MHubertTeacher instance
    """
    teacher = MHubertTeacher(model_name, device)

    if auto_load:
        teacher.load_model()

    return teacher


# Available mHuBERT and similar models with their hidden dimensions
HUBERT_MODELS = {
    'mhubert-base': ('facebook/mhubert-base-25langs', 768),         # 95M params, 25 languages
    'hubert-base': ('facebook/hubert-base-ls960', 768),             # 95M params, English
    'hubert-large': ('facebook/hubert-large-ll60k', 1024),          # 316M params
    'wav2vec2-base': ('facebook/wav2vec2-base', 768),               # 95M params
    'wav2vec2-large': ('facebook/wav2vec2-large', 1024),            # 316M params
    'wav2vec2-xls-r-300m': ('facebook/wav2vec2-xls-r-300m', 1024),  # 300M params, 128 languages
    'wav2vec2-xls-r-1b': ('facebook/wav2vec2-xls-r-1b', 1280),      # 1B params, 128 languages
    'wav2vec2-xls-r-2b': ('facebook/wav2vec2-xls-r-2b', 1920),      # 2B params, 128 languages
}


def get_hubert_model_name(model_size: str) -> str:
    """Get HuggingFace model name from identifier."""
    if model_size in HUBERT_MODELS:
        return HUBERT_MODELS[model_size][0]
    return HUBERT_MODELS['mhubert-base'][0]


def get_hubert_hidden_dim(model_size: str) -> int:
    """Get hidden dimension for a given model size."""
    if model_size in HUBERT_MODELS:
        return HUBERT_MODELS[model_size][1]
    return 768  # Default
