"""
Audio Utilities
"""
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Tuple
import hashlib
import os


def cache_id(path: str) -> str:
    """Return a stable unique id for a file path.

    Uses the normalized absolute path string. This prevents cache collisions when
    different folders contain the same filename.
    """
    p = os.path.abspath(path)
    p = os.path.normpath(p).replace(os.sep, "/")
    return hashlib.md5(p.encode("utf-8")).hexdigest()


def load_audio(
    audio_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
    duration: Optional[float] = None,
    offset: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file
    """
    audio, sr = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=mono,
        duration=duration,
        offset=offset,
    )

    return audio, sr


def save_audio(
    audio_path: str,
    audio: np.ndarray,
    sample_rate: int = 16000,
):
    """
    Save audio file
    """
    sf.write(audio_path, audio, sample_rate)


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
) -> np.ndarray:
    """
    Normalize audio to target dB
    """
    # Compute current RMS
    rms = np.sqrt(np.mean(audio ** 2))

    # Convert target dB to amplitude
    target_amp = 10 ** (target_db / 20)

    # Normalize
    if rms > 0:
        normalized = audio * (target_amp / rms)
    else:
        normalized = audio

    return normalized


def trim_silence(
    audio: np.ndarray,
    top_db: int = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    return trimmed


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    if orig_sr == target_sr:
        return audio

    resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    return resampled


def compute_rms_energy(
    audio: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    return energy


def apply_preemphasis(
    audio: np.ndarray,
    coef: float = 0.97,
) -> np.ndarray:
    emphasized = np.append(audio[0], audio[1:] - coef * audio[:-1])

    return emphasized


def apply_deemphasis(
    audio: np.ndarray,
    coef: float = 0.97,
) -> np.ndarray:
    deemphasized = np.zeros_like(audio)
    deemphasized[0] = audio[0]

    for i in range(1, len(audio)):
        deemphasized[i] = audio[i] + coef * deemphasized[i - 1]

    return deemphasized


def split_audio_chunks(
    audio: np.ndarray,
    chunk_length: int,
    overlap: int = 0,
) -> list:
    chunks = []
    hop = chunk_length - overlap

    for start in range(0, len(audio) - chunk_length + 1, hop):
        chunk = audio[start:start + chunk_length]
        chunks.append(chunk)

    return chunks


def test_audio_utils():
    """Test audio utilities"""
    print("Testing Audio Utilities...")

    # Create dummy audio
    audio = np.random.randn(16000 * 3).astype(np.float32)
    print(f"Original audio shape: {audio.shape}")

    # Test normalization
    normalized = normalize_audio(audio, target_db=-20.0)
    print(f"Normalized audio shape: {normalized.shape}")

    # Test RMS energy
    energy = compute_rms_energy(audio, hop_length=320)
    print(f"Energy shape: {energy.shape}")

    # Test resampling
    resampled = resample_audio(audio, orig_sr=16000, target_sr=8000)
    print(f"Resampled audio shape: {resampled.shape}")

    # Test chunking
    chunks = split_audio_chunks(audio, chunk_length=16000, overlap=0)
    print(f"Number of chunks: {len(chunks)}")

    print("\nTest passed!")


if __name__ == "__main__":
    test_audio_utils()
