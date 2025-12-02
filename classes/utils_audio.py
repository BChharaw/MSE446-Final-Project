# utils_audio.py
import random
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch

# ============================================================================
# Audio I/O and STFT utilities
# ============================================================================
# This module centralizes all low-level audio handling:
#   - Loading and resampling WAV/MP3/FLAC files into mono float32 numpy arrays.
#   - Saving waveforms back to disk in a consistent format.
#   - Computing STFT / iSTFT in numpy-land, with a unified signature that
#     hides whether torchaudio or librosa is actually doing the work.
#   - Basic helpers for magnitude/phase extraction, file discovery, and
#     reproducibility (global seed setting).
#
# The design goal is to allow the rest of the codebase (datasets, models,
# training, evaluation) to rely on a small, stable API without worrying about
# backend details or library availability differences.
# ============================================================================

# Prefer torchaudio if available for speed/resampling. Fall back to librosa + soundfile.
try:
    import torchaudio

    HAS_TORCHAUDIO = True
except Exception:
    HAS_TORCHAUDIO = False
    import librosa
    import soundfile as sf


def load_wav(path: Path, target_sr: int = 16000) -> np.ndarray:
    """
    Load audio file (wav/mp3/flac) as mono float32 numpy array resampled to target_sr.
    Returns waveform only (shape=(n_samples,)).

    Behavior:
      - If torchaudio is available:
          * Use torchaudio.load to support many formats and get (channels, samples).
          * Average across channels to get mono.
          * If the source sample rate differs from target_sr, resample using
            torchaudio.functional.resample for consistent downstream processing.
      - If torchaudio is not available:
          * Use librosa.load with sr=target_sr and mono=True.
      - In both paths:
          * Normalize by the maximum absolute value to keep samples in [-1, 1].
          * Return a float32 numpy array for compatibility with STFT utilities.
    """
    p = str(path)
    if HAS_TORCHAUDIO:
        waveform, src_sr = torchaudio.load(p)  # shape (channels, samples)
        waveform = waveform.mean(dim=0).numpy()  # mono
        if src_sr != target_sr:
            waveform = (
                torchaudio.functional.resample(
                    torch.from_numpy(waveform).unsqueeze(0), src_sr, target_sr
                )
                .squeeze(0)
                .numpy()
            )
        # normalize to [-1,1]
        mx = np.max(np.abs(waveform)) or 1.0
        return (waveform / mx).astype(np.float32)
    else:
        y, _ = librosa.load(p, sr=target_sr, mono=True)
        mx = np.max(np.abs(y)) or 1.0
        return (y / mx).astype(np.float32)


def save_wave(waveform, path, sample_rate: int = 16000):
    """
    Save a waveform to disk as a 16-bit PCM WAV file.

    Details:
      - Accepts either numpy arrays or torch.Tensor.
      - Any extra dimensions (e.g., batch, channels) are flattened into a
        single 1D signal, which is sufficient for logging/inspection.
      - Ensures float32 dtype before handing off to soundfile.
      - Creates parent directories as needed.
      - Uses subtype="PCM_16" to produce standard 16-bit WAV files that are
        widely compatible with external tools and audio players.
    """
    # Convert torch tensor to numpy
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
        # # print("Converted from tensor to numpy, shape:", waveform.shape)

    # Flatten all dimensions except last one
    if waveform.ndim > 1:
        waveform = np.reshape(waveform, (-1,))
        # # print("Flattened waveform shape:", waveform.shape)

    # Ensure float32
    waveform = waveform.astype(np.float32)
    # print("Final dtype:", waveform.dtype)

    # Path checks
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # print("Path parent created if needed.")

    # Attempt save
    try:
        sf.write(str(path), waveform, samplerate=sample_rate, subtype="PCM_16")
        # print("WAV saved successfully!")
    except Exception:
        # print("ERROR saving WAV:", e)
        raise


def stft_np(waveform: np.ndarray, n_fft: int = 512, hop_length: int = 128):
    """
    Return complex STFT matrix (numpy). Signature matches calls in your code.

    The goal of this function is to provide a unified numpy-based STFT interface:
      - If torchaudio is available, leverage torch.stft (via a small wrapper)
        for performance and GPU friendliness in other contexts.
      - If not, defer to librosa.stft, which is widely available and robust.

    The returned matrix is complex-valued with shape (freq_bins, time_frames),
    and uses a Hann window with win_length = n_fft to match iSTFT assumptions.
    """
    if HAS_TORCHAUDIO:
        import torch

        t = torch.from_numpy(waveform).unsqueeze(0)
        spec = torch.stft(
            t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
            return_complex=True,
        )
        return spec.squeeze(0).numpy()
    else:
        import librosa

        return librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
        )


def istft_np(spec_complex: np.ndarray, hop_length: int = 128):
    """
    Inverse STFT. Returns time-domain numpy waveform.

    Implementation notes:
      - The function mirrors stft_np and uses the same n_fft and window
        assumptions to ensure perfect (or near-perfect) reconstruction for
        well-behaved inputs.
      - When using PyTorch:
          * Convert the numpy complex array to a complex torch tensor.
          * Use torch.istft directly, which is more consistent across
            torchaudio versions than torchaudio.functional.istft.
      - When using librosa:
          * Compute n_fft from spec shape and call librosa.istft with a Hann
            window and matching parameters.
    """
    if HAS_TORCHAUDIO:
        import torch

        # Convert numpy complex array to a complex torch tensor. Some torchaudio
        # builds don't provide torchaudio.functional.istft; use torch.istft
        # (part of PyTorch) which is commonly available.
        t = torch.from_numpy(spec_complex)
        # ensure complex dtype
        if not torch.is_complex(t):
            t = t.to(torch.complex64)
        t = t.unsqueeze(0)
        n_fft = (spec_complex.shape[0] - 1) * 2
        # Use torch.istft (safer across torchaudio versions)
        x = torch.istft(
            t,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=torch.hann_window(n_fft),
        )
        return x.squeeze(0).numpy()
    else:
        import librosa

        n_fft = (spec_complex.shape[0] - 1) * 2
        return librosa.istft(
            spec_complex,
            hop_length=hop_length,
            win_length=n_fft,
            window="hann",
            center=True,
        )


def magphase(spec_complex: np.ndarray):
    """
    Decompose a complex STFT matrix into magnitude and phase.

    Returns:
      - magnitude: np.abs(spec_complex)
      - phase:     np.angle(spec_complex)

    Keeping this logic in a dedicated helper ensures that both preprocessing
    and inference use the exact same definition of "magnitude" and "phase".
    """
    return np.abs(spec_complex), np.angle(spec_complex)


def collect_files(
    directory: Path, extensions: List[str] = (".wav", ".mp3", ".flac")
) -> List[Path]:
    """
    Recursively collect audio files with given extensions.

    Args:
      directory:  Root directory to search under.
      extensions: Tuple/list of file suffixes to include.

    Returns:
      Sorted list of Path objects, which keeps ordering stable for any
      subsequent index-based selection (e.g., sample_index in eval).
    """
    p = Path(directory)
    out = []
    for ext in extensions:
        out.extend(list(p.rglob(f"*{ext}")))
    return sorted(out)


def set_seed(seed: int = 0) -> None:
    """
    Set seeds for Python's random, numpy, and (if available) torch RNGs.

    This helper is used by both preprocess.py and train.py to make runs
    reproducible, controlling:
      - Example ordering and random choices in preprocessing.
      - Weight initialization and any stochastic operations in PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass
