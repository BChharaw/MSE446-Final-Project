# utils_audio.py
from pathlib import Path
from typing import List
import random
import numpy as np
import torch


import numpy as np
import torch
import soundfile as sf

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
    """
    p = str(path)
    if HAS_TORCHAUDIO:
        waveform, src_sr = torchaudio.load(p)          # shape (channels, samples)
        waveform = waveform.mean(dim=0).numpy()        # mono
        if src_sr != target_sr:
            waveform = torchaudio.functional.resample(
                torch.from_numpy(waveform).unsqueeze(0), src_sr, target_sr
            ).squeeze(0).numpy()
        # normalize to [-1,1]
        mx = np.max(np.abs(waveform)) or 1.0
        return (waveform / mx).astype(np.float32)
    else:
        y, _ = librosa.load(p, sr=target_sr, mono=True)
        mx = np.max(np.abs(y)) or 1.0
        return (y / mx).astype(np.float32)

# def save_wav(path: Path, waveform: np.ndarray, sample_rate: int = 16000) -> None:
#     """Write waveform to disk. Overwrites existing file."""
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     if HAS_TORCHAUDIO:
#         import torch
#         tensor = torch.from_numpy(waveform).unsqueeze(0)
#         torchaudio.save(str(path), tensor, sample_rate)
#     else:
#         sf.write(str(path), waveform, sample_rate)


def save_wave(waveform, path, sample_rate: int = 16000):
    import numpy as np
    import torch
    import soundfile as sf
    from pathlib import Path

    # # print("\n--- save_wave DEBUG ---")
    # # print("Original type:", type(waveform))
    # # print("Original shape:", getattr(waveform, 'shape', 'N/A'))

    # Convert torch tensor to numpy
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
        # # print("Converted from tensor to numpy, shape:", waveform.shape)

    # # Check for NaNs / infs
    # if np.any(np.isnan(waveform)):
    #     # print("WARNING: waveform contains NaNs!")
    # if np.any(np.isinf(waveform)):
    #     # print("WARNING: waveform contains infs!")

    # Flatten all dimensions except last one
    if waveform.ndim > 1:
        waveform = np.reshape(waveform, (-1,))
        # # print("Flattened waveform shape:", waveform.shape)

    # Ensure float32
    waveform = waveform.astype(np.float32)
    # print("Final dtype:", waveform.dtype)

    # Path checks
    path = Path(path)
    # print("Saving to:", path)
    # print("Path parent exists:", path.parent.exists())
    path.parent.mkdir(parents=True, exist_ok=True)
    # print("Path parent created if needed.")

    # Attempt save
    try:
        sf.write(str(path), waveform, samplerate=sample_rate, subtype='PCM_16')
        # print("WAV saved successfully!")
    except Exception as e:
        # print("ERROR saving WAV:", e)
        raise


def stft_np(waveform: np.ndarray, n_fft: int = 512, hop_length: int = 128):
    """
    Return complex STFT matrix (numpy). Signature matches calls in your code.
    """
    if HAS_TORCHAUDIO:
        import torch
        t = torch.from_numpy(waveform).unsqueeze(0)
        spec = torch.stft(t, n_fft=n_fft, hop_length=hop_length,
                          win_length=n_fft, window=torch.hann_window(n_fft),
                          return_complex=True)
        return spec.squeeze(0).numpy()
    else:
        import librosa
        return librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length,
                            win_length=n_fft, window='hann', center=True)

def istft_np(spec_complex: np.ndarray, hop_length: int = 128):
    """
    Inverse STFT. Returns time-domain numpy waveform.
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
        x = torch.istft(t, n_fft=n_fft, hop_length=hop_length,
                         win_length=n_fft, window=torch.hann_window(n_fft))
        return x.squeeze(0).numpy()
    else:
        import librosa
        n_fft = (spec_complex.shape[0] - 1) * 2
        return librosa.istft(spec_complex, hop_length=hop_length, win_length=n_fft, window='hann', center=True)

def magphase(spec_complex: np.ndarray):
    """Return magnitude and phase (numpy) from complex STFT matrix."""
    return np.abs(spec_complex), np.angle(spec_complex)

def collect_files(directory: Path, extensions: List[str] = ('.wav', '.mp3', '.flac')) -> List[Path]:
    """Recursively collect audio files. Returns sorted list of Path objects."""
    p = Path(directory)
    out = []
    for ext in extensions:
        out.extend(list(p.rglob(f'*{ext}')))
    return sorted(out)

def set_seed(seed: int = 0) -> None:
    """Set python/numpy/random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass