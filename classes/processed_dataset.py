from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from classes.utils_audio import load_wav, magphase, stft_np


class ProcessedPairDataset(Dataset):
    """
    Dataset wrapper for *pre-generated* clean/noisy WAV pairs produced by preprocess.py.

    Design philosophy:
      - All randomness (cropping, mixing, SNR, gains, etc.) happens offline in the
        preprocessing script, so this dataset is deliberately simple and fully
        deterministic.
      - Each item corresponds to a pair of files:
            clean_XXXXX.wav  (target)
            noisy_XXXXX.wav  (input)
      - At access time, both clips are:
            1) Loaded and resampled to a consistent sample_rate.
            2) Transformed to STFT domain (same n_fft / hop_length as training).
            3) Decomposed into magnitude and phase via magphase().
            4) Magnitudes are log-scaled; phases are normalized to [-1, 1] (angle/π).

    Output signature:
      - If return_phase is True:
            (noisy_mag_log, clean_mag_log, noisy_phase_norm, clean_phase_norm)
        All arrays are converted to torch.FloatTensor with shape (freq, time).
      - If return_phase is False:
            (noisy_mag_log, clean_mag_log)
        Phase is discarded, preserving backward compatibility with magnitude-only
        training code.

    This separation keeps the training loop agnostic to how the data is stored
    on disk while ensuring that the spectral representations are consistent
    across preprocessing, training, and evaluation.
    """

    def __init__(
        self,
        clean_files: List[Path],
        noisy_files: List[Path],
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        return_phase: bool = True,
    ):
        assert len(clean_files) == len(noisy_files)
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_phase = return_phase

    def __len__(self) -> int:
        return len(self.clean_files)

    def __getitem__(self, idx: int):
        """
        Load the idx-th clean/noisy pair and return their STFT-domain features.

        Steps:
          1) Read clean/noisy waveforms from disk and resample to sample_rate.
          2) Compute complex STFT for each using identical n_fft / hop_length.
          3) Split into magnitude and complex phase via magphase().
          4) Apply log1p to magnitudes to compress dynamic range and stabilize
             training (values are stored as float32).
          5) Normalize phase (angle) from [-π, π] to [-1, 1] for easier learning
             and consistent scaling with other channels.
        """
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]

        # Load waveforms at a consistent sample rate; any format-specific details
        # are handled inside load_wav (e.g., mono conversion, scaling).
        clean = load_wav(clean_path, target_sr=self.sample_rate)
        noisy = load_wav(noisy_path, target_sr=self.sample_rate)

        # STFT into complex time-frequency domain, shared config with rest of pipeline.
        spec_clean = stft_np(clean, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_noisy = stft_np(noisy, n_fft=self.n_fft, hop_length=self.hop_length)

        # magphase() returns magnitude and complex phase representation.
        mag_clean, phase_clean = magphase(spec_clean)
        mag_noisy, phase_noisy = magphase(spec_noisy)

        # Log-scale magnitude to compress large dynamic ranges. log1p keeps
        # silence (0) mapped exactly to 0 and avoids log(0).
        mag_clean_log = np.log1p(mag_clean).astype(np.float32)
        mag_noisy_log = np.log1p(mag_noisy).astype(np.float32)

        # Normalize phase from [-π, π] to [-1, 1] to keep values in a compact,
        # symmetric range; this makes it more suitable as a network target.
        phase_clean_norm = (np.angle(phase_clean) / np.pi).astype(np.float32)
        phase_noisy_norm = (np.angle(phase_noisy) / np.pi).astype(np.float32)

        if self.return_phase:
            # Return: noisy_mag, clean_mag, noisy_phase, clean_phase
            # Shape for each: (freq_bins, time_frames) as float32 tensors.
            return (
                torch.from_numpy(mag_noisy_log),
                torch.from_numpy(mag_clean_log),
                torch.from_numpy(phase_noisy_norm),
                torch.from_numpy(phase_clean_norm),
            )
        else:
            # Backward compatibility: only return magnitudes (no phase channels).
            return torch.from_numpy(mag_noisy_log), torch.from_numpy(mag_clean_log)
