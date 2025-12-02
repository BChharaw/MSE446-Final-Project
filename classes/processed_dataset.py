# classes/processed_dataset.py
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from classes.utils_audio import load_wav, stft_np, magphase


class ProcessedPairDataset(Dataset):
    """
    Uses pre-generated clean/noisy wavs from preprocess.py.
    No dynamic mixing/augmentation.
    Returns magnitude and phase information.
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
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]

        clean = load_wav(clean_path, target_sr=self.sample_rate)
        noisy = load_wav(noisy_path, target_sr=self.sample_rate)

        spec_clean = stft_np(clean, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_noisy = stft_np(noisy, n_fft=self.n_fft, hop_length=self.hop_length)

        mag_clean, phase_clean = magphase(spec_clean)
        mag_noisy, phase_noisy = magphase(spec_noisy)

        # Log-scale magnitude
        mag_clean_log = np.log1p(mag_clean).astype(np.float32)
        mag_noisy_log = np.log1p(mag_noisy).astype(np.float32)

        # Normalize phase to [-1, 1] range (phase is in radians, divide by pi)
        phase_clean_norm = (np.angle(phase_clean) / np.pi).astype(np.float32)
        phase_noisy_norm = (np.angle(phase_noisy) / np.pi).astype(np.float32)

        if self.return_phase:
            # Return: noisy_mag, clean_mag, noisy_phase, clean_phase
            return (
                torch.from_numpy(mag_noisy_log),
                torch.from_numpy(mag_clean_log),
                torch.from_numpy(phase_noisy_norm),
                torch.from_numpy(phase_clean_norm),
            )
        else:
            # Backward compatibility: only return magnitudes
            return torch.from_numpy(mag_noisy_log), torch.from_numpy(mag_clean_log)
