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
    """

    def __init__(
        self,
        clean_files: List[Path],
        noisy_files: List[Path],
        sample_rate: int,
        n_fft: int,
        hop_length: int,
    ):
        assert len(clean_files) == len(noisy_files)
        self.clean_files = clean_files
        self.noisy_files = noisy_files
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self) -> int:
        return len(self.clean_files)

    def __getitem__(self, idx: int):
        clean_path = self.clean_files[idx]
        noisy_path = self.noisy_files[idx]

        clean = load_wav(clean_path, target_sr=self.sample_rate)
        noisy = load_wav(noisy_path, target_sr=self.sample_rate)

        spec_clean = stft_np(clean, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_noisy = stft_np(noisy, n_fft=self.n_fft, hop_length=self.hop_length)

        mag_clean, _ = magphase(spec_clean)
        mag_noisy, _ = magphase(spec_noisy)

        mag_clean = np.log1p(mag_clean).astype(np.float32)
        mag_noisy = np.log1p(mag_noisy).astype(np.float32)

        # DataLoader will stack to (B, F, T); SpeechDenoisingModel adds channel dim
        return torch.from_numpy(mag_noisy), torch.from_numpy(mag_clean)
