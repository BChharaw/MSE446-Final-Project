import math, random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from utils_audio import stft_np, magphase  # note: we use npy caches instead of load_wav

class SpeechNoiseDataset(Dataset):
    """
    Samples random crops from speech cache files (npy) and overlays noise from cache files.
    Accepts either lists of original file paths or lists of cached .npy paths.
    If a file path ends with .npy it is treated as a cached waveform.
    """

    def __init__(self,
                 speech_files,
                 noise_files,
                 sample_rate,
                 clip_seconds,
                 snr_range,
                 gain_range,
                 clips_per_file,
                 noise_probability,
                 max_noise_overlays,
                 n_fft,
                 hop_length,
                 use_lazy_cache=True):
        # speech_files / noise_files can be list[Path] pointing to .npy caches or original audio paths
        self.speech_files = [Path(p) for p in speech_files]
        self.noise_files = [Path(p) for p in noise_files]
        self.sample_rate = sample_rate
        self.clip_length = int(sample_rate * clip_seconds)
        self.snr_range = snr_range
        self.gain_range = gain_range
        self.clips_per_file = clips_per_file
        self.noise_probability = noise_probability
        self.max_noise_overlays = max_noise_overlays
        self.n_fft = n_fft
        self.hop_length = hop_length

        # lazy cache: map Path -> np.ndarray loaded on first use (keeps memory lower)
        self.use_lazy_cache = use_lazy_cache
        self._waveform_cache = {}  # Path -> np.ndarray

    def __len__(self):
        return max(1, len(self.speech_files) * self.clips_per_file)

    def _load_waveform(self, p: Path):
        # if path points to saved .npy produced by preprocess_datasets.py, use np.load
        if p.suffix == '.npy':
            if p not in self._waveform_cache:
                arr = np.load(str(p))
                self._waveform_cache[p] = arr.astype(np.float32)
            return self._waveform_cache[p]
        # fallback: try to load via utils_audio.load_wav if original audio path used
        from utils_audio import load_wav
        arr = load_wav(p, self.sample_rate)  # load_wav returns numpy array here
        return arr

    # (keep other helper functions same: random_crop, mix_by_snr, overlay_noises)
    # replace calls to load_wav(...) with self._load_waveform(...)

    def random_crop(self, signal):
        if signal is None:
            return np.zeros(self.clip_length, dtype=np.float32)
        if len(signal) < self.clip_length:
            signal = np.pad(signal, (0, self.clip_length - len(signal)))
        start = random.randint(0, max(0, len(signal) - self.clip_length))
        return signal[start:start + self.clip_length]

    def mix_by_snr(self, clean, noise, snr_db):
        signal_power = np.mean(clean**2) + 1e-12
        noise_power = np.mean(noise**2) + 1e-12
        scale = math.sqrt(signal_power / (noise_power * 10**(snr_db / 10)))
        return clean + noise * scale

    def overlay_noises(self, clean):
        if not self.noise_files or random.random() > self.noise_probability:
            return clean.copy(), False

        output = clean.copy()
        overlays = random.randint(1, min(self.max_noise_overlays, len(self.noise_files)))

        for _ in range(overlays):
            noise_file = self.noise_files[random.randint(0, len(self.noise_files)-1)]
            noise = self._load_waveform(noise_file)

            if len(noise) > self.clip_length:
                start = random.randint(0, len(noise) - self.clip_length)
                noise = noise[start:start + self.clip_length]

            # shorten random subset of noises
            if random.random() < 0.7 and len(noise) > 1:
                max_len = int(0.8 * self.clip_length)
                new_len = random.randint(int(0.05 * self.clip_length), max_len)
                start = random.randint(0, max(1, len(noise) - new_len))
                noise = noise[start:start + new_len]

            offset = random.randint(0, max(0, self.clip_length - len(noise)))
            end = offset + len(noise)

            if random.random() < 0.5:
                snr = random.uniform(*self.snr_range)
                mixed = self.mix_by_snr(output[offset:end], noise, snr)
            else:
                gain = random.uniform(*self.gain_range)
                mixed = output[offset:end] + noise * gain

            output[offset:end] = mixed

        output = np.clip(output, -1.0, 1.0)
        return output.astype(np.float32), True

    def __getitem__(self, index):
        file_idx = (index // self.clips_per_file) % max(1, len(self.speech_files))
        speech_path = self.speech_files[file_idx]
        speech = self._load_waveform(speech_path)
        clean = self.random_crop(speech)
        noisy, _ = self.overlay_noises(clean)

        spec_noisy = stft_np(noisy, n_fft=self.n_fft, hop_length=self.hop_length)
        spec_clean = stft_np(clean, n_fft=self.n_fft, hop_length=self.hop_length)

        mag_noisy, _ = magphase(spec_noisy)
        mag_clean, _ = magphase(spec_clean)

        mag_noisy = np.log1p(mag_noisy).astype(np.float32)
        mag_clean = np.log1p(mag_clean).astype(np.float32)
        return torch.from_numpy(mag_noisy), torch.from_numpy(mag_clean), noisy, clean