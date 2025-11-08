# visualize_results.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils_audio import stft_np

SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128

class ResultVisualizer:
    def __init__(self, output_dir="figs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_waveform(self, waveform, name):
        time_axis = np.arange(waveform.size) / SAMPLE_RATE
        plt.figure(figsize=(8, 2))
        plt.plot(time_axis, waveform)
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{name}_wave.png")
        plt.close()

    def plot_spectrogram(self, magnitude, name):
        plt.figure(figsize=(8, 3))
        plt.imshow(np.log1p(magnitude)[::-1, :], aspect='auto', origin='lower')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{name}_spec.png")
        plt.close()

    def compare(self, noisy, estimated, clean=None):
        self.plot_waveform(noisy, "noisy")
        self.plot_waveform(estimated, "estimated")
        if clean is not None:
            self.plot_waveform(clean, "clean")
        self.plot_spectrogram(np.abs(stft_np(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH)), "noisy")
        self.plot_spectrogram(np.abs(stft_np(estimated, n_fft=N_FFT, hop_length=HOP_LENGTH)), "estimated")
        