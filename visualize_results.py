# visualize_results.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from utils_audio import stft_np, collect_files
from classes.model_class import SpeechDenoisingModel
from classes.datasets_manipulation import SpeechNoiseDataset
from torch.utils.data import DataLoader

SPEECH_DIR = Path("data/speech")
NOISE_DIR = Path("data/noise")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
CLIP_SECONDS = 2.0
BATCH_SIZE = 1
EPOCHS = 2
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = Path("checkpoints/best.pth")

# Dataset behavior
SPEECH_CLIPS_PER_FILE = 200
NOISE_PROBABILITY = 0.6
MAX_NOISE_OVERLAYS = 3
SNR_RANGE = (-10.0, 5.0)
GAIN_RANGE = (0.3, 1.5)

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
        

def main():
    visualizer = ResultVisualizer()
    model = SpeechDenoisingModel(DEVICE, LEARNING_RATE, CHECKPOINT_PATH)
    model.load_checkpoint()
    speech_files = collect_files(SPEECH_DIR)
    noise_files = collect_files(NOISE_DIR)
    test_dataset = SpeechNoiseDataset(
    speech_files[:2], noise_files, SAMPLE_RATE, CLIP_SECONDS, SNR_RANGE, GAIN_RANGE,
    max(1, SPEECH_CLIPS_PER_FILE // 2), NOISE_PROBABILITY, MAX_NOISE_OVERLAYS, N_FFT, HOP_LENGTH
 )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    noisy_waveform = None
    clean_waveform = None
    # Grab the first sample from the test loader (batch_size==1 by default here)
    for i, (mag_noisy, mag_clean, noisy_batch, clean_batch) in enumerate(test_loader):
        # take only the first example in the batch
        if isinstance(noisy_batch, torch.Tensor):
            noisy_waveform = noisy_batch[0].cpu().numpy()
        else:
            noisy_waveform = np.asarray(noisy_batch[0])
        if isinstance(clean_batch, torch.Tensor):
            clean_waveform = clean_batch[0].cpu().numpy()
        else:
            clean_waveform = np.asarray(clean_batch[0])
        break

    if noisy_waveform is None:
        raise RuntimeError("No test samples available to visualize. Is your dataset empty?")

    # model.infer returns (waveform, predicted_magnitude, phase)
    estimated_waveform, _, _ = model.infer(noisy_waveform, N_FFT, HOP_LENGTH)
    print("Visualizing results...")
    visualizer.compare(noisy_waveform, estimated_waveform, clean_waveform)
    print("Figures saved to:", visualizer.output_dir)

if __name__ == "__main__":
    main()
