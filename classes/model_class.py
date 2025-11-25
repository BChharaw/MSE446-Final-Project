# model_class.py
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from classes.utils_audio import stft_np, istft_np, magphase

class SmallDenoiserNetwork(nn.Module):
    """Simple CNN encoder-decoder for speech denoising."""
    def __init__(self, input_channels=1, base_filters=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(base_filters, base_filters * 2, 3, 2, 1), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, 1, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(base_filters, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


class SpeechDenoisingModel:
    """Wraps network, optimizer, and inference methods."""
    def __init__(self, device="cpu", learning_rate=1e-3, checkpoint_path="checkpoints/best.pth"):
        self.device = torch.device(device)
        self.model = SmallDenoiserNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()
        self.checkpoint_path = Path(checkpoint_path)
    def train_step(self, noisy_magnitude, clean_magnitude):
        self.model.train()
        noisy = noisy_magnitude.unsqueeze(1).to(self.device)
        clean = clean_magnitude.unsqueeze(1).to(self.device)
        
        prediction = self.model(noisy)

        # crop clean to match prediction
        if prediction.shape != clean.shape:
            min_freq = min(prediction.shape[2], clean.shape[2])
            min_time = min(prediction.shape[3], clean.shape[3])
            clean = clean[:, :, :min_freq, :min_time]
            prediction = prediction[:, :, :min_freq, :min_time]

        loss = self.loss_function(prediction, clean)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def evaluate_step(self, noisy_magnitude, clean_magnitude):
        #mismatch
        self.model.eval()
        with torch.no_grad():
            noisy = noisy_magnitude.unsqueeze(1).to(self.device)
            clean = clean_magnitude.unsqueeze(1).to(self.device)
            prediction = self.model(noisy)

            if prediction.shape != clean.shape:
                min_freq = min(prediction.shape[2], clean.shape[2])
                min_time = min(prediction.shape[3], clean.shape[3])
                clean = clean[:, :, :min_freq, :min_time]
                prediction = prediction[:, :, :min_freq, :min_time]

            loss = self.loss_function(prediction, clean)
        return loss.item(), prediction.cpu().squeeze(1).numpy()

    def save_checkpoint(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(self.checkpoint_path))

    def load_checkpoint(self):
        self.model.load_state_dict(torch.load(str(self.checkpoint_path), map_location=self.device))
    def infer(self, noisy_waveform, n_fft, hop_length):
        spectrum = stft_np(noisy_waveform, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = magphase(spectrum)
        magnitude_log = np.log1p(magnitude).astype(np.float32)
        input_tensor = torch.from_numpy(magnitude_log).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predicted_log = self.model(input_tensor).cpu().squeeze().numpy()

        predicted_magnitude = np.expm1(predicted_log)

        # --- Align shapes to avoid broadcasting errors ---
        min_freq = min(predicted_magnitude.shape[0], phase.shape[0])
        min_time = min(predicted_magnitude.shape[1], phase.shape[1])
        predicted_magnitude = predicted_magnitude[:min_freq, :min_time]
        phase = phase[:min_freq, :min_time]
        # ----------------------------------------------

        reconstructed = predicted_magnitude * np.exp(1j * phase)
        waveform = istft_np(reconstructed, hop_length=hop_length)

        waveform = waveform / (max(1e-9, np.max(np.abs(waveform))))
        return waveform, predicted_magnitude, phase
