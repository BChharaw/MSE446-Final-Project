# model_class.py
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from classes.utils_audio import istft_np, magphase, stft_np


class SmallDenoiserNetwork(nn.Module):
    """Simple CNN encoder-decoder for speech denoising."""

    def __init__(self, input_channels=1, base_filters=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters * 2, 3, 2, 1),
            nn.ReLU(),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, 1, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_filters, 1, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x


class TemporalDenoiserNetwork(nn.Module):
    """CNN with temporal convolutions for speech denoising."""

    def __init__(self, input_channels=1, base_filters=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters * 2, 3, 2, 1),
            nn.ReLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 2, (1, 5), 1, (0, 2)),
            nn.ReLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, (1, 5), 1, (0, 2)),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(base_filters, 1, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.temporal(x)
        x = self.decoder(x)
        return x


class RNNDenoiserNetwork(nn.Module):
    """Hybrid CNN-RNN network for speech denoising with convolutional feature extraction."""

    def __init__(
        self,
        input_channels=1,
        base_filters=16,
        hidden_size=8,
        num_layers=2,
        dropout=0.0,
    ):
        super().__init__()
        self.base_filters = base_filters
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Convolutional encoder for feature extraction
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, 1, 1),
            nn.ReLU(),
        )

        # LSTM for temporal modeling on conv features
        self.lstm = nn.LSTM(
            input_size=base_filters,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Convolutional decoder
        self.conv_decoder = nn.Sequential(
            nn.Conv2d(hidden_size, base_filters, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(base_filters, input_channels, 3, 1, 1),
        )

    def forward(self, x):
        # x shape: (batch, 1, freq, time)
        batch_size, channels, freq_bins, time_steps = x.shape

        # Convolutional feature extraction
        conv_features = self.conv_encoder(x)  # (batch, base_filters, freq, time)

        # Prepare for RNN: process each frequency bin over time
        # Reshape to (batch * freq, time, base_filters)
        conv_features = conv_features.permute(
            0, 2, 3, 1
        )  # (batch, freq, time, base_filters)
        conv_features = conv_features.contiguous().view(
            batch_size * freq_bins, time_steps, self.base_filters
        )

        # LSTM processing
        lstm_out, _ = self.lstm(conv_features)  # (batch * freq, time, hidden_size)

        # Reshape back to spatial format
        # (batch * freq, time, hidden_size) -> (batch, hidden_size, freq, time)
        lstm_out = lstm_out.view(batch_size, freq_bins, time_steps, self.hidden_size)
        lstm_out = lstm_out.permute(0, 3, 1, 2)  # (batch, hidden_size, freq, time)

        # Convolutional decoder
        output = self.conv_decoder(lstm_out)  # (batch, 1, freq, time)

        return output


class UNetDenoiserNetwork(nn.Module):
    """U-Net architecture for speech denoising with skip connections and attention."""

    def __init__(self, input_channels=1, base_filters=16):
        super().__init__()
        self.base_filters = base_filters
        output_channels = 1

        # Encoder path - 2 levels
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, 3, 1, 1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Add dropout
            nn.Conv2d(base_filters, base_filters, 3, 1, 1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
            nn.Dropout2d(0.2),  # Add dropout
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters * 4, 3, 1, 1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(),
            nn.Conv2d(base_filters * 4, base_filters * 4, 3, 1, 1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(),
        )

        # Decoder path
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_filters * 4, base_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
            nn.Conv2d(base_filters * 2, base_filters * 2, 3, 1, 1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(),
        )

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_filters * 2, base_filters, 3, 1, 1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
            nn.Conv2d(base_filters, base_filters, 3, 1, 1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(base_filters, output_channels, 1)

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        x = self.pool1(enc1_out)

        enc2_out = self.enc2(x)
        x = self.pool2(enc2_out)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        x = self.up2(x)
        x = self._match_size(x, enc2_out)
        x = torch.cat([x, enc2_out], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = self._match_size(x, enc1_out)
        x = torch.cat([x, enc1_out], dim=1)
        x = self.dec1(x)

        output = self.final(x)

        return output

    def _match_size(self, x, target):
        """Match spatial dimensions of x to target using cropping or padding."""
        diff_h = x.shape[2] - target.shape[2]
        diff_w = x.shape[3] - target.shape[3]

        if diff_h > 0 or diff_w > 0:
            # Crop x
            h_start = diff_h // 2
            w_start = diff_w // 2
            x = x[
                :,
                :,
                h_start : h_start + target.shape[2],
                w_start : w_start + target.shape[3],
            ]
        elif diff_h < 0 or diff_w < 0:
            # Pad x
            padding = [
                -diff_w // 2 if diff_w < 0 else 0,  # left
                (-diff_w + 1) // 2 if diff_w < 0 else 0,  # right
                -diff_h // 2 if diff_h < 0 else 0,  # top
                (-diff_h + 1) // 2 if diff_h < 0 else 0,  # bottom
            ]
            x = nn.functional.pad(x, padding)

        return x


class SpeechDenoisingModel:
    """Wraps network, optimizer, and inference methods."""

    def __init__(
        self,
        device="cpu",
        learning_rate=1e-3,
        model_type="small",
        checkpoint_path="checkpoints/best.pth",
        phase_support=True,
    ):
        self.device = torch.device(device)
        # Use UNet with 2 input channels and 2 output channels for magnitude + phase
        if model_type == "small":
            self.model = SmallDenoiserNetwork(input_channels=1).to(self.device)
        elif model_type == "temporal":
            self.model = TemporalDenoiserNetwork(input_channels=1).to(self.device)
        elif model_type == "rnn":
            self.model = RNNDenoiserNetwork(input_channels=1, num_layers=1).to(
                self.device
            )
        elif model_type == "rnn-2":
            self.model = RNNDenoiserNetwork(input_channels=1, num_layers=2).to(
                self.device
            )
        elif model_type == "unet":
            self.model = UNetDenoiserNetwork(input_channels=1).to(self.device)
        self.uses_raw_audio = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss(reduction="none")
        self.checkpoint_path = Path(checkpoint_path)
        self.freq_weights = None
        self.mag_weight = 1.0  # Weight for magnitude loss
        self.phase_weight = (
            0.1  # Weight for phase loss (typically lower than magnitude)
        )
        self.phase_support = phase_support  # Indicates if phase is used

    def _get_frequency_weights(self, freq_bins):
        """Create frequency weights that emphasize low frequencies."""
        if self.freq_weights is None or self.freq_weights.shape[0] != freq_bins:
            # Create exponentially decaying weights from low to high frequency
            freq_indices = torch.arange(freq_bins, dtype=torch.float32)
            # Higher weight for lower frequencies (exponential decay)
            weights = torch.exp(-freq_indices / (freq_bins * 0.6))
            # Normalize so average weight is 1.0
            weights = weights / weights.mean()
            # Reshape for broadcasting: (1, 1, freq_bins, 1)
            self.freq_weights = weights.view(1, 1, -1, 1).to(self.device)
        return self.freq_weights

    def _frequency_weighted_loss(self, prediction, target):
        """Compute MSE loss with frequency weighting."""
        freq_bins = prediction.shape[2]
        weights = self._get_frequency_weights(freq_bins)

        # Compute element-wise MSE
        mse_loss = self.loss_function(prediction, target)

        # Apply frequency weights
        weighted_loss = mse_loss * weights

        # Return mean over all dimensions
        return weighted_loss.mean()

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

        # Use frequency-weighted loss
        loss = self._frequency_weighted_loss(prediction, clean)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_step(self, noisy_magnitude, clean_magnitude):
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

            loss = self._frequency_weighted_loss(prediction, clean)
            return loss.item(), prediction.cpu().squeeze(1).numpy()

    def save_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(checkpoint_path))

    def load_checkpoint(self):
        self.model.load_state_dict(
            torch.load(str(self.checkpoint_path), map_location=self.device)
        )

    def infer(self, noisy_waveform, n_fft, hop_length):
        spectrum = stft_np(noisy_waveform, n_fft=n_fft, hop_length=hop_length)
        magnitude, phase = magphase(spectrum)
        magnitude_log = np.log1p(magnitude).astype(np.float32)
        if self.phase_support:
            phase_normalized = (np.angle(phase) / np.pi).astype(np.float32)
            # Create 2-channel input tensor
            input_tensor = (
                torch.from_numpy(np.stack([magnitude_log, phase_normalized]))
                .unsqueeze(0)
                .to(self.device)
            )
        else:
            # Create 1-channel input tensor
            input_tensor = (
                torch.from_numpy(magnitude_log)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(self.device)
            )

        if self.phase_support:  # Model outputs 2 channels: magnitude and phase
            with torch.no_grad():
                prediction = (
                    self.model(input_tensor).cpu().squeeze(0).numpy()
                )  # (2, freq, time)

            # Extract predicted magnitude and phase
            predicted_log_mag = prediction[0]  # (freq, time)
            predicted_phase_norm = prediction[1]  # (freq, time)

            # Convert back from log scale and normalized phase
            predicted_magnitude = np.expm1(predicted_log_mag)
            predicted_phase = predicted_phase_norm * np.pi  # Denormalize phase

            # Align shapes
            min_freq = min(predicted_magnitude.shape[0], predicted_phase.shape[0])
            min_time = min(predicted_magnitude.shape[1], predicted_phase.shape[1])
            predicted_magnitude = predicted_magnitude[:min_freq, :min_time]
            predicted_phase = predicted_phase[:min_freq, :min_time]

            # Reconstruct using predicted magnitude and predicted phase
            reconstructed = predicted_magnitude * np.exp(1j * predicted_phase)
            waveform = istft_np(reconstructed, hop_length=hop_length)

            waveform = waveform / (max(1e-9, np.max(np.abs(waveform))))
            return waveform, predicted_magnitude, predicted_phase

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
        return waveform, predicted_magnitude, None

    def train_step_with_phase(
        self, noisy_magnitude, clean_magnitude, noisy_phase, clean_phase
    ):
        """Training step with phase information."""
        self.model.train()

        # Stack magnitude and phase as 2-channel input and target
        noisy_input = torch.stack([noisy_magnitude, noisy_phase], dim=1).to(
            self.device
        )  # (batch, 2, freq, time)
        clean_target = torch.stack([clean_magnitude, clean_phase], dim=1).to(
            self.device
        )  # (batch, 2, freq, time)

        prediction = self.model(noisy_input)

        # Crop to match if necessary
        if prediction.shape != clean_target.shape:
            min_freq = min(prediction.shape[2], clean_target.shape[2])
            min_time = min(prediction.shape[3], clean_target.shape[3])
            clean_target = clean_target[:, :, :min_freq, :min_time]
            prediction = prediction[:, :, :min_freq, :min_time]

        loss = self._magnitude_phase_loss(prediction, clean_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_step_with_phase(
        self, noisy_magnitude, clean_magnitude, noisy_phase, clean_phase
    ):
        """Evaluation step with phase information."""
        self.model.eval()
        with torch.no_grad():
            # Stack magnitude and phase as 2-channel input and target
            noisy_input = torch.stack([noisy_magnitude, noisy_phase], dim=1).to(
                self.device
            )
            clean_target = torch.stack([clean_magnitude, clean_phase], dim=1).to(
                self.device
            )

            prediction = self.model(noisy_input)

            if prediction.shape != clean_target.shape:
                min_freq = min(prediction.shape[2], clean_target.shape[2])
                min_time = min(prediction.shape[3], clean_target.shape[3])
                clean_target = clean_target[:, :, :min_freq, :min_time]
                prediction = prediction[:, :, :min_freq, :min_time]

            loss = self._magnitude_phase_loss(prediction, clean_target)

            # Return loss and predicted magnitude only for visualization
            return loss.item(), prediction[:, 0, :, :].cpu().numpy()

    def _magnitude_phase_loss(self, prediction, target):
        """
        Compute weighted loss for both magnitude and phase channels.
        prediction: (batch, 2, freq, time) - channel 0: magnitude, channel 1: phase
        target: (batch, 2, freq, time) - channel 0: magnitude, channel 1: phase
        """
        freq_bins = prediction.shape[2]
        weights = self._get_frequency_weights(freq_bins)

        pred_mag = prediction[:, 0:1, :, :]
        pred_phase = prediction[:, 1:2, :, :]

        target_mag = target[:, 0:1, :, :]
        target_phase = target[:, 1:2, :, :]

        # Magnitude loss
        mag_loss = self.loss_function(pred_mag, target_mag)
        mag_loss_weighted = (mag_loss * weights).mean() * self.mag_weight

        # Phase loss with circular distance
        phase_diff = pred_phase - target_phase
        # Wrap to [-π, π]
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        phase_loss = phase_diff**2
        phase_loss_weighted = (phase_loss * weights).mean() * self.phase_weight

        total_loss = mag_loss_weighted + phase_loss_weighted

        return total_loss
