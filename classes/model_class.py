from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from classes.utils_audio import istft_np, magphase, stft_np


class SmallDenoiserNetwork(nn.Module):
    """
    Simple CNN encoder-decoder for speech denoising.

    Conceptually:
      - Treats the log-magnitude spectrogram as a 2D "image" (freq × time).
      - Encoder: progressively extracts higher-level local patterns via Conv2d.
      - Middle: shallow bottleneck to mix information without further downsampling.
      - Decoder: ConvTranspose2d upsamples back to the original resolution and
        produces a single-channel residual/estimate map.

    This is the baseline architecture against which more complex temporal or
    U-Net-like models can be compared.
    """

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
    """
    CNN with temporal convolutions for speech denoising.

    Compared to SmallDenoiserNetwork, this variant adds explicit temporal
    context via 1×k convolutions in the time dimension after the encoder. The
    goal is to allow the network to model dependencies across multiple frames
    without changing the frequency resolution.
    """

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
    """
    Hybrid CNN-RNN network for speech denoising with convolutional feature extraction.

    High-level structure:
      - A shallow Conv2d encoder extracts local time-frequency features.
      - For each frequency bin, an LSTM scans across time to capture temporal
        dependencies (effectively modeling each "frequency row" as a sequence).
      - A Conv2d decoder maps the LSTM outputs back to a single-channel
        time-frequency representation.

    This architecture is useful when you want to preserve the 2D structure
    but add explicit sequence modeling along the time axis.
    """

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

        # Convolutional feature extraction produces (batch, base_filters, freq, time)
        conv_features = self.conv_encoder(x)

        # Prepare for RNN: treat each frequency bin as its own sequence over time.
        #   (batch, base_filters, freq, time) -> (batch, freq, time, base_filters)
        #   -> (batch * freq, time, base_filters)
        conv_features = conv_features.permute(
            0, 2, 3, 1
        )  # (batch, freq, time, base_filters)
        conv_features = conv_features.contiguous().view(
            batch_size * freq_bins, time_steps, self.base_filters
        )

        # LSTM processes each (freq) sequence independently over time.
        lstm_out, _ = self.lstm(conv_features)  # (batch * freq, time, hidden_size)

        # Reshape LSTM output back to 2D map:
        #   (batch * freq, time, hidden_size) -> (batch, hidden_size, freq, time)
        lstm_out = lstm_out.view(batch_size, freq_bins, time_steps, self.hidden_size)
        lstm_out = lstm_out.permute(0, 3, 1, 2)  # (batch, hidden_size, freq, time)

        # Convolutional decoder maps back to a single-channel denoised estimate.
        output = self.conv_decoder(lstm_out)  # (batch, 1, freq, time)

        return output


class UNetDenoiserNetwork(nn.Module):
    """
    U-Net architecture for speech denoising with skip connections and attention-like
    behavior via concatenation.

    Key ideas:
      - Encoder path progressively downsamples and increases channels, extracting
        higher-level features while reducing spatial (freq×time) resolution.
      - Decoder path upsamples and concatenates encoder feature maps via skip
        connections, helping preserve fine-grained time-frequency detail.
      - The small depth (2 levels + bottleneck) keeps the model compact yet
        expressive enough for spectral denoising tasks.
    """

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
        # Encoder: save feature maps for skip connections.
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
        """
        Match spatial dimensions of x to target using cropping or padding.

        Because pooling/upsampling chains are not always perfectly invertible
        (e.g., odd sizes, boundary effects), encoder and decoder feature maps
        can end up slightly misaligned by 1–2 pixels. This helper enforces
        shape compatibility for concatenation by:
          - Cropping x when it is larger than target.
          - Padding x symmetrically when it is smaller than target.
        """
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
    """
    High-level wrapper around the underlying neural network, optimizer, and
    audio-domain inference utilities.

    Responsibilities:
      - Instantiate a chosen architecture (small / temporal / rnn / unet).
      - Manage optimizer and loss computation (including frequency-dependent
        weighting and optional magnitude+phase losses).
      - Provide one-step training and evaluation helpers for both magnitude-only
        and magnitude+phase regimes.
      - Implement waveform-level inference: STFT → model → iSTFT.

    This class is deliberately model-agnostic: as long as the underlying
    network maps (batch, C, freq, time) to (batch, C, freq, time), the rest
    of the logic remains unchanged.
    """

    def __init__(
        self,
        device="cpu",
        learning_rate=1e-3,
        model_type="small",
        checkpoint_path="checkpoints/best.pth",
        phase_support=True,
    ):
        self.device = torch.device(device)
        # NOTE: `phase_support` determines how many channels are *used* at
        # inference/training time; architectures themselves are constructed
        # with 1 input channel here and are driven via stacking externally.
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
        """
        Create frequency weights that emphasize low frequencies.

        Rationale:
          - Human speech and perceptual quality are more sensitive to errors
            in low–mid frequencies than in very high frequencies.
          - This method builds an exponentially decaying weight vector from
            low to high frequency and normalizes it so that the *average*
            weight is 1.0 (i.e., global loss scale remains comparable).
        """
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
        """
        Compute MSE loss with frequency weighting.

        This is the core building block for both magnitude-only and
        magnitude+phase loss variants:
          - First compute element-wise MSE.
          - Then multiply by a frequency-dependent weight map.
          - Finally average over all dimensions to get a scalar loss.
        """
        freq_bins = prediction.shape[2]
        weights = self._get_frequency_weights(freq_bins)

        # Compute element-wise MSE
        mse_loss = self.loss_function(prediction, target)

        # Apply frequency weights
        weighted_loss = mse_loss * weights

        # Return mean over all dimensions
        return weighted_loss.mean()

    def train_step(self, noisy_magnitude, clean_magnitude):
        """
        Single training step for magnitude-only denoising.

        Input tensors:
          - noisy_magnitude, clean_magnitude: (batch, freq, time)
        Internally:
          - Add channel dimension → (batch, 1, freq, time)
          - Forward through model.
          - Align shapes (crop to min freq/time) to avoid edge mismatches.
          - Compute frequency-weighted loss and update parameters.
        """
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
        """
        Evaluation step for magnitude-only denoising.

        Behavior mirrors train_step but:
          - Runs under torch.no_grad().
          - Returns both scalar loss and predicted magnitude (for visualization).
        """
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
        """
        Save model weights (state_dict) to the configured checkpoint path.

        Note: Only model parameters are saved; optimizer state is not. This
        keeps checkpoints lightweight and focuses on inference reproducibility.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(checkpoint_path))

    def load_checkpoint(self):
        """
        Load model weights from self.checkpoint_path into the current model.

        map_location ensures compatibility when moving checkpoints between
        devices (e.g., training on GPU and evaluating on CPU).
        """
        self.model.load_state_dict(
            torch.load(str(self.checkpoint_path), map_location=self.device)
        )

    def infer(self, noisy_waveform, n_fft, hop_length):
        """
        Waveform-level inference.

        Pipeline:
          1) Compute STFT of noisy waveform.
          2) Decompose into magnitude and complex phase.
          3) Build input tensor(s) depending on phase_support:
             - If True: magnitude (log1p) and normalized phase are stacked
               into 2 channels and the model is expected to output both.
             - If False: only magnitude is fed and phase is reused from noisy.
          4) Run model, convert outputs back to linear magnitude/phase.
          5) Reconstruct complex STFT and run inverse STFT.
          6) Normalize waveform to [-1, 1] to avoid clipping surprises.
        """
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
        """
        Training step when both magnitude and phase are modeled explicitly.

        Inputs:
          - noisy_magnitude, clean_magnitude: (batch, freq, time)
          - noisy_phase, clean_phase:        (batch, freq, time) in normalized units

        The method stacks magnitude and phase into a 2-channel tensor for both
        input and target, runs the model, and optimizes a joint magnitude+phase
        loss that accounts for the circular nature of phase.
        """
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
        """
        Evaluation step with magnitude+phase.

        Returns:
          - Scalar loss.
          - Predicted magnitude channel for analysis/visualization.
        """
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

        Shapes:
          prediction: (batch, 2, freq, time) - channel 0: magnitude, channel 1: phase
          target:     (batch, 2, freq, time) - channel 0: magnitude, channel 1: phase

        Design choices:
          - Magnitude: standard MSE with frequency weighting (as before).
          - Phase: use circular distance by wrapping differences into [-π, π]
            (via atan2(sin Δ, cos Δ)), then square and weight by frequency.
          - Total loss: linear combination of both terms with configurable
            mag_weight and phase_weight.
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
