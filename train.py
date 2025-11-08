# train.py
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from classes.model_class import SpeechDenoisingModel
from classes.datasets_manipulation import SpeechNoiseDataset
from utils_audio import collect_files, set_seed

# --------------------
# Config
# --------------------
SPEECH_DIR = Path("data/speech")
NOISE_DIR = Path("data/noise")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128
CLIP_SECONDS = 2.0
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3
CHECKPOINT_PATH = Path("checkpoints/best.pth")

# Dataset behavior
SPEECH_CLIPS_PER_FILE = 200
NOISE_PROBABILITY = 0.6
MAX_NOISE_OVERLAYS = 3
SNR_RANGE = (-10.0, 5.0)
GAIN_RANGE = (0.3, 1.5)

# --------------------
# Main
# --------------------
def main():
    set_seed(0)

    speech_files = collect_files(SPEECH_DIR)
    noise_files = collect_files(NOISE_DIR)

    if not speech_files:
        print(f"No speech files found in {SPEECH_DIR}.")
        return
    if not noise_files:
        print(f"No noise files found in {NOISE_DIR}.")
        return

    random.shuffle(speech_files)
    split = max(1, int(0.9 * len(speech_files)))
    train_files = speech_files[:split]
    val_files = speech_files[split:] or speech_files[:1]  # fallback if split=0

    # Create datasets
    train_dataset = SpeechNoiseDataset(
        train_files, noise_files, SAMPLE_RATE, CLIP_SECONDS, SNR_RANGE, GAIN_RANGE,
        SPEECH_CLIPS_PER_FILE, NOISE_PROBABILITY, MAX_NOISE_OVERLAYS, N_FFT, HOP_LENGTH
    )
    val_dataset = SpeechNoiseDataset(
        val_files, noise_files, SAMPLE_RATE, CLIP_SECONDS, SNR_RANGE, GAIN_RANGE,
        max(1, SPEECH_CLIPS_PER_FILE // 2), NOISE_PROBABILITY, MAX_NOISE_OVERLAYS, N_FFT, HOP_LENGTH
    )

    if len(train_dataset) == 0:
        print("Training dataset is empty. Did you preprocess the speech files?")
        return
    if len(val_dataset) == 0:
        print("Validation dataset is empty. Using training dataset for validation.")
        val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = SpeechDenoisingModel(device=DEVICE, learning_rate=LEARNING_RATE, checkpoint_path=CHECKPOINT_PATH)

    for epoch in range(1, EPOCHS + 1):
        # Training
        total_loss = 0
        for noisy, clean, _, _ in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            loss = model.train_step(noisy, clean)
            total_loss += loss
        avg_train_loss = total_loss / max(1, len(train_loader))

        # Validation
        total_val_loss = 0
        for noisy, clean, _, _ in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            val_loss, _ = model.evaluate_step(noisy, clean)
            total_val_loss += val_loss
        avg_val_loss = total_val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch}: train={avg_train_loss:.6f} val={avg_val_loss:.6f}")

    # Save model
    model.save_checkpoint()
    print(f"Model checkpoint saved to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()