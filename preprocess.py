# preprocess.py
import random
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from classes.utils_audio import (
    collect_files,
    load_wav,
    save_wave,
    set_seed,
    stft_np,
    magphase,
)

# -----------------------
# Config loading
# -----------------------
CONFIG_PATH = Path("config.json")
with CONFIG_PATH.open("r") as f:
    CONFIG = json.load(f)

GLOBAL_CFG = CONFIG["global"]
PATHS_CFG = CONFIG["paths"]
PRE_CFG = CONFIG["preprocess"]
TRAIN_CFG = CONFIG["train"]   # for train/val split

# Raw data locations
RAW_SPEECH_DIR = Path(PATHS_CFG["raw_speech_dir"]).expanduser()
RAW_NOISE_DIR = Path(PATHS_CFG["raw_noise_dir"]).expanduser()

# Output locations (already point to audio folders)
OUT_TRAIN_AUDIO_DIR = Path(PATHS_CFG["processed_train_dir"])
OUT_VAL_AUDIO_DIR = Path(PATHS_CFG["processed_val_dir"])
OUT_SPEC_DIR = Path(PATHS_CFG["preprocess_vis_dir"])

# Audio / STFT settings
SAMPLE_RATE = int(GLOBAL_CFG["sample_rate"])
N_FFT = int(GLOBAL_CFG["stft"]["n_fft"])
HOP_LENGTH = int(GLOBAL_CFG["stft"]["hop_length"])

CLIP_SECONDS = float(PRE_CFG["clip_seconds"])
CLIP_LEN = int(SAMPLE_RATE * CLIP_SECONDS)

# Noise mixing settings
SNR_RANGE = tuple(PRE_CFG["snr_range_db"])        # (-5.0, 5.0)
GAIN_RANGE = tuple(PRE_CFG["gain_range"])         # (0.1, 0.5)
NOISE_PROBABILITY = float(PRE_CFG["noise_probability"])
MAX_NOISE_OVERLAYS = int(PRE_CFG["max_noise_overlays"])

# Dataset size / visualization
NUM_EXAMPLES = int(PRE_CFG["num_examples"])
NUM_VIS_SAMPLES = int(PRE_CFG["num_vis_samples"])

# Train/val split used to route files
TRAIN_SPLIT = float(TRAIN_CFG["train_split"])


# -----------------------
# Helper functions
# -----------------------

def random_crop(signal: np.ndarray, clip_len: int) -> np.ndarray:
    """Randomly crop/pad a 1D signal to length clip_len."""
    if signal is None or len(signal) == 0:
        return np.zeros(clip_len, dtype=np.float32)

    if len(signal) < clip_len:
        signal = np.pad(signal, (0, clip_len - len(signal)))

    start = random.randint(0, max(0, len(signal) - clip_len))
    return signal[start:start + clip_len]


def mix_by_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Return clean + scaled-noise such that resulting SNR is snr_db."""
    signal_power = np.mean(clean ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return clean + noise * scale


def overlay_noise(clean: np.ndarray, noise_files: list[Path]) -> np.ndarray:
    """
    Overlay one or more noise clips on top of 'clean' with random SNR or gain.
    Uses MP3/WAV/FLAC noise files from RAW_NOISE_DIR.
    """
    if not noise_files or random.random() > NOISE_PROBABILITY:
        return clean.copy()

    output = clean.copy()
    overlays = random.randint(1, MAX_NOISE_OVERLAYS)

    for _ in range(overlays):
        noise_path = random.choice(noise_files)
        noise = load_wav(noise_path, target_sr=SAMPLE_RATE)

        if len(noise) == 0:
            continue

        # crop noise to at most CLIP_LEN
        if len(noise) > CLIP_LEN:
            start = random.randint(0, len(noise) - CLIP_LEN)
            noise = noise[start:start + CLIP_LEN]

        # random offset for inserting noise
        max_offset = max(0, CLIP_LEN - len(noise))
        offset = random.randint(0, max_offset)
        end = offset + len(noise)

        # choose mixing method
        if random.random() < 0.5:
            snr = random.uniform(*SNR_RANGE)
            mixed = mix_by_snr(output[offset:end], noise, snr)
        else:
            gain = random.uniform(*GAIN_RANGE)
            mixed = output[offset:end] + noise * gain

        output[offset:end] = mixed

    return np.clip(output, -1.0, 1.0).astype(np.float32)


def save_side_by_side_spectrogram(
    clean_mag: np.ndarray,
    noisy_mag: np.ndarray,
    out_path: Path,
) -> None:
    """
    Save a 1x2 figure showing clean (left) vs noisy (right) spectrograms.
    Input magnitudes should already be log-scaled (e.g., log1p).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].imshow(clean_mag, origin="lower", aspect="auto")
    axes[0].set_title("Clean")
    axes[0].set_xlabel("Time frames")
    axes[0].set_ylabel("Frequency bins")

    axes[1].imshow(noisy_mag, origin="lower", aspect="auto")
    axes[1].set_title("Noisy")
    axes[1].set_xlabel("Time frames")
    axes[1].set_ylabel("Frequency bins")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


# -----------------------
# Main
# -----------------------

def main():
    set_seed(PRE_CFG.get("seed", 0))

    OUT_TRAIN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_VAL_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SPEC_DIR.mkdir(parents=True, exist_ok=True)

    # Collect raw files
    speech_files = collect_files(RAW_SPEECH_DIR, extensions=(".wav", ".mp3", ".flac"))
    noise_files = collect_files(RAW_NOISE_DIR, extensions=(".wav", ".mp3", ".flac"))

    if not speech_files:
        raise RuntimeError(f"No speech files found in {RAW_SPEECH_DIR}")
    if not noise_files:
        raise RuntimeError(f"No noise files found in {RAW_NOISE_DIR}")

    # Load all speech files to memory (you likely have only 1 long podcast to start)
    speech_waveforms: list[np.ndarray] = []
    for p in tqdm(speech_files, desc="Loading speech files"):
        w = load_wav(p, target_sr=SAMPLE_RATE)
        if len(w) > 0:
            speech_waveforms.append(w.astype(np.float32))

    if not speech_waveforms:
        raise RuntimeError("All speech files appear empty after loading.")

    print(f"Found {len(speech_waveforms)} speech file(s) and {len(noise_files)} noise file(s).")
    print(f"Generating {NUM_EXAMPLES} clean/noisy pairs into train/val splits...")

    vis_saved = 0
    train_count = 0
    val_count = 0

    for _ in tqdm(range(NUM_EXAMPLES), desc="Preprocessing", unit="clip"):
        # choose a random speech source file and crop a chunk
        speech_src = random.choice(speech_waveforms)
        clean = random_crop(speech_src, CLIP_LEN)
        noisy = overlay_noise(clean, noise_files)

        # Decide whether this example goes to train or val
        if random.random() < TRAIN_SPLIT:
            out_dir = OUT_TRAIN_AUDIO_DIR
            idx = train_count
            train_count += 1
        else:
            out_dir = OUT_VAL_AUDIO_DIR
            idx = val_count
            val_count += 1

        clean_path = out_dir / f"clean_{idx:05d}.wav"
        noisy_path = out_dir / f"noisy_{idx:05d}.wav"
        save_wave(clean, clean_path, sample_rate=SAMPLE_RATE)
        save_wave(noisy, noisy_path, sample_rate=SAMPLE_RATE)

        # For the first NUM_VIS_SAMPLES, save side-by-side spectrograms
        if vis_saved < NUM_VIS_SAMPLES:
            spec_clean = stft_np(clean, n_fft=N_FFT, hop_length=HOP_LENGTH)
            spec_noisy = stft_np(noisy, n_fft=N_FFT, hop_length=HOP_LENGTH)

            mag_clean, _ = magphase(spec_clean)
            mag_noisy, _ = magphase(spec_noisy)

            mag_clean_log = np.log1p(mag_clean)
            mag_noisy_log = np.log1p(mag_noisy)

            vis_path = OUT_SPEC_DIR / f"sample_{vis_saved:02d}.png"
            save_side_by_side_spectrogram(mag_clean_log, mag_noisy_log, vis_path)

            vis_saved += 1

    print("Preprocessing complete.")
    print(f"Train pairs saved to: {OUT_TRAIN_AUDIO_DIR} (count={train_count})")
    print(f"Val pairs saved to:   {OUT_VAL_AUDIO_DIR} (count={val_count})")
    print(f"Spectrogram samples saved to: {OUT_SPEC_DIR}")


if __name__ == "__main__":
    main()