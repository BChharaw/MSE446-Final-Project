# eval_model.py
from pathlib import Path
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from classes.utils_audio import load_wav, save_wave, stft_np, magphase
from classes.model_class import SpeechDenoisingModel

# -----------------------
# Config loading
# -----------------------
CONFIG_PATH = Path("config.json")
with CONFIG_PATH.open("r") as f:
    CONFIG = json.load(f)

GLOBAL_CFG = CONFIG["global"]
PATHS_CFG = CONFIG["paths"]
EVAL_CFG = CONFIG["eval"]

SAMPLE_RATE = int(GLOBAL_CFG["sample_rate"])
N_FFT = int(GLOBAL_CFG["stft"]["n_fft"])
HOP_LENGTH = int(GLOBAL_CFG["stft"]["hop_length"])

VAL_AUDIO_DIR = Path(PATHS_CFG["processed_val_dir"])
OUTPUT_DIR = Path(PATHS_CFG["eval_output_dir"])

NUM_PASSES = int(EVAL_CFG["num_passes"])
SAMPLE_INDEX = int(EVAL_CFG.get("sample_index", 0))

device_cfg = EVAL_CFG.get("device", "cpu")
if device_cfg == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = device_cfg

CHECKPOINT_PATH = Path(PATHS_CFG["checkpoint_path"])


def plot_spectrogram_stack(
    waveforms: list[np.ndarray],
    labels: list[str],
    out_path: Path,
) -> None:
    """
    Plot a vertical stack of spectrograms:
    one row per waveform, same STFT config for all.
    """
    assert len(waveforms) == len(labels)

    num_rows = len(waveforms)
    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(10, 3 * num_rows),
        squeeze=False,
    )

    for i, (wave, label) in enumerate(zip(waveforms, labels)):
        spec = stft_np(wave, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mag, _ = magphase(spec)
        mag_log = np.log1p(mag)

        ax = axes[i, 0]
        im = ax.imshow(mag_log, origin="lower", aspect="auto")
        ax.set_title(label)
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Frequency bins")

        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.02)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    noisy_val_files = sorted(VAL_AUDIO_DIR.glob("noisy_*.wav"))
    if not noisy_val_files:
        raise RuntimeError(f"No noisy val files found in {VAL_AUDIO_DIR}")

    # choose index within range
    idx = min(max(0, SAMPLE_INDEX), len(noisy_val_files) - 1)
    noisy_path = noisy_val_files[idx]

    print(f"Using val noisy sample index {idx}: {noisy_path}")

    # load noisy input
    noisy_wave = load_wav(noisy_path, SAMPLE_RATE)

    # try to load matching clean ground truth if it exists
    clean_path = VAL_AUDIO_DIR / f"clean_{idx:05d}.wav"
    clean_wave = None
    if clean_path.exists():
        clean_wave = load_wav(clean_path, SAMPLE_RATE)
        print(f"Found ground truth clean file: {clean_path}")
    else:
        print("No matching clean_XXXXX.wav found; proceeding without ground truth.")

    # init model
    model = SpeechDenoisingModel(
        device=DEVICE,
        learning_rate=1e-3,
        checkpoint_path=str(CHECKPOINT_PATH),
    )
    model.load_checkpoint()
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    # run multipass denoising
    waveforms_for_plot = []
    labels_for_plot = []

    # add ground truth first if available
    if clean_wave is not None:
        waveforms_for_plot.append(clean_wave)
        labels_for_plot.append("Clean (ground truth)")

    # original noisy input
    waveforms_for_plot.append(noisy_wave)
    labels_for_plot.append("Noisy input")

    current = noisy_wave
    for p in range(NUM_PASSES):
        print(f"Running pass {p+1}/{NUM_PASSES}")
        denoised, _, _ = model.infer(current, N_FFT, HOP_LENGTH)
        current = denoised

        # save audio for this pass
        out_wav = OUTPUT_DIR / f"denoised_sample{idx:03d}_pass{p+1}.wav"
        save_wave(current, out_wav, SAMPLE_RATE)
        print(f"Saved: {out_wav}")

        # store for spectrogram visualization
        waveforms_for_plot.append(current)
        labels_for_plot.append(f"Denoised (pass {p+1})")

    # create stacked spectrogram figure
    spec_path = OUTPUT_DIR / f"spectrogram_sample{idx:03d}.png"
    plot_spectrogram_stack(waveforms_for_plot, labels_for_plot, spec_path)
    print(f"Saved spectrogram stack: {spec_path}")

    print("Done.")


if __name__ == "__main__":
    main()