# train.py
import math
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.model_class import SpeechDenoisingModel
from classes.processed_dataset import ProcessedPairDataset
from classes.utils_audio import set_seed

# -----------------------
# Config loading
# -----------------------
CONFIG_PATH = Path("config.json")
with CONFIG_PATH.open("r") as f:
    CONFIG = json.load(f)

GLOBAL_CFG = CONFIG["global"]
PATHS_CFG = CONFIG["paths"]
TRAIN_CFG = CONFIG["train"]
WANDB_CFG = CONFIG.get("wandb", {})

TRAIN_AUDIO_DIR = Path(PATHS_CFG["processed_train_dir"])
VAL_AUDIO_DIR = Path(PATHS_CFG["processed_val_dir"])
CHECKPOINT_PATH = Path(PATHS_CFG["checkpoint_path"])

SAMPLE_RATE = int(GLOBAL_CFG["sample_rate"])
N_FFT = int(GLOBAL_CFG["stft"]["n_fft"])
HOP_LENGTH = int(GLOBAL_CFG["stft"]["hop_length"])

BATCH_SIZE = int(TRAIN_CFG["batch_size"])
EPOCHS = int(TRAIN_CFG["epochs"])
LEARNING_RATE = float(TRAIN_CFG["learning_rate"])
EXPERIMENT_NAME = TRAIN_CFG.get("experiment_name", "run")

PHASE_SUPPORT = TRAIN_CFG.get("phase_support", True)

device_cfg = TRAIN_CFG.get("device", "auto")
if device_cfg == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = device_cfg

# -----------------------
# Optional wandb setup
# -----------------------
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def init_wandb():
    if not WANDB_AVAILABLE:
        return None
    if not WANDB_CFG.get("enabled", False):
        return None

    mode = WANDB_CFG.get("mode", "online")  # "online" | "offline" | "disabled"
    if mode == "disabled":
        return None

    api_key = WANDB_CFG.get("api_key", None)
    if api_key:
        wandb.login(key=api_key)
    else:
        # falls back to WANDB_API_KEY env or existing login
        try:
            wandb.login()
        except Exception:
            # if login fails, silently disable wandb
            return None

    project = WANDB_CFG.get("project", "speech-denoiser")
    entity = WANDB_CFG.get("entity", None)
    tags = WANDB_CFG.get("tags", [])
    notes = WANDB_CFG.get("notes", "")
    group = WANDB_CFG.get("group", None)
    job_type = WANDB_CFG.get("job_type", "train")

    run = wandb.init(
        project=project,
        entity=entity,
        name=EXPERIMENT_NAME,
        group=group,
        job_type=job_type,
        notes=notes,
        tags=tags,
        mode=mode,
        config={
            "train": TRAIN_CFG,
            "global": GLOBAL_CFG,
            "paths": PATHS_CFG,
        },
    )
    return run


# -----------------------
# Dataset creation
# -----------------------
def build_datasets():
    """Build train and val datasets from preprocessed files on disk."""
    clean_train_files = sorted(TRAIN_AUDIO_DIR.glob("clean_*.wav"))
    noisy_train_files = sorted(TRAIN_AUDIO_DIR.glob("noisy_*.wav"))

    clean_val_files = sorted(VAL_AUDIO_DIR.glob("clean_*.wav"))
    noisy_val_files = sorted(VAL_AUDIO_DIR.glob("noisy_*.wav"))

    if not clean_train_files or not noisy_train_files:
        raise RuntimeError(f"No train clean/noisy files in {TRAIN_AUDIO_DIR}")
    if not clean_val_files or not noisy_val_files:
        raise RuntimeError(f"No val clean/noisy files in {VAL_AUDIO_DIR}")

    assert len(clean_train_files) == len(noisy_train_files)
    assert len(clean_val_files) == len(noisy_val_files)

    train_dataset = ProcessedPairDataset(
        clean_files=clean_train_files,
        noisy_files=noisy_train_files,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        return_phase=PHASE_SUPPORT,  # Enable phase return based on config
    )

    val_dataset = ProcessedPairDataset(
        clean_files=clean_val_files,
        noisy_files=noisy_val_files,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        return_phase=PHASE_SUPPORT,  # Enable phase return based on config
    )

    return train_dataset, val_dataset


# -----------------------
# Training loop
# -----------------------
def main():
    set_seed(TRAIN_CFG.get("seed", 0))

    train_dataset, val_dataset = build_datasets()
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SpeechDenoisingModel(
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        checkpoint_path=str(CHECKPOINT_PATH),
        phase_support=PHASE_SUPPORT,
    )

    run = init_wandb()
    best_val = math.inf
    global_step = 0

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # Train
        running_loss = 0.0
        count = 0
        train_bar = tqdm(train_loader, desc="Train", unit="batch")
        if PHASE_SUPPORT:
            for noisy_mag, clean_mag, noisy_phase, clean_phase in train_bar:
                loss = model.train_step_with_phase(noisy_mag, clean_mag, noisy_phase, clean_phase)
                running_loss += loss
                count += 1
                global_step += 1

                avg_loss = running_loss / max(1, count)
                train_bar.set_postfix(loss=f"{loss:.4f}", avg=f"{avg_loss:.4f}")

                if run is not None:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/avg_loss": avg_loss,
                            "epoch": epoch + 1,
                            "step": global_step,
                        }
                    )
        else:
            for noisy_mag, clean_mag in train_bar:
                loss = model.train_step(noisy_mag, clean_mag)
                running_loss += loss
                count += 1
                global_step += 1

                avg_loss = running_loss / max(1, count)
                train_bar.set_postfix(loss=f"{loss:.4f}", avg=f"{avg_loss:.4f}")

                if run is not None:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/avg_loss": avg_loss,
                            "epoch": epoch + 1,
                            "step": global_step,
                        }
                    )

        train_avg = running_loss / max(1, count)
        print(f"Train avg loss: {train_avg:.5f}")

        # Val
        val_loss_sum = 0.0
        val_count = 0
        val_bar = tqdm(val_loader, desc="Val  ", unit="batch")

        with torch.no_grad():
            if PHASE_SUPPORT:
                for noisy_mag, clean_mag, noisy_phase, clean_phase in val_bar:
                    loss, _ = model.evaluate_step_with_phase(noisy_mag, clean_mag, noisy_phase, clean_phase)
                    val_loss_sum += loss
                    val_count += 1
                    val_avg = val_loss_sum / max(1, val_count)
                    val_bar.set_postfix(avg=f"{val_avg:.4f}")
            else:
                for noisy_mag, clean_mag in val_bar:
                    loss, _ = model.evaluate_step(noisy_mag, clean_mag)
                    val_loss_sum += loss
                    val_count += 1
                    val_avg = val_loss_sum / max(1, val_count)
                val_bar.set_postfix(avg=f"{val_avg:.4f}")

        val_avg = val_loss_sum / max(1, val_count)
        print(f"Val avg loss:  {val_avg:.5f}")

        if run is not None:
            wandb.log(
                {
                    "val/loss": val_avg,
                    "epoch": epoch + 1,
                    "step": global_step,
                }
            )

        if val_avg < best_val:
            best_val = val_avg
            model.save_checkpoint()
            print(f"New best model saved (val={best_val:.5f})")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val:.5f}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()