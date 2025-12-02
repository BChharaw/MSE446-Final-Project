# train.py
import json
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.model_class import SpeechDenoisingModel
from classes.processed_dataset import ProcessedPairDataset
from classes.utils_audio import set_seed

# ============================================================================
# Configuration loading and global setup
# ============================================================================
# This block loads *all* experiment configuration from a single JSON file.
# The intent is to ensure that preprocessing, training, evaluation, and logging
# are all driven by identical hyperparameters and paths, avoiding subtle
# mismatches (e.g., training with a different STFT than preprocessing).
#
# The config is intentionally split into logical subsections:
#   - global : signal processing parameters (sample rate, STFT settings, etc.)
#   - paths  : filesystem layout (processed data, checkpoints)
#   - train  : optimization and training-time behavior
#   - wandb  : experiment tracking (optional)
# ============================================================================
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
CHECKPOINT_INTERVAL = int(
    TRAIN_CFG.get("checkpoint_interval", 5)
)  # Save checkpoint every N epochs (purely time-based, not performance-based)

MODEL_TYPE = TRAIN_CFG.get("model_type", "small")

# Signal-processing parameters *must* match those used during preprocessing.
# Any mismatch here would silently corrupt the training data representation.
SAMPLE_RATE = int(GLOBAL_CFG["sample_rate"])
N_FFT = int(GLOBAL_CFG["stft"]["n_fft"])
HOP_LENGTH = int(GLOBAL_CFG["stft"]["hop_length"])

# Core optimization hyperparameters
BATCH_SIZE = int(TRAIN_CFG["batch_size"])
EPOCHS = int(TRAIN_CFG["epochs"])
LEARNING_RATE = float(TRAIN_CFG["learning_rate"])
EXPERIMENT_NAME = TRAIN_CFG.get("experiment_name", "run")

# Whether the dataset/model exposes phase information in addition to magnitude.
# When enabled, the model is trained with explicit phase supervision rather than
# reconstructing phase implicitly or reusing noisy phase at inference.
PHASE_SUPPORT = TRAIN_CFG.get("phase_support", True)

# Device selection logic:
#   - "auto" resolves to CUDA if available, otherwise CPU
#   - explicit device strings (e.g., "cpu", "cuda") bypass auto-detection
device_cfg = TRAIN_CFG.get("device", "auto")
if device_cfg == "auto":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = device_cfg
print(f"Using device: {DEVICE}")

# ============================================================================
# Optional Weights & Biases (wandb) integration
# ============================================================================
# wandb is treated as a *non-critical* dependency. If it is not installed,
# misconfigured, or disabled via config, training proceeds normally without
# any logging, which is deliberate to keep the training loop robust.
# ============================================================================
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def init_wandb():
    """
    Initialize a wandb run if:
      - wandb is installed,
      - logging is enabled in config,
      - and authentication succeeds.

    On any failure path, this function returns None and training continues
    without experiment tracking. The rest of the code checks for `run is not None`
    before attempting to log, so wandb is never a hard dependency.
    """
    if not WANDB_AVAILABLE:
        return None
    if not WANDB_CFG.get("enabled", False):
        return None

    mode = WANDB_CFG.get("mode", "online")  # "online" | "offline" | "disabled"
    if mode == "disabled":
        return None

    # Authentication priority:
    #   1. Explicit API key in config
    #   2. WANDB_API_KEY environment variable
    #   3. Existing local login
    api_key = WANDB_CFG.get("api_key", None)
    if api_key:
        wandb.login(key=api_key)
    else:
        try:
            wandb.login()
        except Exception:
            # If login fails (e.g., no internet / credentials),
            # disable wandb silently to avoid crashing training.
            return None

    project = WANDB_CFG.get("project", "speech-denoiser")
    entity = WANDB_CFG.get("entity", None)
    tags = WANDB_CFG.get("tags", [])
    notes = WANDB_CFG.get("notes", "")
    group = WANDB_CFG.get("group", None)
    job_type = WANDB_CFG.get("job_type", "train")

    # Log *entire* config blocks so every run is fully reproducible from wandb
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


# ============================================================================
# Dataset construction
# ============================================================================
def build_datasets():
    """
    Construct training and validation datasets from *preprocessed* audio files.

    Important design choice:
    - There is NO on-the-fly mixing or augmentation here.
    - All clean/noisy pairs are assumed to have been generated offline and saved
      to disk by preprocess.py. This keeps training deterministic and fast, and
      ensures exact reproducibility across runs.
    """
    clean_train_files = sorted(TRAIN_AUDIO_DIR.glob("clean_*.wav"))
    noisy_train_files = sorted(TRAIN_AUDIO_DIR.glob("noisy_*.wav"))

    clean_val_files = sorted(VAL_AUDIO_DIR.glob("clean_*.wav"))
    noisy_val_files = sorted(VAL_AUDIO_DIR.glob("noisy_*.wav"))

    # Early failure if preprocessing did not produce expected outputs.
    if not clean_train_files or not noisy_train_files:
        raise RuntimeError(f"No train clean/noisy files in {TRAIN_AUDIO_DIR}")
    if not clean_val_files or not noisy_val_files:
        raise RuntimeError(f"No val clean/noisy files in {VAL_AUDIO_DIR}")

    # Enforce strict one-to-one pairing between clean and noisy signals.
    assert len(clean_train_files) == len(noisy_train_files)
    assert len(clean_val_files) == len(noisy_val_files)

    train_dataset = ProcessedPairDataset(
        clean_files=clean_train_files,
        noisy_files=noisy_train_files,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        return_phase=PHASE_SUPPORT,  # Controls dataset output signature
    )

    val_dataset = ProcessedPairDataset(
        clean_files=clean_val_files,
        noisy_files=noisy_val_files,
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        return_phase=PHASE_SUPPORT,
    )

    return train_dataset, val_dataset


# ============================================================================
# Training / validation loop
# ============================================================================
def main():
    # Fix all relevant RNG sources (Python, NumPy, Torch) for reproducibility.
    # This ensures consistent data ordering, initialization, and results.
    set_seed(TRAIN_CFG.get("seed", 0))

    train_dataset, val_dataset = build_datasets()
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    # DataLoaders handle batching and shuffling only.
    # All heavy signal processing already happened offline.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # SpeechDenoisingModel encapsulates:
    #   - Network creation (based on model_type)
    #   - Optimizer and scheduler
    #   - Training and evaluation steps
    #   - Checkpoint save/load logic
    model = SpeechDenoisingModel(
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        model_type=MODEL_TYPE,
        checkpoint_path=str(CHECKPOINT_PATH),
        phase_support=PHASE_SUPPORT,
    )

    run = init_wandb()
    best_val = math.inf  # Track best validation loss seen so far
    global_step = 0      # Monotonic step counter across epochs

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")

        # -----------------------
        # Training phase
        # -----------------------
        running_loss = 0.0
        count = 0
        train_bar = tqdm(train_loader, desc="Train", unit="batch")

        # Two code paths exist solely to accommodate different dataset signatures
        # depending on PHASE_SUPPORT. No other behavior changes.
        if PHASE_SUPPORT:
            for noisy_mag, clean_mag, noisy_phase, clean_phase in train_bar:
                loss = model.train_step_with_phase(
                    noisy_mag, clean_mag, noisy_phase, clean_phase
                )
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

        # -----------------------
        # Validation phase
        # -----------------------
        val_loss_sum = 0.0
        val_count = 0
        val_bar = tqdm(val_loader, desc="Val  ", unit="batch")

        # Gradients are disabled to reduce memory usage and enforce evaluation mode.
        with torch.no_grad():
            if PHASE_SUPPORT:
                for noisy_mag, clean_mag, noisy_phase, clean_phase in val_bar:
                    loss, _ = model.evaluate_step_with_phase(
                        noisy_mag, clean_mag, noisy_phase, clean_phase
                    )
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

        # Save the best-performing model purely based on validation loss.
        if val_avg < best_val:
            best_val = val_avg
            model.save_checkpoint()
            print(f"New best model saved (val={best_val:.5f})")

        # Periodic checkpointing independent of validation performance.
        # This is useful for debugging and long runs where recovery is needed.
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = Path(f"checkpoints/epoch_{epoch}_loss_{val_avg:.5f}.pth")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_checkpoint(checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val:.5f}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
