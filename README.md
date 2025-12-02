# Speech Denoising Project

This project trains a machine-learning model to remove background noise from speech audio clips.
All configuration is handled through a single `config.json` for consistent preprocessing, training, and evaluation.

---

## 1. Project Structure

```
.
├── config.json
├── preprocess.py
├── train.py
├── eval.py
├── classes/
│   ├── model_class.py
│   ├── processed_dataset.py
│   └── utils_audio.py
├── data_raw/
│   ├── speech/        # raw speech recordings
│   └── noise/         # raw noise clips
├── data_processed/
│   ├── train/audio/   # generated clean/noisy training pairs
│   └── val/audio/     # generated clean/noisy validation pairs
├── checkpoints/
│   └── best.pth       # saved model
└── evaluation/
    └── results/       # denoised audio outputs
```

---

## 2. Configuration

All parameters live in:

```
config.json
```

You can modify:

- noise mixing settings
- STFT parameters
- number of examples
- training hyperparameters
- evaluation device
- which validation file to denoise

---

## 3. Preprocessing — Build Dataset

Generate noisy/clean training pairs using:

```bash
python3 preprocess.py
```

This:

1. Loads raw speech + noise from `data_raw/`
2. Randomly crops clean segments
3. Mixes noise with SNR/gain rules in `config.json`
4. Splits into train/val sets
5. Saves:
   - `data_processed/train/audio/clean_XXXXX.wav`
   - `data_processed/train/audio/noisy_XXXXX.wav`
   - … and matching val files
6. Produces preview spectrograms in:
   `visualizations/preprocess/spectrograms/`

---

## 4. Training

Train the denoising model:

```bash
python3 train.py
```

This:

- Loads processed train and val sets
- Converts wav → STFT magnitude
- Trains for a specified number of epochs
- Tracks validation loss
- Saves the best model to:

  ```
  checkpoints/best.pth
  ```

Training progress uses tqdm progress bars.

---

## 5. Evaluation

Denoise a noisy clip from the validation set:

```bash
python3 eval.py
```

This:

- Loads noisy validation sample (chosen via `sample_index` in `config.json`)
- Runs one or more denoising passes (`num_passes`)
- Saves outputs to:

  ```
  evaluation/results/denoised_sample###_pass#.wav
  ```

---

## 6. Quick Start

1. Add speech clips to `data_raw/speech/`
2. Add noise clips to `data_raw/noise/`
3. Adjust `config.json` as needed
4. Run preprocessing

   ```bash
   python3 preprocess.py
   ```

5. Train

   ```bash
   python3 train.py
   ```

6. Evaluate

   ```bash
   python3 eval.py
   ```

---

## 7. Notes

- All scripts rely on the same STFT parameters to maintain consistency.
- Multipass inference helps remove residual noise.
- No command-line arguments are needed; the JSON config controls everything.
- For wandb, pip install wandb, login to your wandb account (in browser), copy your API key and then wandb login in terminal.
