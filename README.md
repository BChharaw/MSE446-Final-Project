# Speech Denoising Project

This project trains a machine-learning model to remove background noise from speech audio clips.
All configuration is handled through a single `config.json` for consistent preprocessing, training, and evaluation.

---
## Setup — Clone, Virtual Environment, Dependencies

### 1. Clone the repository

```bash
git clone https://github.com/BChharaw/MSE446-Final-Project.git
cd MSE446-Final-Project
```

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell)**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Download datasets and place unzipped files under 
```
├── data_raw/
│   ├── speech/        # raw speech recordings
│   └── noise/         # raw noise clips
```

For ESC 50 (noise files) we used the entire dataset, and for LibriSpeech we used the train-clean-100 subset.
Datasets can be found by following the instructions at https://github.com/karolpiczak/ESC-50 and https://www.openslr.org/12 respectively. 

A small toy dataset is already included such that the model can be run out of the box however for the results in our report we use the above linked datasets. Routing to pull from a different folder can be found in config.json

### 5. Done!
You're ready to run our 3 main scripts as described in the readme below:
(1) preprocess.py (2) train.py (3) eval.py
All edits, if any, should be made to config.json as 1, 2, and 3 are designed to run without and user file edits

Note we use wandb so wandb login {insert wandb api key} may be required in terminal if don't have it setup already on your local machine. This is simple as creating an account, and pasting the API key found in settings.
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
Selecting model type can be done in config.json. Choose from "small", "temporal", "rnn", "rnn-2" (2 LSTM layers), and "unet"
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
Selecting model type can be done in config.json. Choose from "small", "temporal", "rnn", "rnn-2" (2 LSTM layers), and "unet"

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
