# preprocess_datasets.py
import os
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# -------------------------
# STATIC CONFIG
# -------------------------
SPEECH_DIR = Path("data/speech")
NOISE_DIR = Path("data/noise")
CACHE_DIR = Path("cache")
MANIFESTS_DIR = Path("manifests")
SAMPLE_RATE = 16000
CLIP_SECONDS = 2.0
MIN_NOISE_LENGTH = 0.05  # fraction of clip
MAX_NOISE_LENGTH = 0.8   # fraction of clip
CLIPS_PER_SPEECH_FILE = 200

CACHE_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_audio(path, sr=SAMPLE_RATE):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)

def save_audio(path, y, sr=SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, y, sr)

def chop_speech_file(speech_path, clip_seconds=CLIP_SECONDS, clips_per_file=CLIPS_PER_SPEECH_FILE):
    y = load_audio(speech_path)
    clip_len = int(clip_seconds * SAMPLE_RATE)
    clips = []
    for i in tqdm(range(clips_per_file), desc=f"Chopping {speech_path.name}"):
        if len(y) <= clip_len:
            start = 0
        else:
            start = np.random.randint(0, len(y) - clip_len)
        clip = y[start:start + clip_len]
        clip_path = CACHE_DIR / f"{speech_path.stem}_clip{i}.wav"
        save_audio(clip_path, clip)
        clips.append(clip_path)
    return clips

# -------------------------
# PROCESS DATASETS
# -------------------------
speech_files = list(SPEECH_DIR.glob("*.wav")) + list(SPEECH_DIR.glob("*.mp3"))
noise_files = list(NOISE_DIR.glob("*.wav")) + list(NOISE_DIR.glob("*.mp3"))

speech_manifest = MANIFESTS_DIR / "speech_files.txt"
noise_manifest = MANIFESTS_DIR / "noise_files.txt"

# chop long speech files
all_speech_clips = []
for speech_file in tqdm(speech_files, desc="Processing speech files"):
    clips = chop_speech_file(speech_file)
    all_speech_clips.extend(clips)

# write speech manifest
with open(speech_manifest, "w") as f:
    for clip in all_speech_clips:
        f.write(str(clip) + "\n")

# noise manifest is just the raw files
with open(noise_manifest, "w") as f:
    for nf in noise_files:
        f.write(str(nf) + "\n")

print(f"Prepared {len(all_speech_clips)} speech clips")
print(f"Found {len(noise_files)} noise files")
print("Manifests written to:", MANIFESTS_DIR)