"""
Emotion detection dataset utilities.

Uses the Berlin Database of Emotional Speech (Emo-DB).
Small dataset (~40MB), 10 speakers, 7 emotions, ~500 utterances.

Emotions: anger (W), boredom (L), disgust (E), fear (A),
          happiness (F), sadness (T), neutral (N)
"""

import os
import zipfile
import urllib.request

import numpy as np
from tqdm import tqdm

from speaker_verify.features import audio_to_embedding

# Emo-DB emotion code -> label index + name
EMOTION_CODES = {
    "N": (0, "neutral"),
    "W": (1, "anger"),
    "A": (2, "fear"),
    "F": (3, "happiness"),
    "T": (4, "sadness"),
    "E": (5, "disgust"),
    "L": (6, "boredom"),
}

EMOTION_NAMES = [
    name for _, (_, name) in sorted(EMOTION_CODES.items(), key=lambda x: x[1][0])
]

EMODB_URL = "http://emodb.bilderbar.info/download/download.zip"


def download_emodb(root):
    """
    Download and extract the Emo-DB dataset.

    Args:
        root: Directory to download into.

    Returns:
        Path to the extracted wav directory.
    """
    wav_dir = os.path.join(root, "emodb", "wav")
    if os.path.isdir(wav_dir) and len(os.listdir(wav_dir)) > 0:
        print("Emo-DB already downloaded.")
        return wav_dir

    os.makedirs(os.path.join(root, "emodb"), exist_ok=True)
    zip_path = os.path.join(root, "emodb", "download.zip")

    print("Downloading Emo-DB dataset...")
    urllib.request.urlretrieve(EMODB_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(os.path.join(root, "emodb"))

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print(f"Emo-DB extracted to {wav_dir}")
    return wav_dir


def scan_emodb(wav_dir):
    """
    Scan Emo-DB directory and parse filenames.

    Emo-DB filename format: SSaTTEv.wav
      SS = speaker number (03,08,09,10,11,12,13,14,15,16)
      a  = text code (a or b)
      TT = text number (01-07)
      E  = emotion code (W,L,E,A,F,T,N)
      v  = version (a,b)

    Args:
        wav_dir: Path to Emo-DB wav directory.

    Returns:
        List of (path, emotion_idx, emotion_name, speaker_id) tuples.
    """
    entries = []
    for fname in sorted(os.listdir(wav_dir)):
        if not fname.endswith(".wav"):
            continue

        name = fname[:-4]  # strip .wav
        if len(name) < 6:
            continue

        emotion_code = name[5]
        if emotion_code not in EMOTION_CODES:
            continue

        emotion_idx, emotion_name = EMOTION_CODES[emotion_code]
        speaker_id = name[:2]

        entries.append(
            (
                os.path.join(wav_dir, fname),
                emotion_idx,
                emotion_name,
                speaker_id,
            )
        )

    return entries


def build_emotion_dataset(wav_dir, sr=16000, n_mfcc=20, enhanced=False):
    """
    Build feature dataset from Emo-DB.

    Args:
        wav_dir: Path to Emo-DB wav directory.
        sr: Sample rate.
        n_mfcc: Number of MFCCs.
        enhanced: Use enhanced features.

    Returns:
        X: np.ndarray of shape (n_samples, feat_dim).
        y: np.ndarray of shape (n_samples,) — emotion indices.
        names: List of emotion name strings.
    """
    entries = scan_emodb(wav_dir)

    X_list = []
    y_list = []

    for path, emotion_idx, _, _ in tqdm(entries, desc="Extracting emotion features"):
        emb = audio_to_embedding(path, sr=sr, n_mfcc=n_mfcc, enhanced=enhanced)
        X_list.append(emb)
        y_list.append(emotion_idx)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64),
        EMOTION_NAMES,
    )
