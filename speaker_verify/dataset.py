"""
Dataset utilities for speaker verification and related tasks.

Handles LibriSpeech download, speaker scanning, pair generation,
gender metadata parsing, and feature extraction for training.
"""

import os
import random

import numpy as np
import torchaudio
from tqdm import tqdm

from speaker_verify.features import audio_to_embedding, pair_features


def download_librispeech(root, subset="test-clean"):
    """
    Download LibriSpeech subset via torchaudio.

    Args:
        root: Directory to download into.
        subset: LibriSpeech subset name (default: "test-clean").

    Returns:
        Path to the downloaded dataset root.
    """
    os.makedirs(root, exist_ok=True)
    print(f"Downloading LibriSpeech {subset}...")
    torchaudio.datasets.LIBRISPEECH(root=root, url=subset, download=True)
    return root


def scan_speakers(root, subset="test-clean"):
    """
    Scan a LibriSpeech directory and map speaker IDs to audio file paths.

    LibriSpeech structure: root/LibriSpeech/test-clean/<speaker>/<chapter>/<file>.flac

    Args:
        root: Dataset root directory.
        subset: Subset name for path construction.

    Returns:
        dict mapping speaker_id (str) -> list of audio file paths.
    """
    speakers = {}
    dataset_dir = os.path.join(root, "LibriSpeech", subset)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    for speaker_id in sorted(os.listdir(dataset_dir)):
        speaker_dir = os.path.join(dataset_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue

        audio_files = []
        for chapter_id in os.listdir(speaker_dir):
            chapter_dir = os.path.join(speaker_dir, chapter_id)
            if not os.path.isdir(chapter_dir):
                continue
            for fname in os.listdir(chapter_dir):
                if fname.endswith(".flac"):
                    audio_files.append(os.path.join(chapter_dir, fname))

        if audio_files:
            speakers[speaker_id] = sorted(audio_files)

    return speakers


def parse_gender_metadata(root):
    """
    Parse LibriSpeech SPEAKERS.TXT to get gender for each speaker.

    Args:
        root: Dataset root directory (contains LibriSpeech/).

    Returns:
        dict mapping speaker_id (str) -> gender ("M" or "F").
    """
    speakers_file = os.path.join(root, "LibriSpeech", "SPEAKERS.TXT")
    if not os.path.exists(speakers_file):
        raise FileNotFoundError(f"SPEAKERS.TXT not found at {speakers_file}")

    gender_map = {}
    with open(speakers_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                speaker_id = parts[0].strip()
                gender = parts[1].strip()
                if gender in ("M", "F"):
                    gender_map[speaker_id] = gender

    return gender_map


def generate_pairs(speakers, n_pairs=2000, seed=42):
    """
    Generate balanced same-speaker / different-speaker pairs.

    Args:
        speakers: dict mapping speaker_id -> list of audio paths.
        n_pairs: Total number of pairs (half same, half different).
        seed: Random seed for reproducibility.

    Returns:
        List of (path_a, path_b, label) tuples.
        label=1 for same speaker, label=0 for different speaker.
    """
    rng = random.Random(seed)
    speaker_ids = list(speakers.keys())

    if len(speaker_ids) < 2:
        raise ValueError("Need at least 2 speakers to generate pairs.")

    pairs = []
    n_same = n_pairs // 2
    n_diff = n_pairs - n_same

    # Same-speaker pairs
    for _ in range(n_same):
        valid = [s for s in speaker_ids if len(speakers[s]) >= 2]
        if not valid:
            raise ValueError("No speaker has >= 2 utterances.")
        sid = rng.choice(valid)
        a, b = rng.sample(speakers[sid], 2)
        pairs.append((a, b, 1))

    # Different-speaker pairs
    for _ in range(n_diff):
        s1, s2 = rng.sample(speaker_ids, 2)
        a = rng.choice(speakers[s1])
        b = rng.choice(speakers[s2])
        pairs.append((a, b, 0))

    rng.shuffle(pairs)
    return pairs


def build_dataset(pairs, sr=16000, n_mfcc=20, enhanced=False):
    """
    Extract features for all pairs.

    Args:
        pairs: List of (path_a, path_b, label) tuples.
        sr: Sample rate for audio loading.
        n_mfcc: Number of MFCC coefficients.
        enhanced: If True, use enhanced embeddings with delta MFCCs.

    Returns:
        X: np.ndarray of shape (n_pairs, feat_dim) — pair feature vectors.
        y: np.ndarray of shape (n_pairs,) — labels (1=same, 0=different).
    """
    embedding_cache = {}
    X_list = []
    y_list = []

    for path_a, path_b, label in tqdm(pairs, desc="Extracting features"):
        if path_a not in embedding_cache:
            embedding_cache[path_a] = audio_to_embedding(
                path_a, sr=sr, n_mfcc=n_mfcc, enhanced=enhanced
            )
        if path_b not in embedding_cache:
            embedding_cache[path_b] = audio_to_embedding(
                path_b, sr=sr, n_mfcc=n_mfcc, enhanced=enhanced
            )

        feat = pair_features(embedding_cache[path_a], embedding_cache[path_b])
        X_list.append(feat)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_concat_dataset(pairs, sr=16000, n_mfcc=20, enhanced=False):
    """
    Extract concatenated pair features for encrypted template protection.

    Instead of |emb_A - emb_B| (40-dim), produces [emb_A || emb_B] (80-dim).
    This allows both voiceprints to be encrypted together, with the comparison
    happening entirely under FHE — the server never sees individual embeddings.

    Args:
        pairs: List of (path_a, path_b, label) tuples.
        sr: Sample rate for audio loading.
        n_mfcc: Number of MFCC coefficients.
        enhanced: If True, use enhanced embeddings with delta MFCCs.

    Returns:
        X: np.ndarray of shape (n_pairs, 2 * emb_dim) — concatenated features.
        y: np.ndarray of shape (n_pairs,) — labels (1=same, 0=different).
    """
    from speaker_verify.features import pair_features_concat

    embedding_cache = {}
    X_list = []
    y_list = []

    for path_a, path_b, label in tqdm(pairs, desc="Extracting features"):
        if path_a not in embedding_cache:
            embedding_cache[path_a] = audio_to_embedding(
                path_a, sr=sr, n_mfcc=n_mfcc, enhanced=enhanced
            )
        if path_b not in embedding_cache:
            embedding_cache[path_b] = audio_to_embedding(
                path_b, sr=sr, n_mfcc=n_mfcc, enhanced=enhanced
            )

        feat = pair_features_concat(embedding_cache[path_a], embedding_cache[path_b])
        X_list.append(feat)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def build_single_utterance_dataset(
    speakers, sr=16000, n_mfcc=20, enhanced=False, max_per_speaker=None
):
    """
    Build dataset of single utterance embeddings labeled by speaker ID.

    Used for speaker identification and gender classification.

    Args:
        speakers: dict mapping speaker_id -> list of audio paths.
        sr: Sample rate.
        n_mfcc: Number of MFCC coefficients.
        enhanced: If True, use enhanced embeddings.
        max_per_speaker: Limit utterances per speaker (None = use all).

    Returns:
        X: np.ndarray of shape (n_samples, feat_dim).
        y_speaker: np.ndarray of shape (n_samples,) — speaker ID indices.
        speaker_ids: List of speaker ID strings (index maps to y_speaker).
    """
    speaker_ids = sorted(speakers.keys())
    X_list = []
    y_list = []

    for idx, sid in enumerate(tqdm(speaker_ids, desc="Processing speakers")):
        paths = speakers[sid]
        if max_per_speaker is not None:
            paths = paths[:max_per_speaker]
        for path in paths:
            emb = audio_to_embedding(path, sr=sr, n_mfcc=n_mfcc, enhanced=enhanced)
            X_list.append(emb)
            y_list.append(idx)

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.int64),
        speaker_ids,
    )
