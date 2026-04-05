"""
Feature extraction for speaker verification and related tasks.

Extracts MFCC-based embeddings from audio files and computes
pair features for speaker comparison.
"""

import numpy as np
import librosa


def extract_mfcc(audio_path, sr=16000, n_mfcc=20):
    """
    Extract MFCCs from an audio file.

    Args:
        audio_path: Path to WAV file.
        sr: Target sample rate (default 16kHz).
        n_mfcc: Number of MFCC coefficients.

    Returns:
        np.ndarray of shape (n_frames, n_mfcc).
    """
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T  # (n_frames, n_mfcc)


def extract_mfcc_from_audio(audio, sr=16000, n_mfcc=20):
    """Extract MFCCs from a raw audio array (no file I/O)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.T


def utterance_embedding(mfcc):
    """
    Compute a fixed-length embedding from variable-length MFCCs.

    Concatenates mean and standard deviation across frames.

    Args:
        mfcc: np.ndarray of shape (n_frames, n_mfcc).

    Returns:
        np.ndarray of shape (2 * n_mfcc,) — e.g., 40-dim for 20 MFCCs.
    """
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)
    return np.concatenate([mean, std])


def utterance_embedding_enhanced(mfcc):
    """
    Enhanced embedding: MFCCs + delta MFCCs, mean + std for each.

    Captures temporal dynamics in addition to static features.

    Args:
        mfcc: np.ndarray of shape (n_frames, n_mfcc).

    Returns:
        np.ndarray of shape (4 * n_mfcc,) — e.g., 80-dim for 20 MFCCs.
    """
    # Static MFCC stats
    mean = np.mean(mfcc, axis=0)
    std = np.std(mfcc, axis=0)

    # Delta MFCCs (temporal derivative)
    delta = librosa.feature.delta(mfcc.T).T  # (n_frames, n_mfcc)
    delta_mean = np.mean(delta, axis=0)
    delta_std = np.std(delta, axis=0)

    return np.concatenate([mean, std, delta_mean, delta_std])


def pair_features(emb_a, emb_b):
    """
    Compute pair features for speaker verification.

    Uses absolute difference: |emb_a - emb_b|.
    This halves input size vs concatenation and has cleaner privacy semantics.

    Args:
        emb_a: np.ndarray embedding for utterance A.
        emb_b: np.ndarray embedding for utterance B.

    Returns:
        np.ndarray of shape (len(emb_a),) — absolute difference.
    """
    return np.abs(emb_a - emb_b)


def audio_to_embedding(audio_path, sr=16000, n_mfcc=20, enhanced=False):
    """
    Convenience: audio file -> utterance embedding in one call.

    Args:
        audio_path: Path to WAV file.
        sr: Target sample rate.
        n_mfcc: Number of MFCC coefficients.
        enhanced: If True, use enhanced embedding with delta MFCCs.

    Returns:
        np.ndarray of shape (2*n_mfcc,) or (4*n_mfcc,) if enhanced.
    """
    mfcc = extract_mfcc(audio_path, sr=sr, n_mfcc=n_mfcc)
    if enhanced:
        return utterance_embedding_enhanced(mfcc)
    return utterance_embedding(mfcc)
