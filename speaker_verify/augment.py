"""
Data augmentation for audio and features.

Augmentations run client-side on plaintext audio before encryption.
"""

import numpy as np
import librosa


def add_noise(audio, snr_db=20, rng=None):
    """Add Gaussian noise at a given SNR level."""
    if rng is None:
        rng = np.random.default_rng()
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(audio))
    return audio + noise


def pitch_shift(audio, sr=16000, n_steps=None, rng=None):
    """Shift pitch by n_steps semitones (random if not specified)."""
    if rng is None:
        rng = np.random.default_rng()
    if n_steps is None:
        n_steps = rng.uniform(-2, 2)
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)


def time_stretch(audio, rate=None, rng=None):
    """Time-stretch audio by a factor (random if not specified)."""
    if rng is None:
        rng = np.random.default_rng()
    if rate is None:
        rate = rng.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y=audio, rate=rate)


def volume_perturb(audio, gain_db=None, rng=None):
    """Apply random volume change in dB."""
    if rng is None:
        rng = np.random.default_rng()
    if gain_db is None:
        gain_db = rng.uniform(-6, 6)
    return audio * (10 ** (gain_db / 20))


def augment_audio(audio, sr=16000, augmentations=None, rng=None):
    """
    Apply a random subset of augmentations to audio.

    Args:
        audio: np.ndarray audio signal.
        sr: Sample rate.
        augmentations: List of augmentation names to apply.
            Options: "noise", "pitch", "stretch", "volume".
            If None, randomly picks 1-2 augmentations.
        rng: Random number generator.

    Returns:
        Augmented audio signal.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_augs = ["noise", "pitch", "stretch", "volume"]

    if augmentations is None:
        n_augs = rng.integers(1, 3)
        augmentations = list(rng.choice(all_augs, size=n_augs, replace=False))

    for aug in augmentations:
        if aug == "noise":
            snr = rng.uniform(15, 30)
            audio = add_noise(audio, snr_db=snr, rng=rng)
        elif aug == "pitch":
            audio = pitch_shift(audio, sr=sr, rng=rng)
        elif aug == "stretch":
            audio = time_stretch(audio, rng=rng)
        elif aug == "volume":
            audio = volume_perturb(audio, rng=rng)

    return audio


def feature_noise(features, noise_std=0.1, rng=None):
    """
    Add Gaussian noise to feature vectors.

    Used during training to simulate FHE precision loss,
    making the model learn wider decision margins.

    Args:
        features: np.ndarray of feature vectors.
        noise_std: Standard deviation of noise.
        rng: Random number generator.

    Returns:
        Noisy features.
    """
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0, noise_std, features.shape).astype(features.dtype)
    return features + noise
