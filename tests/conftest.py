"""
Shared test fixtures for orion-voice tests.

Generates synthetic audio data so tests don't require LibriSpeech.
"""

import os
import sys

import pytest
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))


@pytest.fixture
def synthetic_wav(tmp_path):
    """Generate a short sine-wave WAV file."""
    sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    wav_path = str(tmp_path / "test.wav")
    sf.write(wav_path, audio, sr)
    return wav_path


@pytest.fixture
def speaker_wavs(tmp_path):
    """
    Create 2 'speakers' with different frequency patterns.

    Speaker A: 440 Hz (A4 note)
    Speaker B: 880 Hz (A5 note)

    Each speaker gets 3 utterances with slight variations.
    """
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    speakers = {}
    for speaker_id, base_freq in [("speaker_A", 440), ("speaker_B", 880)]:
        speaker_dir = tmp_path / speaker_id
        speaker_dir.mkdir()
        paths = []
        for i in range(3):
            freq = base_freq + i * 10  # slight variation
            audio = 0.5 * np.sin(2 * np.pi * freq * t)
            # Add some noise for variation
            audio += 0.05 * np.random.RandomState(i).randn(len(audio))
            wav_path = str(speaker_dir / f"utterance_{i}.wav")
            sf.write(wav_path, audio, sr)
            paths.append(wav_path)
        speakers[speaker_id] = paths

    return speakers


@pytest.fixture
def mock_librispeech(tmp_path):
    """
    Create a mock LibriSpeech directory structure for testing scan_speakers.

    Structure: root/LibriSpeech/test-clean/<speaker>/<chapter>/<file>.flac
    """
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    base = tmp_path / "LibriSpeech" / "test-clean"
    for spk_id in ["1001", "1002"]:
        for chap_id in ["100", "101"]:
            chap_dir = base / spk_id / chap_id
            chap_dir.mkdir(parents=True)
            for utt in range(2):
                audio = 0.5 * np.sin(2 * np.pi * (300 + int(spk_id)) * t)
                audio += 0.05 * np.random.RandomState(int(spk_id) + utt).randn(
                    len(audio)
                )
                fpath = chap_dir / f"{spk_id}-{chap_id}-{utt:04d}.flac"
                sf.write(str(fpath), audio, sr)

    return str(tmp_path)


@pytest.fixture
def trained_model(tmp_path):
    """Train a small model on synthetic data and save artifacts."""
    import torch
    from speaker_verify.model import SpeakerVerifyNet
    from speaker_verify.train import train_model

    np.random.seed(42)

    # Generate synthetic pair features (40-dim) with wide separation
    # so the model learns a clear decision boundary that survives FHE noise
    n_samples = 400
    # Same-speaker pairs: values near zero (small |diff|)
    X_same = np.abs(np.random.randn(n_samples // 2, 40).astype(np.float32) * 0.2)
    y_same = np.ones(n_samples // 2, dtype=np.float32)
    # Different-speaker pairs: values far from zero (large |diff|)
    X_diff = np.abs(np.random.randn(n_samples // 2, 40).astype(np.float32) * 1.0) + 2.0
    y_diff = np.zeros(n_samples // 2, dtype=np.float32)

    X = np.concatenate([X_same, X_diff])
    y = np.concatenate([y_same, y_diff])

    model = SpeakerVerifyNet(input_dim=40)
    model, scaler, metrics, X_val, y_val = train_model(X, y, model, epochs=50, lr=1e-3)

    # Save artifacts
    model_path = str(tmp_path / "speaker_model.pt")
    scaler_path = str(tmp_path / "scaler.npz")
    sample_path = str(tmp_path / "test_samples.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    np.savez(sample_path, X=X_val, y=y_val)

    return {
        "model": model,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "sample_path": sample_path,
        "metrics": metrics,
        "X_val": X_val,
        "y_val": y_val,
    }


@pytest.fixture
def fhe_config():
    """Path to FHE config file."""
    return os.path.join(os.path.dirname(__file__), "..", "configs", "fhe_config.yml")
