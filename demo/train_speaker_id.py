"""
CLI: Train speaker identification model on LibriSpeech.

Identifies WHICH speaker is talking from a single utterance.

Usage:
    python demo/train_speaker_id.py
"""

import os
import sys
import argparse

import torch
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))

from speaker_verify.model import SpeakerIDNet
from speaker_verify.dataset import download_librispeech, scan_speakers, build_single_utterance_dataset
from speaker_verify.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train Speaker ID Model")
    parser.add_argument("--data-root", default=os.path.join(
        os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.1,
                        help="FHE-aware noise injection")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(args.data_root)

    download_librispeech(data_root)
    speakers = scan_speakers(data_root)
    print(f"Found {len(speakers)} speakers")

    # Build single-utterance dataset
    X, y, speaker_ids = build_single_utterance_dataset(speakers)
    print(f"Dataset: {X.shape[0]} utterances, {len(speaker_ids)} speakers")
    print(f"Feature dim: {X.shape[1]}")

    # Train
    n_speakers = len(speaker_ids)
    model = SpeakerIDNet(input_dim=X.shape[1], n_speakers=n_speakers)
    print(f"\nTraining SpeakerIDNet ({X.shape[1]}->{n_speakers})...")

    model, scaler, metrics, X_val, y_val = train_model(
        X, y.astype(np.float32), model,
        epochs=args.epochs, lr=args.lr, noise_std=args.noise_std
    )

    print(f"\nBest validation accuracy: {metrics['val_acc']:.4f}")
    print(f"Train accuracy: {metrics['train_acc']:.4f}")

    # Save
    model_path = os.path.join(save_dir, "speaker_id_model.pt")
    scaler_path = os.path.join(save_dir, "speaker_id_scaler.npz")
    sample_path = os.path.join(save_dir, "speaker_id_test_samples.npz")
    meta_path = os.path.join(save_dir, "speaker_id_meta.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    np.savez(sample_path, X=X_val, y=y_val)
    np.savez(meta_path, speaker_ids=np.array(speaker_ids))

    print(f"\nModel saved to {model_path}")
    print(f"Speaker IDs: {speaker_ids[:5]}... ({n_speakers} total)")


if __name__ == "__main__":
    main()
