"""
CLI: Download LibriSpeech, extract features, train speaker verification model.

Usage:
    python demo/train_model.py
    python demo/train_model.py --n-pairs 5000 --epochs 300
"""

import os
import sys
import argparse

import torch
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))

from speaker_verify.model import SpeakerVerifyNet
from speaker_verify.dataset import (
    download_librispeech,
    scan_speakers,
    generate_pairs,
    build_dataset,
)
from speaker_verify.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train Speaker Verification Model")
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "..", "data"),
        help="Root directory for LibriSpeech download",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=2000, help="Number of training pairs"
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.3,
        help="FHE-aware noise injection std (0=off)",
    )
    parser.add_argument(
        "--n-mfcc", type=int, default=20, help="Number of MFCC coefficients"
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Download LibriSpeech
    data_root = os.path.abspath(args.data_root)
    download_librispeech(data_root)

    # Step 2: Scan speakers
    speakers = scan_speakers(data_root)
    print(f"Found {len(speakers)} speakers")
    total_utts = sum(len(v) for v in speakers.values())
    print(f"Total utterances: {total_utts}")

    # Step 3: Generate pairs
    pairs = generate_pairs(speakers, n_pairs=args.n_pairs)
    n_same = sum(1 for _, _, lbl in pairs if lbl == 1)
    n_diff = sum(1 for _, _, lbl in pairs if lbl == 0)
    print(f"Generated {len(pairs)} pairs ({n_same} same, {n_diff} different)")

    # Step 4: Extract features
    X, y = build_dataset(pairs, n_mfcc=args.n_mfcc)
    print(f"Feature matrix: {X.shape} | Labels: {y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))} (different / same)")

    # Step 5: Train
    print(f"\nTraining SpeakerVerifyNet (input_dim={X.shape[1]})...")
    model = SpeakerVerifyNet(input_dim=X.shape[1])
    model, scaler, metrics, X_val, y_val = train_model(
        X, y, model, epochs=args.epochs, lr=args.lr, noise_std=args.noise_std
    )

    print(
        f"\nBest validation accuracy: {metrics['val_acc']:.4f} (epoch {metrics['best_epoch']})"
    )
    print(f"Train accuracy: {metrics['train_acc']:.4f}")

    # Classification report
    model.eval()
    with torch.no_grad():
        val_out = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred = val_out.argmax(dim=1).numpy()
    print("\nClassification Report:")
    print(
        classification_report(
            y_val, val_pred, target_names=["Diff Speaker", "Same Speaker"]
        )
    )

    # Step 6: Save artifacts
    model_path = os.path.join(save_dir, "speaker_model.pt")
    scaler_path = os.path.join(save_dir, "scaler.npz")
    sample_path = os.path.join(save_dir, "test_samples.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    np.savez(sample_path, X=X_val, y=y_val)

    print(f"\nModel saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")
    print(f"Test samples saved to {sample_path}")


if __name__ == "__main__":
    main()
