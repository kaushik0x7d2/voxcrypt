"""
CLI: Train encrypted template protection speaker verification model.

Unlike the standard model (which takes |emb_A - emb_B| computed in cleartext),
this model takes concatenated [emb_A || emb_B] as input. Both voiceprints are
encrypted together, and the comparison happens entirely under FHE.

This is the first FHE neural network approach to speaker verification with
encrypted biometric template comparison, improving over Nautsch et al. (2018)
which used Paillier partial HE with fixed cosine/Euclidean distance metrics.

Usage:
    python demo/train_encrypted_verify.py
    python demo/train_encrypted_verify.py --n-pairs 5000 --noise-std 0.3
"""

import os
import sys
import argparse

import torch
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))

from speaker_verify.model import EncryptedVerifyNet
from speaker_verify.dataset import (
    download_librispeech,
    scan_speakers,
    generate_pairs,
    build_concat_dataset,
)
from speaker_verify.train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Train Encrypted Template Protection Model"
    )
    parser.add_argument(
        "--data-root", default=os.path.join(os.path.dirname(__file__), "..", "data")
    )
    parser.add_argument("--n-pairs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--n-mfcc", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(args.data_root)

    # Download & scan
    download_librispeech(data_root)
    speakers = scan_speakers(data_root)
    print(f"Found {len(speakers)} speakers")

    # Generate pairs
    pairs = generate_pairs(speakers, n_pairs=args.n_pairs)
    n_same = sum(1 for _, _, lbl in pairs if lbl == 1)
    n_diff = sum(1 for _, _, lbl in pairs if lbl == 0)
    print(f"Generated {len(pairs)} pairs ({n_same} same, {n_diff} different)")

    # Extract CONCATENATED features (both embeddings together)
    X, y = build_concat_dataset(pairs, n_mfcc=args.n_mfcc)
    print(f"Feature matrix: {X.shape} | Labels: {y.shape}")
    print("  (80-dim = [emb_A || emb_B] concatenated embeddings)")

    # Train
    model = EncryptedVerifyNet(input_dim=X.shape[1])
    print(f"\nTraining EncryptedVerifyNet ({X.shape[1]}->128->64->2, GELU)...")
    model, scaler, metrics, X_val, y_val = train_model(
        X, y, model, epochs=args.epochs, lr=args.lr, noise_std=args.noise_std
    )

    print(
        f"\nBest validation accuracy: {metrics['val_acc']:.4f} "
        f"(epoch {metrics['best_epoch']})"
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

    # Save
    model_path = os.path.join(save_dir, "encrypted_model.pt")
    scaler_path = os.path.join(save_dir, "encrypted_scaler.npz")
    sample_path = os.path.join(save_dir, "encrypted_test_samples.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    np.savez(sample_path, X=X_val, y=y_val)

    print(f"\nEncrypted template model saved to {model_path}")
    print("  Input: [emb_A || emb_B] (80-dim concatenated)")
    print("  Both voiceprints encrypted together for FHE comparison")


if __name__ == "__main__":
    main()
