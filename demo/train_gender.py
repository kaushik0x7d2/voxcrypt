"""
CLI: Train gender classification model on LibriSpeech.

Classifies speaker gender (male/female) from a single utterance.

Usage:
    python demo/train_gender.py
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

from speaker_verify.model import GenderNet
from speaker_verify.dataset import (download_librispeech, scan_speakers,
                                     parse_gender_metadata, build_single_utterance_dataset)
from speaker_verify.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train Gender Classification Model")
    parser.add_argument("--data-root", default=os.path.join(
        os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(args.data_root)

    download_librispeech(data_root)
    speakers = scan_speakers(data_root)
    gender_map = parse_gender_metadata(data_root)
    print(f"Found {len(speakers)} speakers with gender metadata")

    # Build single-utterance dataset
    X, y_speaker, speaker_ids = build_single_utterance_dataset(speakers)

    # Map speaker indices to gender labels (0=Female, 1=Male)
    gender_labels = []
    for idx in y_speaker:
        sid = speaker_ids[idx]
        gender = gender_map.get(sid, "M")
        gender_labels.append(0 if gender == "F" else 1)
    y_gender = np.array(gender_labels, dtype=np.float32)

    n_female = int((y_gender == 0).sum())
    n_male = int((y_gender == 1).sum())
    print(f"Dataset: {X.shape[0]} utterances ({n_female} female, {n_male} male)")

    # Train
    model = GenderNet(input_dim=X.shape[1])
    print(f"\nTraining GenderNet...")
    model, scaler, metrics, X_val, y_val = train_model(
        X, y_gender, model,
        epochs=args.epochs, lr=args.lr, noise_std=args.noise_std
    )

    print(f"\nBest validation accuracy: {metrics['val_acc']:.4f}")

    model.eval()
    with torch.no_grad():
        val_out = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred = val_out.argmax(dim=1).numpy()
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, target_names=["Female", "Male"]))

    # Save
    model_path = os.path.join(save_dir, "gender_model.pt")
    scaler_path = os.path.join(save_dir, "gender_scaler.npz")
    sample_path = os.path.join(save_dir, "gender_test_samples.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    np.savez(sample_path, X=X_val, y=y_val)

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
