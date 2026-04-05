"""
CLI: Train emotion detection model on Emo-DB.

Classifies speech emotion: neutral, anger, fear, happiness, sadness, disgust, boredom.

Usage:
    python demo/train_emotion.py
"""

import os
import sys
import argparse

import torch
import numpy as np
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))

from speaker_verify.model import EmotionNet
from speaker_verify.emotion import download_emodb, build_emotion_dataset
from speaker_verify.train import train_model


def main():
    parser = argparse.ArgumentParser(description="Train Emotion Detection Model")
    parser.add_argument(
        "--data-root", default=os.path.join(os.path.dirname(__file__), "..", "data")
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.1)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(args.data_root)

    # Download Emo-DB
    wav_dir = download_emodb(data_root)

    # Build dataset
    X, y, emotion_names = build_emotion_dataset(wav_dir)
    print(f"Dataset: {X.shape[0]} utterances, {len(emotion_names)} emotions")
    print(f"Feature dim: {X.shape[1]}")
    for i, name in enumerate(emotion_names):
        count = int((y == i).sum())
        print(f"  {name}: {count}")

    # Train
    model = EmotionNet(input_dim=X.shape[1], n_emotions=len(emotion_names))
    print("\nTraining EmotionNet...")
    model, scaler, metrics, X_val, y_val = train_model(
        X,
        y.astype(np.float32),
        model,
        epochs=args.epochs,
        lr=args.lr,
        noise_std=args.noise_std,
    )

    print(f"\nBest validation accuracy: {metrics['val_acc']:.4f}")

    model.eval()
    with torch.no_grad():
        val_out = model(torch.tensor(X_val, dtype=torch.float32))
        val_pred = val_out.argmax(dim=1).numpy()
    print("\nClassification Report:")
    print(classification_report(y_val, val_pred, target_names=emotion_names))

    # Save
    model_path = os.path.join(save_dir, "emotion_model.pt")
    scaler_path = os.path.join(save_dir, "emotion_scaler.npz")
    sample_path = os.path.join(save_dir, "emotion_test_samples.npz")
    meta_path = os.path.join(save_dir, "emotion_meta.npz")

    torch.save(model.state_dict(), model_path)
    np.savez(scaler_path, mean=scaler.mean_, scale=scaler.scale_)
    np.savez(sample_path, X=X_val, y=y_val)
    np.savez(meta_path, emotion_names=np.array(emotion_names))

    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
