"""
Hyperparameter optimization for speaker verification.

Runs a grid search over model and training hyperparameters,
logs results to results.tsv, and reports the best configuration.

Usage:
    python demo/optimize.py
    python demo/optimize.py --quick  # fewer experiments
"""

import os
import sys
import time
import argparse
import csv

import torch
import numpy as np

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


EXPERIMENTS = [
    # baseline
    {"name": "baseline", "n_pairs": 2000, "epochs": 200, "lr": 1e-3, "noise_std": 0.0},
    # more data
    {"name": "more_data", "n_pairs": 5000, "epochs": 200, "lr": 1e-3, "noise_std": 0.0},
    # FHE-aware noise
    {
        "name": "fhe_noise_01",
        "n_pairs": 5000,
        "epochs": 200,
        "lr": 1e-3,
        "noise_std": 0.1,
    },
    {
        "name": "fhe_noise_03",
        "n_pairs": 5000,
        "epochs": 200,
        "lr": 1e-3,
        "noise_std": 0.3,
    },
    {
        "name": "fhe_noise_05",
        "n_pairs": 5000,
        "epochs": 200,
        "lr": 1e-3,
        "noise_std": 0.5,
    },
    # learning rates
    {"name": "lr_5e4", "n_pairs": 5000, "epochs": 300, "lr": 5e-4, "noise_std": 0.1},
    {"name": "lr_5e3", "n_pairs": 5000, "epochs": 200, "lr": 5e-3, "noise_std": 0.1},
    # longer training
    {
        "name": "long_train",
        "n_pairs": 5000,
        "epochs": 500,
        "lr": 5e-4,
        "noise_std": 0.1,
    },
]

EXPERIMENTS_QUICK = EXPERIMENTS[:4]


def run_experiment(config, X_all, y_all, save_dir):
    """Run a single training experiment."""
    n_pairs = min(config["n_pairs"], len(X_all))
    X = X_all[:n_pairs]
    y = y_all[:n_pairs]

    model = SpeakerVerifyNet(input_dim=X.shape[1])
    model, scaler, metrics, X_val, y_val = train_model(
        X,
        y,
        model,
        epochs=config["epochs"],
        lr=config["lr"],
        noise_std=config["noise_std"],
    )

    return {
        "name": config["name"],
        "n_pairs": n_pairs,
        "epochs": config["epochs"],
        "lr": config["lr"],
        "noise_std": config["noise_std"],
        "val_acc": metrics["val_acc"],
        "train_acc": metrics["train_acc"],
        "best_epoch": metrics["best_epoch"],
        "model": model,
        "scaler": scaler,
        "X_val": X_val,
        "y_val": y_val,
    }


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")
    parser.add_argument("--quick", action="store_true", help="Run fewer experiments")
    parser.add_argument(
        "--data-root", default=os.path.join(os.path.dirname(__file__), "..", "data")
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(save_dir, "..", "results.tsv")

    # Load/download data
    data_root = os.path.abspath(args.data_root)
    download_librispeech(data_root)
    speakers = scan_speakers(data_root)
    print(f"Found {len(speakers)} speakers")

    # Generate max pairs (we'll subset per experiment)
    max_pairs = 5000
    pairs = generate_pairs(speakers, n_pairs=max_pairs, seed=42)
    print(f"Generating features for {len(pairs)} pairs...")
    X_all, y_all = build_dataset(pairs)
    print(f"Feature matrix: {X_all.shape}")

    experiments = EXPERIMENTS_QUICK if args.quick else EXPERIMENTS

    # Run experiments
    results = []
    best_result = None

    print(f"\n{'=' * 70}")
    print(f"  Running {len(experiments)} experiments")
    print(f"{'=' * 70}\n")

    for i, config in enumerate(experiments):
        print(f"\n--- Experiment {i + 1}/{len(experiments)}: {config['name']} ---")
        print(
            f"    n_pairs={config['n_pairs']}, epochs={config['epochs']}, "
            f"lr={config['lr']}, noise_std={config['noise_std']}"
        )

        t0 = time.time()
        result = run_experiment(config, X_all, y_all, save_dir)
        elapsed = time.time() - t0

        result["time"] = elapsed
        results.append(result)

        print(
            f"    Val Acc: {result['val_acc']:.4f} | "
            f"Train Acc: {result['train_acc']:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if best_result is None or result["val_acc"] > best_result["val_acc"]:
            best_result = result

    # Write results.tsv
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "experiment",
                "n_pairs",
                "epochs",
                "lr",
                "noise_std",
                "val_acc",
                "train_acc",
                "best_epoch",
                "time_s",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["name"],
                    r["n_pairs"],
                    r["epochs"],
                    r["lr"],
                    r["noise_std"],
                    f"{r['val_acc']:.4f}",
                    f"{r['train_acc']:.4f}",
                    r["best_epoch"],
                    f"{r['time']:.1f}",
                ]
            )

    print(f"\nResults saved to {results_path}")

    # Save best model
    print(f"\n{'=' * 70}")
    print(f"  Best: {best_result['name']} — Val Acc: {best_result['val_acc']:.4f}")
    print(f"{'=' * 70}")

    model_path = os.path.join(save_dir, "speaker_model.pt")
    scaler_path = os.path.join(save_dir, "scaler.npz")
    sample_path = os.path.join(save_dir, "test_samples.npz")

    torch.save(best_result["model"].state_dict(), model_path)
    np.savez(
        scaler_path,
        mean=best_result["scaler"].mean_,
        scale=best_result["scaler"].scale_,
    )
    np.savez(sample_path, X=best_result["X_val"], y=best_result["y_val"])

    print(f"Best model saved to {model_path}")


if __name__ == "__main__":
    main()
