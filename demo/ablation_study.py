"""
Activation Function Ablation Study for FHE Speaker Verification.

Systematic comparison of polynomial activation functions under CKKS FHE:
  - GELU (Orion's built-in polynomial approximation)
  - SiLU degree=3, 5, 7

Measures: cleartext accuracy, FHE accuracy, FHE-Clear agreement,
precision bits, and inference time for each activation.

This is the first systematic comparison of polynomial activations for
encrypted voice models under CKKS, providing guidance for FHE model design.

Usage:
    python demo/ablation_study.py
    python demo/ablation_study.py --n-pairs 3000 --fhe-samples 5
"""

import os
import sys
import time
import math
import argparse
import json

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))

import orion
import orion.nn as on

from speaker_verify.dataset import (download_librispeech, scan_speakers,
                                     generate_pairs, build_dataset)


class AblationNet(on.Module):
    """MLP with configurable activation for ablation study."""
    def __init__(self, input_dim=40, activation="gelu"):
        super().__init__()
        self.fc1 = on.Linear(input_dim, 128)
        self.fc2 = on.Linear(128, 64)
        self.fc3 = on.Linear(64, 2)
        self.act1 = self._make_activation(activation)
        self.act2 = self._make_activation(activation)
        self.activation_name = activation

    def _make_activation(self, activation):
        if activation == "gelu":
            return on.GELU()
        elif activation.startswith("silu"):
            degree = int(activation.split("_d")[1])
            return on.SiLU(degree=degree)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


def train_ablation_model(X_train, y_train, X_val, y_val, activation,
                         epochs=200, lr=1e-3, noise_std=0.3):
    """Train a model with given activation and return metrics."""
    model = AblationNet(input_dim=X_train.shape[1], activation=activation)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_v = torch.tensor(X_val, dtype=torch.float32)
    y_v = torch.tensor(y_val, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        if noise_std > 0:
            noise = torch.randn_like(X_t) * noise_std
            out = model(X_t + noise)
        else:
            out = model(X_t)

        loss = criterion(out, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val_out = model(X_v)
                val_pred = val_out.argmax(dim=1).numpy()
                val_acc = accuracy_score(y_val, val_pred)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        train_pred = model(X_t).argmax(dim=1).numpy()
        train_acc = accuracy_score(y_train, train_pred)

    return model, best_val_acc, train_acc, best_epoch


def run_fhe_evaluation(model, X_test, y_test, config_path, num_samples=5):
    """Run FHE inference and return metrics."""
    num_samples = min(num_samples, len(X_test))

    # Cleartext predictions
    model.eval()
    X_all = torch.tensor(X_test[:num_samples], dtype=torch.float32)
    with torch.no_grad():
        clear_out = model(X_all)
        clear_preds = clear_out.argmax(dim=1).numpy()

    clear_correct = sum(1 for i in range(num_samples)
                        if clear_preds[i] == int(y_test[i]))

    # FHE
    scheme = orion.init_scheme(config_path)
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    input_level = orion.compile(model)

    fhe_preds = []
    times = []
    bits_list = []

    for i in range(num_samples):
        sample = torch.tensor(X_test[i:i+1], dtype=torch.float32)

        ptxt = orion.encode(sample, input_level)
        ctxt = orion.encrypt(ptxt)
        model.he()

        t0 = time.time()
        out_ctxt = model(ctxt)
        t_inf = time.time() - t0

        out_fhe = out_ctxt.decrypt().decode()
        fhe_out = out_fhe.flatten()[:2]
        pred = fhe_out.argmax().item()
        fhe_preds.append(pred)
        times.append(t_inf)

        model.eval()
        with torch.no_grad():
            c_out = model(sample).flatten()[:2]
        mae = (c_out - fhe_out).abs().mean().item()
        bits = -math.log2(mae) if mae > 0 else float('inf')
        bits_list.append(bits)

    fhe_correct = sum(1 for i in range(num_samples)
                      if fhe_preds[i] == int(y_test[i]))
    agree = sum(1 for i in range(num_samples)
                if fhe_preds[i] == clear_preds[i])

    scheme.delete_scheme()

    return {
        "input_level": input_level,
        "fhe_accuracy": fhe_correct / num_samples,
        "fhe_correct": fhe_correct,
        "clear_accuracy": clear_correct / num_samples,
        "agreement": agree / num_samples,
        "agree_count": agree,
        "avg_time": np.mean(times),
        "avg_bits": np.mean(bits_list),
        "min_bits": np.min(bits_list),
        "num_samples": num_samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Activation Function Ablation Study for FHE")
    parser.add_argument("--data-root", default=os.path.join(
        os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--n-pairs", type=int, default=3000,
                        help="Training pairs per activation")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--fhe-samples", type=int, default=5,
                        help="Number of FHE test samples per activation")
    parser.add_argument("--skip-fhe", action="store_true",
                        help="Skip FHE evaluation (cleartext only)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "..", "configs", "fhe_config.yml")
    data_root = os.path.abspath(args.data_root)

    # Activations to test
    activations = ["gelu", "silu_d3", "silu_d5", "silu_d7"]

    print("=" * 70)
    print("  ACTIVATION FUNCTION ABLATION STUDY")
    print("  Polynomial Activations under CKKS FHE for Speaker Verification")
    print("=" * 70)
    print(f"\n  Activations: {', '.join(activations)}")
    print(f"  Architecture: 40->128->64->2 MLP")
    print(f"  Training: {args.n_pairs} pairs, {args.epochs} epochs, "
          f"noise_std={args.noise_std}")
    print(f"  FHE samples: {args.fhe_samples}")
    print()

    # Prepare data once
    print("Preparing dataset...")
    download_librispeech(data_root)
    speakers = scan_speakers(data_root)
    print(f"  Found {len(speakers)} speakers")

    pairs = generate_pairs(speakers, n_pairs=args.n_pairs)
    X, y = build_dataset(pairs, n_mfcc=20)
    print(f"  Features: {X.shape}")

    # Train/val split
    n_val = int(len(X) * 0.2)
    indices = np.random.permutation(len(X))
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    y_train = y[train_idx].astype(int)
    y_val = y[val_idx].astype(int)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}\n")

    # Run ablation
    results = {}

    for act in activations:
        print(f"\n{'='*50}")
        print(f"  Training with: {act}")
        print(f"{'='*50}")

        torch.manual_seed(42)
        model, val_acc, train_acc, best_epoch = train_ablation_model(
            X_train, y_train, X_val, y_val, act,
            epochs=args.epochs, lr=args.lr, noise_std=args.noise_std)

        print(f"  Train acc: {train_acc:.4f}")
        print(f"  Val acc:   {val_acc:.4f} (epoch {best_epoch})")

        entry = {
            "activation": act,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "best_epoch": best_epoch,
        }

        if not args.skip_fhe:
            print(f"\n  Running FHE evaluation ({args.fhe_samples} samples)...")
            fhe_metrics = run_fhe_evaluation(
                model, X_val, y_val, config_path, args.fhe_samples)
            entry.update(fhe_metrics)

            print(f"  FHE accuracy:  {fhe_metrics['fhe_correct']}/{fhe_metrics['num_samples']} "
                  f"({fhe_metrics['fhe_accuracy']:.0%})")
            print(f"  Agreement:     {fhe_metrics['agree_count']}/{fhe_metrics['num_samples']}")
            print(f"  Precision:     {fhe_metrics['avg_bits']:.1f} bits avg, "
                  f"{fhe_metrics['min_bits']:.1f} bits min")
            print(f"  Inference:     {fhe_metrics['avg_time']:.1f}s avg")
            print(f"  Input level:   {fhe_metrics['input_level']}")

        results[act] = entry

    # Results table
    print(f"\n\n{'='*70}")
    print(f"  ABLATION RESULTS SUMMARY")
    print(f"{'='*70}")

    if args.skip_fhe:
        print(f"\n  {'Activation':<12s} | {'Val Acc':>8s} | {'Train Acc':>9s} | {'Epoch':>5s}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*9}-+-{'-'*5}")
        for act, r in results.items():
            print(f"  {act:<12s} | {r['val_acc']:>7.1%} | {r['train_acc']:>8.1%} | {r['best_epoch']:>5d}")
    else:
        print(f"\n  {'Activation':<12s} | {'Val Acc':>8s} | {'FHE Acc':>8s} | {'Agree':>6s} | "
              f"{'Bits':>6s} | {'Time':>6s} | {'Level':>5s}")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}")
        for act, r in results.items():
            print(f"  {act:<12s} | {r['val_acc']:>7.1%} | {r['fhe_accuracy']:>7.0%} | "
                  f"{r['agree_count']:>2d}/{r['num_samples']:<2d} | "
                  f"{r['avg_bits']:>5.1f} | {r['avg_time']:>5.1f}s | {r['input_level']:>5d}")

    print(f"\n  Key findings:")
    if not args.skip_fhe:
        best_fhe = max(results.values(), key=lambda r: r.get('fhe_accuracy', 0))
        worst_fhe = min(results.values(), key=lambda r: r.get('fhe_accuracy', 1))
        best_bits = max(results.values(), key=lambda r: r.get('avg_bits', 0))
        print(f"    Best FHE accuracy:  {best_fhe['activation']} ({best_fhe['fhe_accuracy']:.0%})")
        print(f"    Worst FHE accuracy: {worst_fhe['activation']} ({worst_fhe['fhe_accuracy']:.0%})")
        print(f"    Best precision:     {best_bits['activation']} ({best_bits['avg_bits']:.1f} bits)")

    # Save results
    results_path = os.path.join(demo_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")


if __name__ == "__main__":
    main()
