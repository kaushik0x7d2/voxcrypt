"""
CLI: End-to-end FHE speaker verification demo.

Loads trained model, runs cleartext inference, then FHE inference,
and compares results.

Usage:
    python demo/fhe_demo.py
    python demo/fhe_demo.py --num-samples 5
"""

import os
import sys
import time
import math
import argparse

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))

import orion
from speaker_verify.model import SpeakerVerifyNet


def main():
    parser = argparse.ArgumentParser(description="FHE Speaker Verification Demo")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of test pairs for FHE inference")
    args = parser.parse_args()

    torch.manual_seed(42)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "..", "configs", "fhe_config.yml")

    # Check required files
    for f in ["speaker_model.pt", "test_samples.npz"]:
        if not os.path.exists(os.path.join(demo_dir, f)):
            print(f"Missing {f}. Run train_model.py first.")
            return

    # Load model
    model = SpeakerVerifyNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "speaker_model.pt"), weights_only=True))
    model.eval()

    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    X_test, y_test = samples["X"], samples["y"]

    num_samples = min(args.num_samples, len(X_test))
    labels = ["Diff Speaker", "Same Speaker"]

    # === Cleartext Inference ===
    print(f"=== Cleartext Inference ({num_samples} pairs) ===")
    X_all = torch.tensor(X_test[:num_samples], dtype=torch.float32)
    with torch.no_grad():
        clear_out = model(X_all)
        clear_preds = clear_out.argmax(dim=1).numpy()

    for i in range(num_samples):
        pred = clear_preds[i]
        actual = int(y_test[i])
        status = "ok" if pred == actual else "WRONG"
        print(f"  Pair {i+1:2d}/{num_samples}: {labels[pred]:<16s} "
              f"(actual: {labels[actual]:<16s}) [{status:>5s}]")

    clear_acc = (clear_preds == y_test[:num_samples].astype(int)).mean()
    print(f"\nCleartext Accuracy: {int(clear_acc * num_samples)}/{num_samples} ({clear_acc:.0%})")

    # === FHE Inference ===
    print(f"\n=== FHE Inference ({num_samples} pairs) ===")

    print("\n[1] Initializing scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    print(f"    Done ({time.time()-t0:.2f}s)")

    print("[2] Fitting...")
    t0 = time.time()
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    print(f"    Done ({time.time()-t0:.2f}s)")

    print("[3] Compiling...")
    t0 = time.time()
    input_level = orion.compile(model)
    print(f"    Done ({time.time()-t0:.2f}s) | Input level: {input_level}")

    results = []
    times = []

    for i in range(num_samples):
        sample = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        actual = int(y_test[i])

        t0 = time.time()
        ptxt = orion.encode(sample, input_level)
        ctxt = orion.encrypt(ptxt)
        model.he()

        t0_inf = time.time()
        out_ctxt = model(ctxt)
        t_inf = time.time() - t0_inf

        out_fhe = out_ctxt.decrypt().decode()
        fhe_out = out_fhe.flatten()[:2]
        pred = fhe_out.argmax().item()
        results.append(pred)
        times.append(t_inf)

        # Precision
        model.eval()
        with torch.no_grad():
            c_out = model(sample).flatten()[:2]
        mae = (c_out - fhe_out).abs().mean().item()
        bits = -math.log2(mae) if mae > 0 else float('inf')

        status = "ok" if pred == actual else "WRONG"
        print(f"  Pair {i+1:2d}/{num_samples}: {labels[pred]:<16s} "
              f"(actual: {labels[actual]:<16s}) [{status:>5s}] "
              f"| {t_inf:.1f}s | {bits:.1f} bits")

    # Summary
    correct = sum(1 for i in range(len(results)) if results[i] == int(y_test[i]))
    agree = sum(1 for i in range(len(results)) if results[i] == clear_preds[i])

    print(f"\n=== Summary ===")
    print(f"  FHE Accuracy:         {correct}/{num_samples} ({correct/num_samples:.0%})")
    print(f"  FHE-Clear Agreement:  {agree}/{num_samples}")
    print(f"  Avg Inference Time:   {np.mean(times):.1f}s per pair")
    print(f"  Model:                SpeakerVerifyNet (40->128->64->2, GELU)")

    scheme.delete_scheme()


if __name__ == "__main__":
    main()
