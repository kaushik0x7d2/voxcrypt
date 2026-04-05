"""
CLI: FHE demo for Encrypted Biometric Template Protection.

Unlike standard verification (which computes |emb_A - emb_B| in cleartext),
this approach encrypts both voiceprints together as [emb_A || emb_B] and
performs the comparison entirely under FHE. The server never sees individual
embeddings — both biometric templates remain encrypted throughout.

This is the first FHE neural network approach to speaker verification with
encrypted biometric template comparison, improving over Nautsch et al. (2018)
which used Paillier partial HE with fixed cosine/Euclidean distance metrics.

Usage:
    python demo/encrypted_verify_demo.py
    python demo/encrypted_verify_demo.py --num-samples 10
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))

import orion
from speaker_verify.model import EncryptedVerifyNet


def main():
    parser = argparse.ArgumentParser(
        description="Encrypted Template Protection FHE Demo"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of test pairs for FHE inference",
    )
    args = parser.parse_args()

    torch.manual_seed(42)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "..", "configs", "fhe_config.yml")

    # Check required files
    for f in ["encrypted_model.pt", "encrypted_test_samples.npz"]:
        if not os.path.exists(os.path.join(demo_dir, f)):
            print(f"Missing {f}. Run train_encrypted_verify.py first.")
            return

    # Load model
    model = EncryptedVerifyNet(input_dim=80)
    model.load_state_dict(
        torch.load(os.path.join(demo_dir, "encrypted_model.pt"), weights_only=True)
    )
    model.eval()

    samples = np.load(os.path.join(demo_dir, "encrypted_test_samples.npz"))
    X_test, y_test = samples["X"], samples["y"]

    num_samples = min(args.num_samples, len(X_test))
    labels = ["Diff Speaker", "Same Speaker"]

    print("=" * 65)
    print("  ENCRYPTED BIOMETRIC TEMPLATE PROTECTION — FHE Demo")
    print("=" * 65)
    print("\n  Input: [emb_A || emb_B] (80-dim concatenated embeddings)")
    print("  Both voiceprints encrypted together — server never sees")
    print("  individual embeddings. Comparison happens under FHE.")
    print("  Model: EncryptedVerifyNet (80->128->64->2, GELU)")
    print()

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
        print(
            f"  Pair {i + 1:2d}/{num_samples}: {labels[pred]:<16s} "
            f"(actual: {labels[actual]:<16s}) [{status:>5s}]"
        )

    clear_acc = (clear_preds == y_test[:num_samples].astype(int)).mean()
    print(
        f"\nCleartext Accuracy: {int(clear_acc * num_samples)}/{num_samples} "
        f"({clear_acc:.0%})"
    )

    # === FHE Inference ===
    print(f"\n=== FHE Inference ({num_samples} pairs) ===")
    print("  Both embeddings are encrypted together — the server performs")
    print("  the speaker comparison entirely on ciphertext.\n")

    print("[1] Initializing CKKS scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    print(f"    Done ({time.time() - t0:.2f}s)")

    print("[2] Fitting model...")
    t0 = time.time()
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    print(f"    Done ({time.time() - t0:.2f}s)")

    print("[3] Compiling for FHE...")
    t0 = time.time()
    input_level = orion.compile(model)
    print(f"    Done ({time.time() - t0:.2f}s) | Input level: {input_level}")

    results = []
    times = []
    bits_list = []

    for i in range(num_samples):
        sample = torch.tensor(X_test[i : i + 1], dtype=torch.float32)
        actual = int(y_test[i])

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
        bits = -math.log2(mae) if mae > 0 else float("inf")
        bits_list.append(bits)

        status = "ok" if pred == actual else "WRONG"
        print(
            f"  Pair {i + 1:2d}/{num_samples}: {labels[pred]:<16s} "
            f"(actual: {labels[actual]:<16s}) [{status:>5s}] "
            f"| {t_inf:.1f}s | {bits:.1f} bits"
        )

    # Summary
    correct = sum(1 for i in range(len(results)) if results[i] == int(y_test[i]))
    agree = sum(1 for i in range(len(results)) if results[i] == clear_preds[i])

    print(f"\n{'=' * 65}")
    print("  RESULTS — Encrypted Template Protection")
    print(f"{'=' * 65}")
    print(
        f"  FHE Accuracy:         {correct}/{num_samples} ({correct / num_samples:.0%})"
    )
    print(f"  FHE-Clear Agreement:  {agree}/{num_samples}")
    print(f"  Avg Inference Time:   {np.mean(times):.1f}s per pair")
    print(f"  Avg Precision:        {np.mean(bits_list):.1f} bits")
    print("\n  Novel contribution:")
    print("    Both voiceprints encrypted together as [emb_A || emb_B]")
    print("    Comparison learned by neural network under FHE")
    print("    Server never sees individual biometric templates")
    print("    Improves over Nautsch et al. (2018) Paillier partial HE")

    scheme.delete_scheme()


if __name__ == "__main__":
    main()
