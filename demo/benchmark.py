"""
Benchmark FHE inference performance.

Measures encryption, inference, and decryption times across multiple samples.

Usage:
    python demo/benchmark.py
    python demo/benchmark.py --num-samples 20
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
    parser = argparse.ArgumentParser(description="FHE Benchmark")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of samples to benchmark")
    args = parser.parse_args()

    torch.manual_seed(42)

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "..", "configs", "fhe_config.yml")

    for f in ["speaker_model.pt", "test_samples.npz"]:
        if not os.path.exists(os.path.join(demo_dir, f)):
            print(f"Missing {f}. Run train_model.py first.")
            return

    model = SpeakerVerifyNet()
    model.load_state_dict(torch.load(
        os.path.join(demo_dir, "speaker_model.pt"), weights_only=True))
    model.eval()

    samples = np.load(os.path.join(demo_dir, "test_samples.npz"))
    X_test = samples["X"]

    num_samples = min(args.num_samples, len(X_test))

    # Init scheme
    print("Initializing FHE scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    t_init = time.time() - t0
    print(f"  Scheme init: {t_init:.2f}s")

    # Fit
    print("Fitting model...")
    t0 = time.time()
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    t_fit = time.time() - t0
    print(f"  Fit: {t_fit:.2f}s")

    # Compile
    print("Compiling model...")
    t0 = time.time()
    input_level = orion.compile(model)
    t_compile = time.time() - t0
    print(f"  Compile: {t_compile:.2f}s | Input level: {input_level}")

    # Benchmark inference
    print(f"\nBenchmarking {num_samples} samples...")
    print(f"{'Sample':>8s} | {'Encode':>8s} | {'Encrypt':>8s} | {'Infer':>8s} | {'Decrypt':>8s} | {'Total':>8s} | {'Bits':>6s}")
    print("-" * 72)

    enc_times = []
    inf_times = []
    dec_times = []
    total_times = []
    precision_bits = []

    for i in range(num_samples):
        sample = torch.tensor(X_test[i:i+1], dtype=torch.float32)
        t_start = time.time()

        t0 = time.time()
        ptxt = orion.encode(sample, input_level)
        ctxt = orion.encrypt(ptxt)
        model.he()
        t_enc = time.time() - t0

        t0 = time.time()
        out_ctxt = model(ctxt)
        t_inf = time.time() - t0

        t0 = time.time()
        out_fhe = out_ctxt.decrypt().decode()
        t_dec = time.time() - t0

        t_total = time.time() - t_start

        # Precision
        fhe_out = out_fhe.flatten()[:2]
        model.eval()
        with torch.no_grad():
            c_out = model(sample).flatten()[:2]
        mae = (c_out - fhe_out).abs().mean().item()
        bits = -math.log2(mae) if mae > 0 else float('inf')

        enc_times.append(t_enc)
        inf_times.append(t_inf)
        dec_times.append(t_dec)
        total_times.append(t_total)
        precision_bits.append(bits)

        print(f"{i+1:>8d} | {t_enc:>7.2f}s | {t_enc:>7.2f}s | {t_inf:>7.2f}s | {t_dec:>7.2f}s | {t_total:>7.2f}s | {bits:>5.1f}")

    # Summary
    print(f"\n{'='*72}")
    print(f"  Samples:         {num_samples}")
    print(f"  Setup:           {t_init + t_fit + t_compile:.2f}s (init + fit + compile)")
    print(f"  Avg Encode+Enc:  {np.mean(enc_times):.2f}s")
    print(f"  Avg Inference:   {np.mean(inf_times):.2f}s")
    print(f"  Avg Decrypt:     {np.mean(dec_times):.2f}s")
    print(f"  Avg Total:       {np.mean(total_times):.2f}s")
    print(f"  Min/Max Infer:   {np.min(inf_times):.2f}s / {np.max(inf_times):.2f}s")
    print(f"  Avg Precision:   {np.mean(precision_bits):.1f} bits")
    print(f"  Model:           SpeakerVerifyNet (40->128->64->2, SiLU)")
    print(f"{'='*72}")

    scheme.delete_scheme()


if __name__ == "__main__":
    main()
