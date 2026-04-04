"""
FHE inference pipeline for speaker verification.

Handles scheme initialization, model compilation, and encrypted inference.
"""

import os
import sys
import time
import math

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))
import orion


def init_fhe(config_path, model, sample_path):
    """
    Initialize FHE scheme and compile model.

    Args:
        config_path: Path to fhe_config.yml.
        model: SpeakerVerifyNet instance with loaded weights.
        sample_path: Path to test_samples.npz (for fitting).

    Returns:
        scheme: Initialized FHE scheme.
        input_level: Input level for encoding.
    """
    print("[FHE] Initializing scheme...")
    t0 = time.time()
    scheme = orion.init_scheme(config_path)
    print(f"[FHE] Scheme ready ({time.time()-t0:.2f}s)")

    print("[FHE] Fitting model...")
    t0 = time.time()
    samples = np.load(sample_path)
    fit_X = torch.tensor(samples["X"], dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))
    print(f"[FHE] Fit done ({time.time()-t0:.2f}s)")

    print("[FHE] Compiling model...")
    t0 = time.time()
    input_level = orion.compile(model)
    print(f"[FHE] Compiled ({time.time()-t0:.2f}s) | Input level: {input_level}")

    return scheme, input_level


def fhe_predict(model, sample, input_level):
    """
    Run encrypted inference on a single sample.

    Args:
        model: Compiled SpeakerVerifyNet.
        sample: torch.Tensor of shape (1, input_dim).
        input_level: Input level from compilation.

    Returns:
        prediction: int (0=different, 1=same speaker).
        fhe_output: torch.Tensor of raw FHE output values.
        elapsed: float, inference time in seconds.
        precision_bits: float, precision compared to cleartext.
    """
    # Encode + encrypt
    ptxt = orion.encode(sample, input_level)
    ctxt = orion.encrypt(ptxt)
    model.he()

    # Encrypted inference
    t0 = time.time()
    out_ctxt = model(ctxt)
    elapsed = time.time() - t0

    # Decrypt + decode
    out_fhe = out_ctxt.decrypt().decode()
    fhe_output = out_fhe.flatten()[:2]
    prediction = fhe_output.argmax().item()

    # Compute precision vs cleartext
    model.eval()
    with torch.no_grad():
        clear_out = model(sample).flatten()[:2]
    mae = (clear_out - fhe_output).abs().mean().item()
    precision_bits = -math.log2(mae) if mae > 0 else float('inf')

    return prediction, fhe_output, elapsed, precision_bits


def cleanup(scheme):
    """Clean up FHE scheme resources."""
    scheme.delete_scheme()
