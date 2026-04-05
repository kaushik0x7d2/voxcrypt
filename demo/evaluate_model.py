"""
Standalone model evaluation — FAR/FRR/EER analysis.

Loads a trained model and computes comprehensive evaluation metrics.

Usage:
    python demo/evaluate_model.py
    python demo/evaluate_model.py --model speaker_model.pt
"""

import os
import sys
import json
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))

from speaker_verify.model import SpeakerVerifyNet
from speaker_verify.evaluation import (
    compute_scores, evaluation_report, compute_far_frr)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Speaker Verification Model")
    parser.add_argument("--model", default="speaker_model.pt")
    parser.add_argument("--scaler", default="scaler.npz")
    parser.add_argument("--samples", default="test_samples.npz")
    args = parser.parse_args()

    demo_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(demo_dir, args.model)
    scaler_path = os.path.join(demo_dir, args.scaler)
    samples_path = os.path.join(demo_dir, args.samples)

    for p, n in [(model_path, args.model), (samples_path, args.samples)]:
        if not os.path.exists(p):
            print(f"Missing {n}. Run train_model.py first.")
            return

    # Load model
    model = SpeakerVerifyNet()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load scaler
    scaler = None
    if os.path.exists(scaler_path):
        s = np.load(scaler_path)
        scaler = {"mean": s["mean"], "scale": s["scale"]}

    # Load test data
    samples = np.load(samples_path)
    X_test, y_test = samples["X"], samples["y"]

    print(f"Evaluating model: {args.model}")
    print(f"Test samples: {len(X_test)} ({int(sum(y_test))} same, "
          f"{int(len(y_test) - sum(y_test))} different)")

    # Compute scores
    scores = compute_scores(model, X_test, scaler)

    # Full evaluation
    report = evaluation_report(scores, y_test.astype(int))

    print(f"\n{'='*50}")
    print(f"  EVALUATION REPORT")
    print(f"{'='*50}")
    print(f"  EER:                    {report['eer']:.4f} ({report['eer']*100:.2f}%)")
    print(f"  EER threshold:          {report['eer_threshold']:.4f}")
    print(f"  Accuracy @ EER:         {report['accuracy_at_eer']:.4f}")
    print(f"  Accuracy @ argmax:      {report['accuracy_at_argmax']:.4f}")
    print(f"  Threshold (1% FAR):     {report['threshold_1pct_far']:.4f}")
    print(f"  Threshold (5% FAR):     {report['threshold_5pct_far']:.4f}")
    print(f"  Genuine samples:        {report['n_genuine']}")
    print(f"  Impostor samples:       {report['n_impostor']}")

    # Save report
    report_path = os.path.join(demo_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved to {report_path}")


if __name__ == "__main__":
    main()
