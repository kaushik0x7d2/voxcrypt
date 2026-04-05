"""
Unified FHE demo — runs encrypted inference for all available tasks.

Demonstrates FHE on: speaker verification, speaker ID, gender, emotion.

Usage:
    python demo/fhe_multi_demo.py
    python demo/fhe_multi_demo.py --task gender --num-samples 5
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


TASKS = {
    "verify": {
        "model_file": "speaker_model.pt",
        "sample_file": "test_samples.npz",
        "labels": ["Diff Speaker", "Same Speaker"],
    },
    "speaker_id": {
        "model_file": "speaker_id_model.pt",
        "sample_file": "speaker_id_test_samples.npz",
        "meta_file": "speaker_id_meta.npz",
    },
    "gender": {
        "model_file": "gender_model.pt",
        "sample_file": "gender_test_samples.npz",
        "labels": ["Female", "Male"],
    },
    "emotion": {
        "model_file": "emotion_model.pt",
        "sample_file": "emotion_test_samples.npz",
        "meta_file": "emotion_meta.npz",
    },
}


def load_model_for_task(task_name, demo_dir):
    """Load the appropriate model and labels for a task."""
    task = TASKS[task_name]

    if task_name == "verify":
        from speaker_verify.model import SpeakerVerifyNet

        model = SpeakerVerifyNet()
    elif task_name == "speaker_id":
        from speaker_verify.model import SpeakerIDNet

        meta = np.load(os.path.join(demo_dir, task["meta_file"]), allow_pickle=True)
        speaker_ids = list(meta["speaker_ids"])
        model = SpeakerIDNet(n_speakers=len(speaker_ids))
        task["labels"] = [f"Speaker {s}" for s in speaker_ids]
    elif task_name == "gender":
        from speaker_verify.model import GenderNet

        model = GenderNet()
    elif task_name == "emotion":
        from speaker_verify.model import EmotionNet

        meta = np.load(os.path.join(demo_dir, task["meta_file"]), allow_pickle=True)
        emotion_names = list(meta["emotion_names"])
        model = EmotionNet(n_emotions=len(emotion_names))
        task["labels"] = emotion_names

    model.load_state_dict(
        torch.load(os.path.join(demo_dir, task["model_file"]), weights_only=True)
    )
    model.eval()

    samples = np.load(os.path.join(demo_dir, task["sample_file"]))
    X_test, y_test = samples["X"], samples["y"]
    labels = task["labels"]

    return model, X_test, y_test, labels


def run_fhe_task(task_name, demo_dir, config_path, num_samples=10):
    """Run cleartext + FHE inference for a task."""
    model, X_test, y_test, labels = load_model_for_task(task_name, demo_dir)
    num_samples = min(num_samples, len(X_test))

    # Cleartext
    print(f"\n=== {task_name.upper()}: Cleartext ({num_samples} samples) ===")
    X_all = torch.tensor(X_test[:num_samples], dtype=torch.float32)
    with torch.no_grad():
        clear_out = model(X_all)
        clear_preds = clear_out.argmax(dim=1).numpy()

    correct_clear = sum(
        1 for i in range(num_samples) if clear_preds[i] == int(y_test[i])
    )
    for i in range(num_samples):
        pred = clear_preds[i]
        actual = int(y_test[i])
        status = "ok" if pred == actual else "WRONG"
        print(
            f"  {i + 1:2d}/{num_samples}: {labels[pred]:<16s} "
            f"(actual: {labels[actual]:<16s}) [{status:>5s}]"
        )
    print(f"  Cleartext Accuracy: {correct_clear}/{num_samples}")

    # FHE
    print(f"\n=== {task_name.upper()}: FHE ({num_samples} samples) ===")

    print("  Initializing scheme...")
    scheme = orion.init_scheme(config_path)

    print("  Fitting...")
    fit_X = torch.tensor(X_test, dtype=torch.float32)
    fit_dataset = TensorDataset(fit_X, torch.zeros(len(fit_X)))
    orion.fit(model, DataLoader(fit_dataset, batch_size=32))

    print("  Compiling...")
    input_level = orion.compile(model)
    print(f"  Input level: {input_level}")

    results = []
    times = []

    for i in range(num_samples):
        sample = torch.tensor(X_test[i : i + 1], dtype=torch.float32)
        actual = int(y_test[i])

        ptxt = orion.encode(sample, input_level)
        ctxt = orion.encrypt(ptxt)
        model.he()

        t0 = time.time()
        out_ctxt = model(ctxt)
        t_inf = time.time() - t0

        out_fhe = out_ctxt.decrypt().decode()
        n_out = len(labels)
        fhe_out = out_fhe.flatten()[:n_out]
        pred = fhe_out.argmax().item()
        results.append(pred)
        times.append(t_inf)

        model.eval()
        with torch.no_grad():
            c_out = model(sample).flatten()[:n_out]
        mae = (c_out - fhe_out).abs().mean().item()
        bits = -math.log2(mae) if mae > 0 else float("inf")

        status = "ok" if pred == actual else "WRONG"
        print(
            f"  {i + 1:2d}/{num_samples}: {labels[pred]:<16s} "
            f"(actual: {labels[actual]:<16s}) [{status:>5s}] "
            f"| {t_inf:.1f}s | {bits:.1f} bits"
        )

    correct_fhe = sum(1 for i in range(len(results)) if results[i] == int(y_test[i]))
    agree = sum(1 for i in range(len(results)) if results[i] == clear_preds[i])

    print(
        f"\n  FHE Accuracy:        {correct_fhe}/{num_samples} ({correct_fhe / num_samples:.0%})"
    )
    print(f"  FHE-Clear Agreement: {agree}/{num_samples}")
    print(f"  Avg Inference Time:  {np.mean(times):.1f}s")

    scheme.delete_scheme()
    return correct_fhe, num_samples, agree


def main():
    parser = argparse.ArgumentParser(description="Multi-Task FHE Demo")
    parser.add_argument("--task", choices=list(TASKS.keys()) + ["all"], default="all")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(demo_dir, "..", "configs", "fhe_config.yml")

    tasks_to_run = list(TASKS.keys()) if args.task == "all" else [args.task]

    # Filter to available tasks
    available = []
    for task in tasks_to_run:
        model_file = os.path.join(demo_dir, TASKS[task]["model_file"])
        if os.path.exists(model_file):
            available.append(task)
        else:
            print(f"Skipping {task} — {TASKS[task]['model_file']} not found")

    if not available:
        print("No trained models found. Run the training scripts first.")
        return

    print(f"Running FHE demo for: {', '.join(available)}")

    summary = {}
    for task in available:
        correct, total, agree = run_fhe_task(
            task, demo_dir, config_path, args.num_samples
        )
        summary[task] = (correct, total, agree)

    print(f"\n{'=' * 50}")
    print("  SUMMARY")
    print(f"{'=' * 50}")
    for task, (correct, total, agree) in summary.items():
        print(
            f"  {task:<12s}: FHE {correct}/{total} ({correct / total:.0%}) | "
            f"Agreement {agree}/{total}"
        )


if __name__ == "__main__":
    main()
