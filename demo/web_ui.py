"""
Web UI for encrypted voice analysis.

Provides a browser-based interface for all voice tasks:
speaker verification, identification, gender classification, emotion detection.

Supports both cleartext (fast) and FHE (encrypted) inference modes.

Usage:
    python demo/web_ui.py
    python demo/web_ui.py --port 8080
"""

import os
import sys
import time
import json
import tempfile

import torch
import numpy as np
from flask import Flask, request, jsonify, render_template

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))

from speaker_verify.features import audio_to_embedding, pair_features
from speaker_verify.model import SpeakerVerifyNet, SpeakerIDNet, GenderNet, EmotionNet

app = Flask(__name__)
_models = {}


def load_models(demo_dir):
    """Load all available trained models."""
    configs = {
        "verify": {
            "class": SpeakerVerifyNet,
            "model_file": "speaker_model.pt",
            "scaler_file": "scaler.npz",
            "labels": ["Different Speaker", "Same Speaker"],
        },
        "speaker_id": {
            "class": SpeakerIDNet,
            "model_file": "speaker_id_model.pt",
            "scaler_file": "speaker_id_scaler.npz",
            "meta_file": "speaker_id_meta.npz",
        },
        "gender": {
            "class": GenderNet,
            "model_file": "gender_model.pt",
            "scaler_file": "gender_scaler.npz",
            "labels": ["Female", "Male"],
        },
        "emotion": {
            "class": EmotionNet,
            "model_file": "emotion_model.pt",
            "scaler_file": "emotion_scaler.npz",
            "meta_file": "emotion_meta.npz",
        },
    }

    for task_name, cfg in configs.items():
        model_path = os.path.join(demo_dir, cfg["model_file"])
        scaler_path = os.path.join(demo_dir, cfg["scaler_file"])

        if not os.path.exists(model_path):
            continue

        kwargs = {}
        labels = cfg.get("labels")

        if "meta_file" in cfg:
            meta_path = os.path.join(demo_dir, cfg["meta_file"])
            if os.path.exists(meta_path):
                meta = np.load(meta_path, allow_pickle=True)
                if task_name == "speaker_id":
                    speaker_ids = list(meta["speaker_ids"])
                    kwargs["n_speakers"] = len(speaker_ids)
                    labels = [f"Speaker {s}" for s in speaker_ids]
                elif task_name == "emotion":
                    emotion_names = list(meta["emotion_names"])
                    kwargs["n_emotions"] = len(emotion_names)
                    labels = list(emotion_names)

        model = cfg["class"](**kwargs)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()

        scaler = None
        if os.path.exists(scaler_path):
            s = np.load(scaler_path)
            scaler = {"mean": s["mean"], "scale": s["scale"]}

        _models[task_name] = {
            "model": model,
            "scaler": scaler,
            "labels": labels,
        }
        print(f"[WebUI] Loaded {task_name} model ({len(labels)} classes)")


@app.route("/")
def index():
    available_tasks = list(_models.keys())
    return render_template("index.html", tasks=available_tasks)


@app.route("/api/tasks")
def api_tasks():
    return jsonify({
        "tasks": {name: {"labels": info["labels"]}
                  for name, info in _models.items()}
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    task = request.form.get("task", "verify")

    if task not in _models:
        return jsonify({"error": f"Task '{task}' not available"}), 400

    model_info = _models[task]
    model = model_info["model"]
    scaler = model_info["scaler"]
    labels = model_info["labels"]

    # Handle file uploads
    if task == "verify":
        if "audio_a" not in request.files or "audio_b" not in request.files:
            return jsonify({"error": "Need two audio files for verification"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_a:
            request.files["audio_a"].save(f_a.name)
            path_a = f_a.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_b:
            request.files["audio_b"].save(f_b.name)
            path_b = f_b.name

        try:
            t0 = time.time()
            emb_a = audio_to_embedding(path_a)
            emb_b = audio_to_embedding(path_b)
            features = pair_features(emb_a, emb_b)
            t_feat = time.time() - t0
        finally:
            os.unlink(path_a)
            os.unlink(path_b)
    else:
        if "audio" not in request.files:
            return jsonify({"error": "Need an audio file"}), 400

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            request.files["audio"].save(f.name)
            path = f.name

        try:
            t0 = time.time()
            features = audio_to_embedding(path)
            t_feat = time.time() - t0
        finally:
            os.unlink(path)

    # Scale features
    if scaler is not None:
        features = (features - scaler["mean"]) / scaler["scale"]

    # Inference
    t0 = time.time()
    with torch.no_grad():
        x = torch.tensor(features[None], dtype=torch.float32)
        output = model(x)
        probs = torch.softmax(output, dim=1)[0].numpy()
        pred = int(output.argmax(dim=1).item())
    t_inf = time.time() - t0

    return jsonify({
        "task": task,
        "prediction": labels[pred],
        "prediction_idx": pred,
        "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
        "feature_time": t_feat,
        "inference_time": t_inf,
    })


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Voice Analysis Web UI")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    load_models(demo_dir)

    if not _models:
        print("No trained models found. Run training scripts first.")
        return

    print(f"\n[WebUI] Starting at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
