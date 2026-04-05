"""
ML evaluation metrics for speaker verification.

Computes FAR, FRR, EER, DET curves, and threshold recommendations.
"""

import numpy as np


def compute_scores(model, X, scaler=None):
    """
    Compute verification scores from model output.

    Returns softmax probability of "same speaker" class.
    """
    import torch

    if scaler is not None:
        X = (X - scaler["mean"]) / scaler["scale"]

    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32)
        output = model(x)
        probs = torch.softmax(output, dim=1)
        # Score = probability of "same speaker" (class 1)
        scores = probs[:, 1].numpy()

    return scores


def compute_far_frr(scores, labels, n_thresholds=1000):
    """
    Compute FAR and FRR across threshold range.

    Args:
        scores: np.ndarray of shape (n,) — verification scores.
        labels: np.ndarray of shape (n,) — binary labels (1=same, 0=different).
        n_thresholds: Number of threshold points.

    Returns:
        dict with keys: thresholds, far, frr
    """
    thresholds = np.linspace(0, 1, n_thresholds)
    far = np.zeros(n_thresholds)
    frr = np.zeros(n_thresholds)

    n_genuine = np.sum(labels == 1)
    n_impostor = np.sum(labels == 0)

    if n_genuine == 0 or n_impostor == 0:
        raise ValueError("Need both genuine and impostor samples")

    for i, t in enumerate(thresholds):
        # FAR: impostor accepted (score >= threshold when label=0)
        far[i] = np.sum((scores >= t) & (labels == 0)) / n_impostor
        # FRR: genuine rejected (score < threshold when label=1)
        frr[i] = np.sum((scores < t) & (labels == 1)) / n_genuine

    return {"thresholds": thresholds, "far": far, "frr": frr}


def find_eer(scores, labels, n_thresholds=10000):
    """
    Find the Equal Error Rate (EER) — the point where FAR == FRR.

    Returns:
        (eer, threshold): The EER value and the threshold that achieves it.
    """
    result = compute_far_frr(scores, labels, n_thresholds)
    far, frr = result["far"], result["frr"]
    thresholds = result["thresholds"]

    # Find crossing point
    diff = far - frr
    idx = np.argmin(np.abs(diff))

    eer = (far[idx] + frr[idx]) / 2
    return float(eer), float(thresholds[idx])


def recommend_threshold(scores, labels, target_far=0.01):
    """
    Recommend a threshold for a target FAR.

    Args:
        scores: Verification scores.
        labels: Binary labels.
        target_far: Target false accept rate (default 1%).

    Returns:
        threshold: The recommended threshold.
    """
    result = compute_far_frr(scores, labels, n_thresholds=10000)
    far = result["far"]
    thresholds = result["thresholds"]

    # Find the threshold that gives FAR closest to target
    idx = np.argmin(np.abs(far - target_far))
    return float(thresholds[idx])


def compute_det_curve(scores, labels, n_thresholds=1000):
    """
    Compute Detection Error Tradeoff (DET) curve data.

    Returns:
        dict with far, frr arrays for plotting.
    """
    result = compute_far_frr(scores, labels, n_thresholds)
    return {"far": result["far"], "frr": result["frr"]}


def evaluation_report(scores, labels):
    """
    Generate a comprehensive evaluation report.

    Returns:
        dict with all evaluation metrics.
    """
    eer, eer_threshold = find_eer(scores, labels)

    # Thresholds at various operating points
    threshold_1pct = recommend_threshold(scores, labels, target_far=0.01)
    threshold_5pct = recommend_threshold(scores, labels, target_far=0.05)

    # Accuracy at EER threshold
    preds_eer = (scores >= eer_threshold).astype(int)
    accuracy_eer = np.mean(preds_eer == labels)

    # Accuracy at argmax (threshold=0.5)
    preds_argmax = (scores >= 0.5).astype(int)
    accuracy_argmax = np.mean(preds_argmax == labels)

    return {
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "accuracy_at_eer": float(accuracy_eer),
        "accuracy_at_argmax": float(accuracy_argmax),
        "threshold_1pct_far": float(threshold_1pct),
        "threshold_5pct_far": float(threshold_5pct),
        "n_genuine": int(np.sum(labels == 1)),
        "n_impostor": int(np.sum(labels == 0)),
        "n_total": len(labels),
    }
