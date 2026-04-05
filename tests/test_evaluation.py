"""Tests for ML evaluation metrics."""

import numpy as np
import pytest
from speaker_verify.evaluation import (
    compute_far_frr, find_eer, recommend_threshold, evaluation_report)


class TestFARFRR:
    def test_perfect_separation(self):
        # Impostors all at 0, genuine all at 1
        scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = compute_far_frr(scores, labels)
        # At threshold 0.5: FAR=0, FRR=0
        mid = len(result["thresholds"]) // 2
        assert result["far"][mid] == 0.0
        assert result["frr"][mid] == 0.0

    def test_random_scores(self):
        np.random.seed(42)
        scores = np.random.rand(100)
        labels = np.array([0] * 50 + [1] * 50)
        result = compute_far_frr(scores, labels)
        assert len(result["thresholds"]) == 1000
        assert len(result["far"]) == 1000
        assert len(result["frr"]) == 1000

    def test_needs_both_classes(self):
        with pytest.raises(ValueError):
            compute_far_frr(np.array([0.5, 0.5]),
                            np.array([1, 1]))


class TestEER:
    def test_perfect_separation(self):
        scores = np.concatenate([np.zeros(50), np.ones(50)])
        labels = np.array([0] * 50 + [1] * 50)
        eer, threshold = find_eer(scores, labels)
        assert eer < 0.05  # Near-zero EER

    def test_random_gives_high_eer(self):
        np.random.seed(42)
        scores = np.random.rand(200)
        labels = np.array([0] * 100 + [1] * 100)
        eer, _ = find_eer(scores, labels)
        assert 0.3 < eer < 0.7  # ~50% EER for random

    def test_returns_valid_threshold(self):
        scores = np.concatenate([
            np.random.normal(0.3, 0.1, 50),
            np.random.normal(0.7, 0.1, 50)])
        labels = np.array([0] * 50 + [1] * 50)
        _, threshold = find_eer(scores, labels)
        assert 0 <= threshold <= 1


class TestRecommendThreshold:
    def test_low_far_target(self):
        scores = np.concatenate([
            np.random.normal(0.2, 0.1, 100),
            np.random.normal(0.8, 0.1, 100)])
        labels = np.array([0] * 100 + [1] * 100)
        threshold = recommend_threshold(scores, labels, target_far=0.01)
        assert 0 < threshold < 1


class TestEvaluationReport:
    def test_report_keys(self):
        np.random.seed(42)
        scores = np.concatenate([
            np.random.normal(0.3, 0.15, 50),
            np.random.normal(0.7, 0.15, 50)])
        labels = np.array([0] * 50 + [1] * 50)
        report = evaluation_report(scores, labels)
        assert "eer" in report
        assert "eer_threshold" in report
        assert "accuracy_at_eer" in report
        assert "accuracy_at_argmax" in report
        assert "n_genuine" in report
        assert "n_impostor" in report
        assert report["n_genuine"] == 50
        assert report["n_impostor"] == 50
