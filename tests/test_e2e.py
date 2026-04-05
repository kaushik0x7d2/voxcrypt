"""
End-to-end tests for the full speaker verification pipeline.

These tests are SLOW (~5 min) and cover the complete flow.
Skip with: pytest -m "not e2e"
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))


@pytest.mark.e2e
class TestEndToEnd:
    def test_synthetic_pipeline(self, trained_model, fhe_config):
        """Full pipeline: synthetic data -> train -> FHE inference."""
        from speaker_verify.fhe_inference import init_fhe, fhe_predict, cleanup

        model = trained_model["model"]
        metrics = trained_model["metrics"]

        # Verify training achieved reasonable accuracy
        assert metrics["val_acc"] >= 0.6, (
            f"Val accuracy too low: {metrics['val_acc']:.2%}"
        )

        # Run FHE inference
        scheme, input_level = init_fhe(fhe_config, model, trained_model["sample_path"])

        X_val = trained_model["X_val"]
        y_val = trained_model["y_val"]
        n_test = min(5, len(X_val))

        correct = 0
        total_time = 0

        for i in range(n_test):
            sample = torch.tensor(X_val[i : i + 1], dtype=torch.float32)
            actual = int(y_val[i])
            pred, _, elapsed, bits = fhe_predict(model, sample, input_level)

            if pred == actual:
                correct += 1
            total_time += elapsed

            assert bits > 0, "Precision bits should be positive"

        acc = correct / n_test
        avg_time = total_time / n_test

        print(f"\nE2E Results: {correct}/{n_test} ({acc:.0%})")
        print(f"Avg FHE time: {avg_time:.1f}s")

        cleanup(scheme)

    @pytest.mark.e2e
    def test_feature_to_model_flow(self, speaker_wavs):
        """Test: audio files -> features -> model prediction."""
        from speaker_verify.features import audio_to_embedding, pair_features
        from speaker_verify.model import SpeakerVerifyNet

        model = SpeakerVerifyNet(input_dim=40)
        model.eval()

        # Same speaker
        emb_a1 = audio_to_embedding(speaker_wavs["speaker_A"][0])
        emb_a2 = audio_to_embedding(speaker_wavs["speaker_A"][1])
        feat_same = pair_features(emb_a1, emb_a2)

        # Different speaker
        emb_b1 = audio_to_embedding(speaker_wavs["speaker_B"][0])
        feat_diff = pair_features(emb_a1, emb_b1)

        # Model should produce valid outputs (untrained, just checking shape/type)
        with torch.no_grad():
            out_same = model(torch.tensor(feat_same[None], dtype=torch.float32))
            out_diff = model(torch.tensor(feat_diff[None], dtype=torch.float32))

        assert out_same.shape == (1, 2)
        assert out_diff.shape == (1, 2)
