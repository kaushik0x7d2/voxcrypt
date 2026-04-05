"""
Integration tests for the FHE pipeline.

These tests are SLOW (~2-3 min) as they involve actual FHE operations.
Skip with: pytest -m "not slow"
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))


@pytest.mark.slow
class TestFHEPipeline:
    def test_init_and_compile(self, trained_model, fhe_config):
        from speaker_verify.fhe_inference import init_fhe, cleanup

        scheme, input_level = init_fhe(
            fhe_config,
            trained_model["model"],
            trained_model["sample_path"],
        )
        assert input_level > 0
        cleanup(scheme)

    def test_fhe_predict(self, trained_model, fhe_config):
        from speaker_verify.fhe_inference import init_fhe, fhe_predict, cleanup

        model = trained_model["model"]
        scheme, input_level = init_fhe(fhe_config, model, trained_model["sample_path"])

        sample = torch.tensor(trained_model["X_val"][:1], dtype=torch.float32)
        pred, fhe_out, elapsed, bits = fhe_predict(model, sample, input_level)

        assert pred in (0, 1)
        assert elapsed > 0
        assert bits > 0

        cleanup(scheme)

    def test_fhe_matches_cleartext(self, trained_model, fhe_config):
        from speaker_verify.fhe_inference import init_fhe, fhe_predict, cleanup

        model = trained_model["model"]
        scheme, input_level = init_fhe(fhe_config, model, trained_model["sample_path"])

        X_val = trained_model["X_val"]
        n_test = min(5, len(X_val))
        matches = 0

        for i in range(n_test):
            sample = torch.tensor(X_val[i : i + 1], dtype=torch.float32)

            # Cleartext
            model.eval()
            with torch.no_grad():
                clear_pred = model(sample).argmax(dim=1).item()

            # FHE
            fhe_pred, _, _, _ = fhe_predict(model, sample, input_level)

            if fhe_pred == clear_pred:
                matches += 1

        # At least 60% should match (synthetic data has tighter margins
        # than real speech, so FHE precision loss can flip more predictions)
        assert matches / n_test >= 0.6, (
            f"FHE-Clear agreement too low: {matches}/{n_test}"
        )

        cleanup(scheme)
