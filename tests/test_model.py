"""Unit tests for the speaker verification model."""

import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "orion", "repo"))

on = pytest.importorskip("orion.nn")
from speaker_verify.model import SpeakerVerifyNet  # noqa: E402


class TestSpeakerVerifyNet:
    def test_forward_shape(self):
        model = SpeakerVerifyNet(input_dim=40)
        x = torch.randn(1, 40)
        out = model(x)
        assert out.shape == (1, 2)

    def test_batch_forward(self):
        model = SpeakerVerifyNet(input_dim=40)
        x = torch.randn(8, 40)
        out = model(x)
        assert out.shape == (8, 2)

    def test_is_orion_module(self):
        model = SpeakerVerifyNet()
        assert isinstance(model, on.Module)

    def test_gradient_flow(self):
        model = SpeakerVerifyNet(input_dim=40)
        x = torch.randn(4, 40)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # All parameters should have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_custom_input_dim(self):
        model = SpeakerVerifyNet(input_dim=26)
        x = torch.randn(1, 26)
        out = model(x)
        assert out.shape == (1, 2)
