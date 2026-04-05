"""
Neural network models for encrypted audio tasks.

All models use orion.nn layers so they can run under FHE with zero code changes.
Uses GELU activation (matching Orion's cancer demo which achieves 95% FHE accuracy).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))
import orion.nn as on


class SpeakerVerifyNet(on.Module):
    """
    MLP for speaker verification (same/different speaker).
    40 features -> 128 -> 64 -> 2

    Architecture matches Orion's proven cancer demo (30->128->64->2, GELU).
    GELU provides better FHE numerical stability than SiLU.
    """
    def __init__(self, input_dim=40):
        super().__init__()
        self.fc1 = on.Linear(input_dim, 128)
        self.act1 = on.GELU()
        self.fc2 = on.Linear(128, 64)
        self.act2 = on.GELU()
        self.fc3 = on.Linear(64, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


class SpeakerIDNet(on.Module):
    """
    MLP for speaker identification (who is speaking?).
    40 features -> 128 -> 64 -> n_speakers
    """
    def __init__(self, input_dim=40, n_speakers=40):
        super().__init__()
        self.fc1 = on.Linear(input_dim, 128)
        self.act1 = on.GELU()
        self.fc2 = on.Linear(128, 64)
        self.act2 = on.GELU()
        self.fc3 = on.Linear(64, n_speakers)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


class GenderNet(on.Module):
    """
    MLP for gender classification from voice.
    40 features -> 64 -> 32 -> 2 (male/female)
    """
    def __init__(self, input_dim=40):
        super().__init__()
        self.fc1 = on.Linear(input_dim, 64)
        self.act1 = on.GELU()
        self.fc2 = on.Linear(64, 32)
        self.act2 = on.GELU()
        self.fc3 = on.Linear(32, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x


class EmotionNet(on.Module):
    """
    MLP for emotion detection from voice.
    40 features -> 128 -> 64 -> n_emotions
    """
    def __init__(self, input_dim=40, n_emotions=7):
        super().__init__()
        self.fc1 = on.Linear(input_dim, 128)
        self.act1 = on.GELU()
        self.fc2 = on.Linear(128, 64)
        self.act2 = on.GELU()
        self.fc3 = on.Linear(64, n_emotions)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x
