"""
Speaker verification model.

Small MLP using orion.nn layers so it can run under FHE with zero code changes.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "orion", "repo"))
import orion.nn as on


class SpeakerVerifyNet(on.Module):
    """
    MLP for speaker verification.
    40 features -> 128 -> 64 -> 2 (same/different speaker)

    Uses SiLU activation with degree=7 polynomial approximation for FHE.
    """
    def __init__(self, input_dim=40):
        super().__init__()
        self.fc1 = on.Linear(input_dim, 128)
        self.act1 = on.SiLU(degree=7)
        self.fc2 = on.Linear(128, 64)
        self.act2 = on.SiLU(degree=7)
        self.fc3 = on.Linear(64, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc3(x)
        return x
