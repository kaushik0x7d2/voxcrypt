"""
Generate test WAV assets for quick testing.

Creates 3 short WAV files:
- speaker_A_1.wav, speaker_A_2.wav: same "speaker" (similar frequency)
- speaker_B_1.wav: different "speaker" (different frequency)
"""

import os
import numpy as np
import soundfile as sf


def main():
    sr = 16000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    assets_dir = os.path.dirname(os.path.abspath(__file__))

    # Speaker A: base frequency ~200 Hz (male-like fundamental)
    rng = np.random.RandomState(42)
    audio_a1 = (0.4 * np.sin(2 * np.pi * 200 * t)
                + 0.2 * np.sin(2 * np.pi * 400 * t)
                + 0.1 * np.sin(2 * np.pi * 600 * t)
                + 0.05 * rng.randn(len(t)))
    sf.write(os.path.join(assets_dir, "speaker_A_1.wav"), audio_a1, sr)

    rng = np.random.RandomState(43)
    audio_a2 = (0.4 * np.sin(2 * np.pi * 205 * t)
                + 0.2 * np.sin(2 * np.pi * 410 * t)
                + 0.1 * np.sin(2 * np.pi * 615 * t)
                + 0.05 * rng.randn(len(t)))
    sf.write(os.path.join(assets_dir, "speaker_A_2.wav"), audio_a2, sr)

    # Speaker B: base frequency ~300 Hz (female-like fundamental)
    rng = np.random.RandomState(44)
    audio_b1 = (0.4 * np.sin(2 * np.pi * 300 * t)
                + 0.2 * np.sin(2 * np.pi * 600 * t)
                + 0.1 * np.sin(2 * np.pi * 900 * t)
                + 0.05 * rng.randn(len(t)))
    sf.write(os.path.join(assets_dir, "speaker_B_1.wav"), audio_b1, sr)

    print("Generated:")
    for f in ["speaker_A_1.wav", "speaker_A_2.wav", "speaker_B_1.wav"]:
        path = os.path.join(assets_dir, f)
        print(f"  {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
