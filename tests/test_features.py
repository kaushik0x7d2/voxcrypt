"""Unit tests for feature extraction."""

import numpy as np
import soundfile as sf

from speaker_verify.features import (
    extract_mfcc,
    utterance_embedding,
    pair_features,
    audio_to_embedding,
)


class TestExtractMFCC:
    def test_shape(self, synthetic_wav):
        mfcc = extract_mfcc(synthetic_wav)
        assert mfcc.ndim == 2
        assert mfcc.shape[1] == 20  # default n_mfcc

    def test_custom_n_mfcc(self, synthetic_wav):
        mfcc = extract_mfcc(synthetic_wav, n_mfcc=13)
        assert mfcc.shape[1] == 13

    def test_different_lengths(self, tmp_path):
        sr = 16000
        for duration in [0.5, 1.0, 2.0]:
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)
            path = str(tmp_path / f"test_{duration}.wav")
            sf.write(path, audio, sr)
            mfcc = extract_mfcc(path)
            assert mfcc.shape[1] == 20
            assert mfcc.shape[0] > 0


class TestUtteranceEmbedding:
    def test_shape(self, synthetic_wav):
        mfcc = extract_mfcc(synthetic_wav)
        emb = utterance_embedding(mfcc)
        assert emb.shape == (40,)  # mean + std for 20 MFCCs

    def test_shape_custom(self, synthetic_wav):
        mfcc = extract_mfcc(synthetic_wav, n_mfcc=13)
        emb = utterance_embedding(mfcc)
        assert emb.shape == (26,)  # mean + std for 13 MFCCs

    def test_mean_std_content(self):
        # Known input
        mfcc = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        emb = utterance_embedding(mfcc)
        expected_mean = np.mean(mfcc, axis=0)
        expected_std = np.std(mfcc, axis=0)
        np.testing.assert_allclose(emb[:2], expected_mean)
        np.testing.assert_allclose(emb[2:], expected_std)


class TestPairFeatures:
    def test_shape(self):
        emb_a = np.random.randn(40)
        emb_b = np.random.randn(40)
        feat = pair_features(emb_a, emb_b)
        assert feat.shape == (40,)

    def test_absolute_difference(self):
        emb_a = np.array([1.0, -2.0, 3.0])
        emb_b = np.array([4.0, 1.0, -1.0])
        feat = pair_features(emb_a, emb_b)
        np.testing.assert_allclose(feat, [3.0, 3.0, 4.0])

    def test_same_embedding(self):
        emb = np.random.randn(40)
        feat = pair_features(emb, emb)
        np.testing.assert_allclose(feat, np.zeros(40))


class TestAudioToEmbedding:
    def test_convenience(self, synthetic_wav):
        emb = audio_to_embedding(synthetic_wav)
        assert emb.shape == (40,)

    def test_different_speakers(self, speaker_wavs):
        emb_a = audio_to_embedding(speaker_wavs["speaker_A"][0])
        emb_b = audio_to_embedding(speaker_wavs["speaker_B"][0])
        # Different speakers should have different embeddings
        assert not np.allclose(emb_a, emb_b, atol=0.1)
