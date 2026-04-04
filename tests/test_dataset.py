"""Unit tests for dataset utilities."""

import os

import numpy as np

from speaker_verify.dataset import scan_speakers, generate_pairs, build_dataset


class TestScanSpeakers:
    def test_finds_speakers(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        assert len(speakers) == 2
        assert "1001" in speakers
        assert "1002" in speakers

    def test_finds_utterances(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        # 2 chapters x 2 utterances = 4 per speaker
        assert len(speakers["1001"]) == 4
        assert len(speakers["1002"]) == 4

    def test_all_flac(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        for paths in speakers.values():
            for p in paths:
                assert p.endswith(".flac")


class TestGeneratePairs:
    def test_correct_count(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        pairs = generate_pairs(speakers, n_pairs=20)
        assert len(pairs) == 20

    def test_balanced(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        pairs = generate_pairs(speakers, n_pairs=100)
        n_same = sum(1 for _, _, l in pairs if l == 1)
        n_diff = sum(1 for _, _, l in pairs if l == 0)
        assert n_same == 50
        assert n_diff == 50

    def test_labels_correct(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        pairs = generate_pairs(speakers, n_pairs=50)
        for path_a, path_b, label in pairs:
            # Extract speaker IDs from paths
            spk_a = None
            spk_b = None
            for sid in speakers:
                if any(path_a == p for p in speakers[sid]):
                    spk_a = sid
                if any(path_b == p for p in speakers[sid]):
                    spk_b = sid
            if label == 1:
                assert spk_a == spk_b, "Same-speaker pair has different speakers"
            else:
                assert spk_a != spk_b, "Different-speaker pair has same speaker"

    def test_reproducible(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        pairs1 = generate_pairs(speakers, n_pairs=20, seed=42)
        pairs2 = generate_pairs(speakers, n_pairs=20, seed=42)
        assert pairs1 == pairs2


class TestBuildDataset:
    def test_shape(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        pairs = generate_pairs(speakers, n_pairs=10)
        X, y = build_dataset(pairs)
        assert X.shape == (10, 40)  # 20 MFCCs * 2 (mean + std)
        assert y.shape == (10,)

    def test_labels_preserved(self, mock_librispeech):
        speakers = scan_speakers(mock_librispeech)
        pairs = generate_pairs(speakers, n_pairs=10)
        _, y = build_dataset(pairs)
        expected = np.array([l for _, _, l in pairs], dtype=np.float32)
        np.testing.assert_array_equal(y, expected)
