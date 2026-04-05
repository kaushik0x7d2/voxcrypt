"""
Model artifact management for Orion Voice.

Handles model versioning, manifest generation, and integrity verification.
"""

import hashlib
import json
import os
import time


def compute_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_version(architecture, n_pairs, noise_std, activation="GELU"):
    """Generate a human-readable version string."""
    return f"v1.0.0-{activation.lower()}-{n_pairs}pairs-noise{noise_std}"


class ModelManifest:
    """Model artifact manifest for versioning and integrity."""

    def __init__(self):
        self.version = ""
        self.architecture = ""
        self.input_dim = 0
        self.output_dim = 0
        self.training_config = {}
        self.metrics = {}
        self.file_hashes = {}
        self.created_at = ""
        self.fhe_compatible = True

    @classmethod
    def create(
        cls,
        model_path,
        scaler_path=None,
        samples_path=None,
        architecture="",
        training_config=None,
        metrics=None,
        version=None,
        input_dim=40,
        output_dim=2,
    ):
        """Create a manifest for a set of model artifacts."""
        manifest = cls()
        manifest.version = version or f"v-{int(time.time())}"
        manifest.architecture = architecture
        manifest.input_dim = input_dim
        manifest.output_dim = output_dim
        manifest.training_config = training_config or {}
        manifest.metrics = metrics or {}
        manifest.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        manifest.fhe_compatible = True

        manifest.file_hashes["model"] = compute_file_hash(model_path)
        if scaler_path and os.path.exists(scaler_path):
            manifest.file_hashes["scaler"] = compute_file_hash(scaler_path)
        if samples_path and os.path.exists(samples_path):
            manifest.file_hashes["samples"] = compute_file_hash(samples_path)

        return manifest

    def save(self, path):
        """Save manifest to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        """Load manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)
        manifest = cls()
        manifest.version = data.get("version", "")
        manifest.architecture = data.get("architecture", "")
        manifest.input_dim = data.get("input_dim", 0)
        manifest.output_dim = data.get("output_dim", 0)
        manifest.training_config = data.get("training_config", {})
        manifest.metrics = data.get("metrics", {})
        manifest.file_hashes = data.get("file_hashes", {})
        manifest.created_at = data.get("created_at", "")
        manifest.fhe_compatible = data.get("fhe_compatible", True)
        return manifest

    def to_dict(self):
        return {
            "version": self.version,
            "architecture": self.architecture,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "file_hashes": self.file_hashes,
            "created_at": self.created_at,
            "fhe_compatible": self.fhe_compatible,
        }

    def verify_integrity(self, model_path, scaler_path=None, samples_path=None):
        """
        Verify that artifact files match recorded hashes.

        Returns:
            (True, []) if all files match.
            (False, [error_messages]) if any mismatch.
        """
        errors = []

        if "model" in self.file_hashes:
            if not os.path.exists(model_path):
                errors.append(f"Model file not found: {model_path}")
            else:
                actual = compute_file_hash(model_path)
                if actual != self.file_hashes["model"]:
                    errors.append(
                        f"Model hash mismatch: expected "
                        f"{self.file_hashes['model'][:16]}..., "
                        f"got {actual[:16]}..."
                    )

        if "scaler" in self.file_hashes and scaler_path:
            if not os.path.exists(scaler_path):
                errors.append(f"Scaler file not found: {scaler_path}")
            else:
                actual = compute_file_hash(scaler_path)
                if actual != self.file_hashes["scaler"]:
                    errors.append("Scaler hash mismatch")

        return len(errors) == 0, errors
