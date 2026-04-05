# Orion Voice — Encrypted Speaker Verification

## What This Project Is

End-to-end encrypted speaker verification using FHE (Fully Homomorphic Encryption). A client extracts audio features locally, encrypts them, and sends to a server that verifies speaker identity on encrypted data — without ever hearing the audio or seeing the result.

**This is a standalone project. It is NOT part of the Orion repo.** It uses Orion as a dependency (installed from the local repo at `C:/Users/Unitech/Desktop/Kaushik/orion/repo`).

## Owner

Kaushik Kachireddy (GitHub: kaushik0x7d2). AI engineer. This is a demo/portfolio project, not targeting a paper.

## Architecture

```
Client                              Server
┌─────────────────┐                ┌──────────────────────┐
│ 1. Load audio    │                │                      │
│ 2. Extract MFCCs │                │                      │
│ 3. Compute       │                │                      │
│    |emb_A-emb_B| │                │                      │
│ 4. Encrypt       │──ciphertext──>│ 5. FHE inference     │
│    (40-dim vec)  │                │    (MLP on encrypted │
│                  │<──ciphertext──│     features)         │
│ 6. Decrypt       │                │    Never sees audio  │
│ 7. Same/Diff     │                │    or result         │
└─────────────────┘                └──────────────────────┘
```

## Pipeline

1. **Feature extraction** (client-side, plaintext):
   - Load audio at 16kHz mono
   - Extract 20 MFCCs per frame using librosa
   - Compute mean + std across frames → 40-dim embedding per utterance
   - For verification: compute |emb_A - emb_B| → 40-dim feature vector
   - Normalize with saved StandardScaler

2. **FHE inference** (server-side, encrypted):
   - Model: `SpeakerVerifyNet` — 40→128→64→2 MLP with GELU
   - CKKS scheme via Orion/Lattigo backend
   - Config: reuse heart_config.yml params (LogN=14, same depth profile)
   - Expected FHE time: ~15-20s per verification pair

3. **Training** (offline, plaintext):
   - Dataset: LibriSpeech test-clean (~20 speakers, ~10 utterances each)
   - Generate balanced same-speaker / different-speaker pairs
   - Train with Adam + CrossEntropyLoss
   - Target: >80% cleartext accuracy (MFCC-based, intentionally simple for FHE)

## Project Structure

```
orion-voice/
├── CLAUDE.md                          # This file
├── README.md
├── pyproject.toml
├── .gitignore
├── LICENSE
│
├── configs/
│   └── fhe_config.yml                 # CKKS parameters (same as Orion heart_config.yml)
│
├── speaker_verify/                    # Main package
│   ├── __init__.py
│   ├── features.py                    # MFCC extraction, utterance embedding, pair features
│   ├── dataset.py                     # LibriSpeech download, pair generation
│   ├── model.py                       # SpeakerVerifyNet (orion.nn.Module)
│   ├── train.py                       # Training loop
│   └── fhe_inference.py               # FHE pipeline: init, fit, compile, encrypt, infer, decrypt
│
├── demo/
│   ├── train_model.py                 # CLI: download data, extract features, train
│   ├── fhe_demo.py                    # CLI: end-to-end FHE demo
│   ├── server.py                      # Flask server for encrypted verification
│   ├── client.py                      # Client: load audio, encrypt, send, decrypt
│   └── benchmark.py                   # Benchmarks
│
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   ├── test_fhe_pipeline.py
│   └── test_dataset.py
│
└── assets/                            # 2-3 short WAV files for quick testing
    ├── speaker_A_1.wav
    ├── speaker_A_2.wav
    └── speaker_B_1.wav
```

## Key Design Decisions

- **Feature: |MFCC_A - MFCC_B| (40-dim)** not concatenation (80-dim). Halves input size → faster FHE. Cleaner privacy semantics.
- **Activation: GELU**. Orion's cancer demo achieves 95% FHE accuracy with GELU. SiLU(degree=7) gives poor FHE precision (-3 bits) while GELU gives positive precision (+4-6 bits).
- **Model depth: 3 Linear + 2 GELU = ~13 multiplicative levels**. Fits within LogN=14 CKKS config.
- **Dataset: LibriSpeech test-clean** (~350MB, auto-download via torchaudio). No registration needed unlike VoxCeleb.
- **Simple MFCC features** (not deep embeddings). Intentional — keeps model small enough for FHE. Accuracy will be 75-85%, not SOTA. The point is encrypted inference, not SOTA speaker verification.

## Dependencies

```
orion (local: C:/Users/Unitech/Desktop/Kaushik/orion/repo)
torch>=2.2.0
torchaudio>=2.2.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
PyYAML>=6.0
tqdm>=4.30.0
flask>=3.0.0 (optional, for server demo)
pytest>=7.0.0 (dev)
```

## Patterns to Follow

All patterns come from Orion's existing demos. Reference files:

- **Training**: Follow `orion/repo/demo/train_cancer.py` — orion.nn.Module model, StandardScaler, CrossEntropyLoss, save model.pt + scaler.npz + test_samples.npz
- **FHE inference**: Follow `orion/repo/demo/cancer_fhe_inference.py` — orion.init_scheme, orion.fit, orion.compile, orion.encode, orion.encrypt, model.he(), model(ctxt), ctxt.decrypt().decode()
- **Server**: Follow `orion/repo/demo/server.py` — Flask app, /info and /predict endpoints, base64 ciphertext serialization
- **Client**: Follow `orion/repo/demo/client.py` — setup_client with key loading, encrypt, send, decrypt
- **Config**: Copy `orion/repo/demo/heart_config.yml` verbatim for fhe_config.yml

## Implementation Order

1. Project skeleton: pyproject.toml, .gitignore, configs/fhe_config.yml
2. `speaker_verify/features.py` — MFCC extraction + pair features
3. `speaker_verify/dataset.py` — LibriSpeech download + pair generation
4. `speaker_verify/model.py` — SpeakerVerifyNet
5. `speaker_verify/train.py` + `demo/train_model.py` — training pipeline
6. `speaker_verify/fhe_inference.py` + `demo/fhe_demo.py` — FHE pipeline
7. `demo/server.py` + `demo/client.py` — client-server
8. `tests/` — all test files
9. `demo/benchmark.py` — benchmarks
10. `README.md` — documentation
11. Add 2-3 short WAV assets for quick testing

## Environment

- Python: `C:/Users/Unitech/AppData/Local/Programs/Python/Python313/python.exe`
- Go backend (for Orion FHE): must be built in `orion/repo/orion/backend/lattigo/`
- Platform: Windows 11
- GitHub: kaushik0x7d2
- gh CLI: `export PATH="/c/Program Files/GitHub CLI:$PATH"`

## Model Definition

```python
import orion.nn as on

class SpeakerVerifyNet(on.Module):
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
        return self.fc3(x)
```

## Expected Demo Output

```
=== Cleartext Inference (10 pairs) ===
  Pair  1/10: Same Speaker      (actual: Same Speaker)      [   ok]
  Pair  2/10: Diff Speaker      (actual: Diff Speaker)      [   ok]
  ...

=== FHE Inference (10 pairs) ===
  Pair  1/10: Same Speaker      (actual: Same Speaker)      [   ok] | 16.2s | 1.9 bits
  Pair  2/10: Diff Speaker      (actual: Diff Speaker)      [   ok] | 15.8s | 1.8 bits
  ...

FHE Accuracy: 9/10 (90%)
Average FHE time: 16.1s per pair
```
