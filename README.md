# VoxCrypt — Encrypted Voice Analysis

End-to-end encrypted voice analysis using Fully Homomorphic Encryption (FHE). Supports speaker verification, speaker identification, gender classification, and emotion detection — all on **encrypted data**. The server never hears the audio or sees the result.

## How It Works

```
Client                              Server
┌─────────────────┐                ┌──────────────────────┐
│ 1. Load audio    │                │                      │
│ 2. Extract MFCCs │                │                      │
│ 3. Compute       │                │                      │
│    features      │                │                      │
│ 4. Encrypt       │──ciphertext──>│ 5. FHE inference     │
│                  │                │    (MLP on encrypted │
│                  │<──ciphertext──│     features)         │
│ 6. Decrypt       │                │    Never sees audio  │
│ 7. Result        │                │    or result         │
└─────────────────┘                └──────────────────────┘
```

## Results

| Task | Cleartext Accuracy | FHE Accuracy | FHE-Clear Agreement | FHE Time |
|------|-------------------|-------------|--------------------|---------|
| Speaker Verification | 92.4% | 100% (10/10) | 10/10 | ~18s/pair |
| Encrypted Template Protection | 96.2% | 100% (10/10) | 10/10 | ~9s/pair |
| Gender Classification | 99.6% | 100% (10/10) | 10/10 | ~5s/sample |
| Speaker Identification | 99.2% | — | — | — |
| Emotion Detection | 82.2% | — | — | — |

*Note: FHE accuracy is reported on a 10-sample evaluation subset.*

## Setup

```bash
# Install dependencies
pip install -e .

# Install with server support
pip install -e ".[server]"

# Install with dev dependencies
pip install -e ".[dev]"
```

Requires [Orion](https://github.com/kaushik0x7d2/orion) installed locally.

## Quick Start

### Train Models

```bash
# Speaker verification (downloads LibriSpeech test-clean ~350MB)
python demo/train_model.py --n-pairs 5000 --noise-std 0.3

# Speaker identification
python demo/train_speaker_id.py

# Gender classification
python demo/train_gender.py

# Emotion detection (downloads Emo-DB ~40MB)
python demo/train_emotion.py

# Encrypted template protection (novel: both voiceprints encrypted together)
python demo/train_encrypted_verify.py --n-pairs 5000 --noise-std 0.3
```

### Run FHE Demo

```bash
# Single task
python demo/fhe_demo.py --num-samples 10

# Multi-task FHE demo
python demo/fhe_multi_demo.py --task all --num-samples 5

# Specific task
python demo/fhe_multi_demo.py --task gender --num-samples 10

# Encrypted template protection FHE demo
python demo/encrypted_verify_demo.py --num-samples 10

# Activation function ablation study (GELU vs SiLU under FHE)
python demo/ablation_study.py --fhe-samples 5
```

### Web UI

```bash
python demo/web_ui.py --port 8080
# Open http://localhost:8080
```

### Client-Server Mode

```bash
# Terminal 1: Start the server
python demo/server.py

# Terminal 2: Run the client
python demo/client.py
```

## Testing

```bash
# Fast unit tests only (~15s)
pytest tests/ -m "not slow and not e2e"

# Include FHE integration tests (~3 min)
pytest tests/ -m "not e2e"

# Full end-to-end (~5 min)
pytest tests/
```

## Project Structure

```
voxcrypt/
├── configs/fhe_config.yml             # CKKS parameters
├── speaker_verify/
│   ├── features.py                    # MFCC extraction, embeddings, pair features
│   ├── dataset.py                     # LibriSpeech download, pair generation
│   ├── model.py                       # All models (orion.nn.Module + GELU)
│   ├── train.py                       # Training loop with FHE-aware noise
│   ├── emotion.py                     # Emo-DB dataset handling
│   ├── augment.py                     # Audio augmentation utilities
│   └── fhe_inference.py               # FHE pipeline
├── demo/
│   ├── train_model.py                 # Train speaker verification
│   ├── train_encrypted_verify.py      # Train encrypted template protection
│   ├── train_speaker_id.py            # Train speaker identification
│   ├── train_gender.py                # Train gender classification
│   ├── train_emotion.py               # Train emotion detection
│   ├── fhe_demo.py                    # Single-task FHE demo
│   ├── encrypted_verify_demo.py       # Encrypted template protection FHE demo
│   ├── fhe_multi_demo.py              # Multi-task FHE demo
│   ├── ablation_study.py              # Activation function ablation study
│   ├── web_ui.py                      # Web UI server
│   ├── optimize.py                    # Hyperparameter optimization
│   ├── server.py                      # Flask FHE inference server
│   ├── client.py                      # Encryption client
│   └── benchmark.py                   # Performance benchmarks
├── tests/                             # Unit + integration + e2e tests
└── assets/                            # Test WAV files
```

## Technical Details

- **Features**: 20 MFCCs → mean+std → 40-dim embedding per utterance
- **Verification**: |emb_A - emb_B| → 40-dim pair features
- **Model**: MLP with GELU activation (polynomial approximation for FHE)
- **FHE**: CKKS scheme via Orion/Lattigo (LogN=14, 13 multiplicative levels)
- **FHE-aware training**: Gaussian noise injection during training simulates CKKS precision loss
- **Datasets**: LibriSpeech test-clean (40 speakers), Emo-DB (7 emotions)

### Key Insight: GELU vs SiLU for FHE

GELU provides dramatically better FHE numerical stability than SiLU:
- SiLU(degree=7): -3.6 bits precision, 60% FHE accuracy
- GELU: +4.0 bits precision, 100% FHE accuracy

This matches Orion's cancer demo pattern (30→128→64→2, GELU) which achieves 95% FHE accuracy.

## Novel Contributions

### 1. Encrypted Biometric Template Protection

An FHE neural network approach to speaker verification where **both voiceprints stay encrypted during comparison**. Unlike the standard pipeline (which computes |emb_A - emb_B| in cleartext before encryption), this approach encrypts [emb_A || emb_B] together and the neural network learns the optimal comparison function entirely under FHE.

This improves over Nautsch et al. (2018) which used Paillier partial HE (additive-only, interactive protocol) with fixed cosine/Euclidean distance metrics.

| Approach | Encryption | Comparison | Protocol |
|----------|-----------|------------|----------|
| Nautsch et al. (2018) | Paillier partial HE | Fixed distance metric | Interactive |
| **Ours** | CKKS FHE | **Learned by neural network** | **Non-interactive** |

### 2. Activation Function Ablation Study for CKKS Voice Models

A systematic comparison of polynomial activations under CKKS FHE for speaker verification:

| Activation | Val Acc | FHE Acc | Precision | Input Level | FHE Time |
|-----------|---------|---------|-----------|-------------|----------|
| **GELU** | 91.2% | 100% | **+6.5 bits** | 13 | 25.8s |
| SiLU(d=3) | 91.0% | 20% | -3.0 bits | 7 | 9.3s |
| SiLU(d=5) | 91.0% | 20% | -2.2 bits | 9 | 9.2s |
| SiLU(d=7) | 91.0% | 20% | -1.3 bits | 9 | 9.0s |

Key finding: All SiLU variants fail under FHE despite identical cleartext accuracy. GELU's polynomial approximation preserves **9.5 more bits of precision** than SiLU(d=3), the difference between a working system and random guessing.

## License

MIT
