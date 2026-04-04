# Orion Voice — Encrypted Speaker Verification

End-to-end encrypted speaker verification using Fully Homomorphic Encryption (FHE). A client extracts audio features locally, encrypts them, and sends to a server that verifies speaker identity **on encrypted data** — without ever hearing the audio or seeing the result.

## How It Works

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

## Setup

```bash
# Install dependencies
pip install -e .

# Install with server support
pip install -e ".[server]"

# Install with dev dependencies
pip install -e ".[dev]"
```

Requires [Orion](https://github.com/kaushik0x7d2/orion) installed locally at `C:/Users/Unitech/Desktop/Kaushik/orion/repo`.

## Quick Start

### 1. Train the Model

Downloads LibriSpeech test-clean (~350MB), extracts MFCC features, and trains the speaker verification model.

```bash
python demo/train_model.py
```

### 2. Run FHE Demo

Runs cleartext and FHE inference side-by-side on test samples.

```bash
python demo/fhe_demo.py
```

### 3. Client-Server Mode

```bash
# Terminal 1: Start the server
python demo/server.py

# Terminal 2: Run the client
python demo/client.py
```

## Testing

```bash
# Fast unit tests only
pytest tests/ -m "not slow and not e2e"

# Include FHE integration tests (~3 min)
pytest tests/ -m "not e2e"

# Full end-to-end (~5 min)
pytest tests/
```

## Benchmarks

```bash
python demo/benchmark.py --num-samples 20
```

## Project Structure

```
orion-voice/
├── configs/fhe_config.yml          # CKKS parameters
├── speaker_verify/
│   ├── features.py                 # MFCC extraction, embeddings, pair features
│   ├── dataset.py                  # LibriSpeech download, pair generation
│   ├── model.py                    # SpeakerVerifyNet (orion.nn.Module)
│   ├── train.py                    # Training loop
│   └── fhe_inference.py            # FHE pipeline
├── demo/
│   ├── train_model.py              # CLI: train the model
│   ├── fhe_demo.py                 # CLI: FHE demo
│   ├── server.py                   # Flask FHE inference server
│   ├── client.py                   # Encryption client
│   └── benchmark.py                # Performance benchmarks
├── tests/                          # Unit + integration + e2e tests
└── assets/                         # Test WAV files
```

## Technical Details

- **Features**: 20 MFCCs → mean+std → 40-dim embedding per utterance → |emb_A - emb_B| for pair comparison
- **Model**: 40→128→64→2 MLP with SiLU (degree=7 polynomial for FHE)
- **FHE**: CKKS scheme via Orion/Lattigo (LogN=14, ~11 multiplicative levels)
- **Dataset**: LibriSpeech test-clean (~20 speakers, ~10 utterances each)
- **Expected accuracy**: 75-85% cleartext (MFCC-based, intentionally simple for FHE)
- **FHE time**: ~15-20s per verification pair

## License

MIT
