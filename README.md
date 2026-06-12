# ZVTVC — Zero-Shot Voice Conversion with Bilingual Diffusion Generation

A four-module zero-shot voice conversion system trained from scratch on bilingual (English + Arabic) speech. Implements speaker-disentangled representation learning with conditional flow-matching mel generation.

**Author:** Nouh Jaber · [github.com/nouhjaber](https://github.com/nouhjaber)

> **Project status:** A research / portfolio implementation. All four modules are independently trained and validated. End-to-end audio quality is limited by a known integration mismatch between the Generator's mel output and the vocoder's expected mel format — discussed honestly in [Known Limitations](#known-limitations).

---

## Overview

ZVTVC factorizes speech into three independent representations — **content**, **prosody**, and **timbre** — then synthesizes a target voice via a conditional flow-matching diffusion Generator. The system supports cross-lingual conversion between English and Arabic with a single shared model.

```
                                  ┌──────────────┐
   source audio ─► Content Enc ──►│              │
                                  │              │
   source audio ─► Prosody Enc ──►│  Generator   │──► mel ──► vocoder ──► audio
                                  │ (flow match) │
   target audio ─► Timbre Enc  ──►│              │
                                  └──────────────┘
```

The disentangled factorization means the Generator can mix content from one speaker with prosody from the same speaker and timbre from a different speaker — i.e., zero-shot conversion to an unseen target voice from a single reference clip.

---

## Modules

### 1. Content Encoder

Extracts speaker-independent linguistic content from mel spectrograms via multi-scale dilated convolutions, hierarchical fusion, and an information bottleneck.

| Property | Value |
|---|---|
| Parameters | ~5.4M |
| Input | mel spectrogram (80 bins, 16 kHz, hop=320) |
| Output | [B, 512, T] continuous features at 50 Hz |
| Multi-scale paths | fine (dilation 1), medium (1-2-4-1), coarse (1-4-16-4-1) |
| Information bottleneck | 256 → 128 → 256, α annealed 0.5 → 0.3 → 0.1 across stages |
| Teachers (distillation) | Whisper-large-v3 + mHuBERT-147 (768-dim, 94M params) |
| Training | 3-stage curriculum, ~25h total on A100 |
| Final checkpoint | Stage 2, iter 161,283 |

The three-stage curriculum:

1. **Stage 0 — Bootstrap distillation** (~53k iter). Aligns the encoder to phonetic content via knowledge distillation from frozen Whisper and mHuBERT teachers. Auxiliary phoneme classification head.
2. **Stage 1 — Speaker-adversarial refinement** (~79k iter). Gradient reversal head suppresses residual speaker information. λ_GRL warmup over the stage.
3. **Stage 2 — Bottleneck tightening** (~108k iter). Information bottleneck α reduced from 0.5 → 0.1 and EMA teacher decay raised from 0.99 → 0.999 to stabilize the tightened representation.

Final speaker-adversarial accuracy ~0.12% (chance-level for 1,350 speakers = 0.07%), confirming effective speaker information suppression through the bottleneck.

### 2. Prosody Encoder

Captures speaker-independent suprasegmental features (pitch, energy, rhythm) from explicit feature streams rather than mel.

| Property | Value |
|---|---|
| Parameters | 11,204 (intentionally tiny) |
| Input features | [F0, energy, voicing, rhythm] — 4 channels |
| F0 extractor | torchcrepe (with librosa.pyin fallback) |
| Output | [B, 32, T] refined prosody features |
| Auxiliary heads | F0, energy, voicing, rhythm reconstruction (training only) |
| Training | ~3h on A100 |
| Final checkpoint | iter 12,000, val loss 0.0067 |

Validation on 14,262 held-out samples — all six metrics pass:

| Metric | Value | Target |
|---|---|---|
| F0 Pearson correlation | **0.998** | > 0.80 |
| F0 RMSE (normalized) | 0.002 | < 0.10 |
| Energy correlation | **0.9999** | > 0.80 |
| Energy RMSE (normalized) | 0.001 | < 0.10 |
| Voicing error rate | **0.000** | < 0.10 |
| Rhythm correlation | **0.999** | > 0.75 |

The encoder operates on explicit prosody features rather than mel, ensuring speaker-independence by construction. The intentionally tiny model is sufficient because the input features are already highly informative.

### 3. Timbre Encoder

Speaker identity embedding via ECAPA-TDNN trained with InfoNCE contrastive learning.

| Property | Value |
|---|---|
| Parameters | ~1M (ECAPA-TDNN) |
| Embedding dimension | 256 (L2-normalized; std ≈ 1/√256 ≈ 0.063) |
| Loss | InfoNCE contrastive |
| Batch composition | 32 speakers × 2 utterances |
| Augmentation | GPU-side SpecAugment (freq/time masking) |
| Training | ~5h on A100 |
| Best checkpoint | iter 82,000 (early-stopped on val plateau) |
| Validation EER | 22.47% |

Trained from scratch on the bilingual dataset across all 1,351 train speakers. A custom speaker-aware batch sampler groups speakers by their dominant shard to minimize disk-cache thrashing during contrastive training (a critical optimization — the naive sampler was hitting 28+ shards per batch).

Final training statistics:
- Same-speaker cosine similarity: 0.76 – 0.84
- Different-speaker cosine similarity: 0.17 – 0.42 (improved over training)
- Validation threshold (EER): 0.691

The 22.47% EER is below production-grade speaker verification. See [Known Limitations](#known-limitations) for analysis.

### 4. Generator

Conditional flow-matching diffusion U-Net that predicts the velocity field for mel synthesis given (content, prosody, timbre) conditioning.

| Property | Value |
|---|---|
| Parameters | 37.35M |
| Architecture | U-Net with AdaLN modulation |
| Base model dimension | 256 |
| Channel multipliers | [1, 2, 2] → [256, 512, 512] |
| Residual blocks per level | 2 |
| Attention | Self-attention at resolution 4 (8 heads) |
| Conditioning dims | content=512, prosody=32, timbre=256, time=256 |
| Training objective | Conditional flow matching (CFM) |
| CFG dropout probability | 0.1 (all conditions dropped together) |
| Samplers | Euler / Midpoint / Heun (Euler default) |
| Inference steps | 10 (configurable up to 20+) |
| Training | ~16h on A100, 200k iterations |
| Final checkpoint | iter 195,000, val loss **0.1607** |

The Generator learns a velocity field over the prior-to-target mel trajectory. At inference, 10 Euler steps map Gaussian noise (conditioned on encoder outputs) to a target mel.

Standalone Generator evaluation (32 held-out samples):

| Steps | MSE | MAE | Latency (A100, 400 frames) |
|---|---|---|---|
| 10 | 0.126 | 0.166 | 164 ms (48.7× real-time) |
| 20 | **0.119** | 0.161 | 308 ms (26× real-time) |
| 50 | 0.122 | 0.163 | 758 ms (10.5× real-time) |

Training health throughout the 200k-iteration run:
- Loss trajectory (val): 1.15 → 0.69 → 0.46 → 0.34 → 0.28 → 0.16
- Velocity prediction std: stable ~1.1 (no collapse)
- Gradient norm: stable 0.2 – 0.8 (well within clip threshold)
- Speed: ~9.7 it/s on A100, no NaN events

---

## Data

| Split | English | Arabic | Total |
|---|---|---|---|
| Source | LibriTTS | Mozilla CommonVoice | — |
| Train | ~149k | ~107k | 257,198 |
| Val | — | — | 14,261 |
| Test | — | — | 14,261 |
| **Total samples** | — | — | **285,720** |
| **Speakers** | ~1,050 | ~300 | **1,351 train** |
| **Audio** | — | — | **372.9 hours** |

Audio is resampled to 16 kHz mono. The dataset is sharded for streaming training (~74 GB compressed) using a custom shard format with on-disk LRU caching. F0 / energy / voicing / rhythm features are precomputed during dataset construction.

---

## Mel Spectrogram Configuration

All four modules consume mel-spectrograms with identical specifications:

```
sample_rate    = 16000   Hz
n_fft          = 1024
hop_length     = 320     →  50 frames per second
win_length     = 1024
n_mels         = 80
f_min          = 0       Hz
f_max          = 8000    Hz
```

---

## Infrastructure

- **Compute:** Google Colab A100 40GB (primary), T4 16GB (earlier validation)
- **Storage:** Google Drive (checkpoints, packaged shards) + Colab local SSD (active shards)
- **Frameworks:** PyTorch 2.x, torchaudio, torchcrepe, librosa, transformers
- **Vocoder (intended):** HiFi-GAN UNIVERSAL_V1

A shard-based dataset pipeline handles the 285k-sample corpus with on-disk LRU caching, on-the-fly mel computation, and worker-parallel I/O. Custom packaging utilities upload and retrieve 70+ GB of shards between Drive and Colab local storage in resumable chunks, surviving session disconnects.

**Approximate training wall-clock times** (single A100):

| Module | Time |
|---|---|
| Content Encoder (3 stages) | ~25 h |
| Prosody Encoder | ~3 h |
| Timbre Encoder | ~5 h |
| Generator (200k iter) | ~16 h |

---

## Engineering Challenges

This section documents notable bugs found and fixed during development. They are kept here as a record of the iterative engineering work behind the trained checkpoints — both because the debugging itself was substantial, and because researchers reading this may face similar issues.

### Content Encoder (12 distinct fixes)

The Content Encoder required the most extensive debugging due to its multi-loss, multi-stage curriculum:

- **Learnable `log_sigma` driving loss negative.** A learned uncertainty weight could collapse to negative values, flipping the loss sign.
- **No label smoothing.** Phoneme classification overfit on confident-but-wrong labels.
- **Confidence thresholding missing.** Distillation weights used raw teacher confidence; spec required `((conf - 0.3) / 0.4).clamp(0, 1)`.
- **Adversarial loss values 10–20× too large.** Required rescaling.
- **`λ_GRL` never ramped up.** Stage 1 was effectively disabled.
- **Dual EMA teacher mismatch.** Trainer updated one EMA module; distillation loss read from a different one. Updates didn't propagate.
- **LossScheduler stage never updated.** Loss weights remained Stage-0 values throughout all stages.
- **Whisper projection weights never optimized.** Missing from the optimizer parameter group.
- **Bottleneck α never annealed.** `set_alpha_bn()` existed but was never called.
- **EMA α never annealed.** Same fix pattern as bottleneck α.
- **ETA inflated 100×.** `DetailedTrainingLogger` measured time between log calls and treated it as per-iteration time.
- **EMA state not restored from checkpoint.** Caused fresh EMA on every resume.

### Timbre Encoder (5 significant fixes)

The Timbre Encoder initially hit 31.59% EER. Each fix below dropped EER meaningfully:

- **SpeakerAwareBatchSampler cache thrashing.** Naive sampler hit 28+ shards per batch; rewrote with shard-locality awareness.
- **`_build_speaker_mapping` 40-minute hang.** Iterated 256k samples individually; replaced with single-pass per-shard.
- **Mel computed on CPU in `__getitem__`.** Moved to GPU in `_train_step`.
- **DataLoader workers forking entire shard cache.** Caused OOM; reduced workers, increased per-worker cache limit.
- **LayerNorm wrong dimension after Conv1d.** Replaced with `TransposedLayerNorm` / `BatchNorm1d`.

Plus fixes for validation double-counting, NaN accuracy from `0 * -inf`, and logger debug-spam in forward pass.

Final EER: **22.47%**.

### Prosody Encoder (7 fixes)

- **`torch.tensor(0.0, requires_grad=True)` in-place op crash** in `losses.py`.
- **Model args parsed but never passed** to the constructor.
- **Missing function parameters** in non-shard data mode.
- **Incorrect loss weight assignments.**
- **Metric key mismatch** (`f0_rmse_cents` vs. `f0_rmse`) caused 2/6 metrics to spuriously fail despite perfect underlying values.
- **PyTorch 2.6 `weights_only=True` default** broke checkpoint loading; fixed with `weights_only=False`.
- **`sys.path.append` placed after imports** caused `ModuleNotFoundError`.

### Generator (4+ fixes)

- **Warmup LR computed but never applied** in trainer (`_scheduler_steps` tracking missing).
- **Numpy object array dtype rejection** by `torch.from_numpy` — required explicit `np.asarray(..., dtype=np.float32)`.
- **STFT padding crash** when input T < `fft_size` — required padding to at least `fft_size`.
- **Odd-T skip connection crash** in U-Net (off-by-one between encoder and decoder skip dimensions).

Plus 17-test data-flow verification suite, EMA save/load round-trip, CFG double-call path, multi-resolution STFT auxiliary loss verification.

---

## Known Limitations

This section is an honest assessment of the system's current state.

### End-to-end audio quality is degraded

While each module functions correctly in isolation, end-to-end conversion does not produce clean audio. Two root causes:

1. **Mel-spectrogram mismatch with vocoder.** The Generator outputs mels at 16 kHz / hop=320 (50 fps). HiFi-GAN UNIVERSAL_V1 expects mels at 22050 Hz / hop=256 (~86 fps). Temporal interpolation can bridge the gap but smears spectral structure that vocoders depend on.
2. **Per-utterance mel normalization during Generator training.** The Generator was trained to predict per-utterance normalized mels (mean=0, std=1). Standard vocoders expect raw log-mel in a fixed range (~[-11, +2]). True per-utterance statistics are unrecoverable at inference time.

The proper fix is to retrain the Generator at the vocoder's mel specifications without per-utterance normalization, or to train a custom 16 kHz vocoder. This is planned future work but was outside the available compute budget.

### Timbre EER of 22% is below production-grade

Production speaker verification systems target EER under 3%. The 22% here reflects:

- Small InfoNCE batch (only 31 in-batch negatives per anchor)
- No hard negative mining
- 1M-parameter model vs. 6–15M in production systems
- No augmentation diversity beyond GPU SpecAugment

For voice conversion conditioning, this means the synthesized voice carries speaker characteristics in the right direction but does not exactly match the target.

### What this means in practice

Each module independently demonstrates trained ML competence — the encoders successfully learn their respective representations (verified by per-module metrics above), and the Generator successfully predicts velocity fields conditioned on those representations (verified by standalone evaluation). The system breaks down at the *integration boundary* between the Generator's mel output and the vocoder's mel input, which is a known difficult problem in voice conversion literature.

---

## References

The architecture and training methodology draw on the following papers:

- **Zero-Shot Voice Conversion with Diffusion Transformers.** Primary architectural reference for the diffusion-based Generator and overall factorized approach.
- **NaturalSpeech 3:** *Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models.* Reference for the disentangled content / prosody / timbre factorization and flow-matching training.
- **CosyVoice 2:** *Scalable Streaming Speech Synthesis with Large Language Models.* Reference for content encoder design considerations.
- **EAD-VC:** *Enhancing Speech Auto-Disentanglement for Voice Conversion with IFUB Estimator and Joint Text-Guided Consistent Learning.* Reference for the information bottleneck and adversarial speaker removal in the Content Encoder.

---

## Repository Structure

```
ZVTVC/
├── Content_Encoder/
│   ├── model/                       # Multi-scale encoder, information bottleneck, output projection
│   ├── training/                    # 3-stage trainer, loss scheduler, distillation, GRL adversarial
│   ├── train.py
│   └── checkpoints/                 # stage_2_final.pt (iter 161,283)
│
├── Prosody_Encoder/
│   ├── model/                       # Explicit feature extractor + refinement
│   ├── training/                    # F0/energy/voicing/rhythm losses, reconstruction heads
│   ├── train.py
│   ├── Validate.py
│   └── checkpoints/                 # best.pt (iter 12,000)
│
├── Timbre_Encoder/
│   ├── model/                       # ECAPA-TDNN
│   ├── training/                    # InfoNCE loss, SpeakerAwareBatchSampler, EER validation
│   ├── train.py
│   ├── Validate.py                  # EER, ROC, threshold sweep
│   └── checkpoints/                 # final_model.pt (iter 82,000)
│
├── Generator/
│   ├── model/                       # Flow-matching U-Net, AdaLN blocks, attention
│   ├── inference/                   # Euler/Midpoint/Heun samplers, MelGenerator wrapper, CFG
│   ├── training/                    # CFM loss, multi-resolution STFT aux loss, EMA
│   ├── scripts/
│   │   ├── convert.py               # End-to-end voice conversion
│   │   ├── evaluate.py              # Standalone Generator evaluation (MSE/MAE/latency)
│   │   └── verify_fixes.py          # Encoder loading + mel preprocessing sanity checks
│   ├── precompute_encoder_outputs.py
│   ├── train.py
│   └── checkpoints/                 # best_model.pt (iter 195,000, val 0.1607)
│
├── Shard_dataset_unified.py         # Shared shard-based dataset loaders (4 dataset classes)
├── preprocess_unified_shards.py     # Raw audio → unified shards (audio + F0/energy/voicing/rhythm)
├── create_timbre_shards_direct.py   # Direct raw-audio → timbre-specific shards
├── package_for_drive.py             # Package shards into resumable tar archives for Drive
└── fetch_from_drive.py              # Pull shards from Drive, verify, extract
```

---
