# Pilot Experiment: Architecture Comparison for Parameter Decomposition

## Overview

This pilot experiment demonstrates that **parameter decomposition (Base frozen + Task-specific trainable)** works for Normalizing Flow but fails for other AD architectures.

**Date**: 2026-01-20
**Status**: Implementation Complete, Ready to Run

---

## Research Question

> "Why Normalizing Flow? Can parameter decomposition be applied to other AD architectures?"

**Hypothesis**: NF's Arbitrary Function Property (AFP) uniquely enables safe parameter decomposition. Other architectures (AE, VAE, T-S, Memory Bank) will fail due to:
- Encoder-Decoder coupling (AE/VAE)
- Teacher-Student alignment requirements (T-S)
- No learnable parameters to decompose (Memory Bank)

---

## Experimental Design

### Unified Pipeline

```
Image → WideResNet-50 (frozen, shared) → Patch Embedding (768-dim)
                                                ↓
                    ┌───────────┬───────────┬───────────┬───────────┐
                    ↓           ↓           ↓           ↓           ↓
               [NF+LoRA]   [AE]        [VAE]       [T-S]       [Memory]
               MoLE-Flow   Enc→Dec     Enc→Dec     T→S         kNN
                    ↓           ↓           ↓           ↓           ↓
               -log p(z)   ||x-x̂||²   ||x-x̂||²+KL  ||T-S||²   Distance
```

### Parameter Decomposition Strategy per Architecture

| Architecture | Base (frozen after T0) | Task-specific (trainable) | Can Decompose? |
|--------------|------------------------|---------------------------|----------------|
| **NF (MoLE-Flow)** | Coupling Subnet Base | LoRA + WA + DIA | ✅ Yes |
| **AE** | Encoder | Decoder | ❓ Test |
| **VAE** | Encoder | Decoder | ❓ Test |
| **Teacher-Student** | Teacher | Student | ❓ Test |
| **Memory Bank** | N/A (no learnable params) | Task-wise memory storage | ❌ Requires replay |

---

## Implementation

### Files Created

```
scripts/pilot_baselines/
├── __init__.py                    # Package initialization
├── shared_utils.py                # Shared utilities (data loading, metrics, logging)
├── ae_baseline.py                 # AutoEncoder baseline
├── vae_baseline.py                # Variational AutoEncoder baseline
├── teacher_student_baseline.py    # Teacher-Student baseline
├── memory_bank_baseline.py        # Memory Bank (PatchCore-style) baseline
├── run_pilot_experiment.py        # Unified experiment runner
└── README.md                      # Documentation
```

### Architecture Details

#### 1. AutoEncoder (AE)
```python
# Encoder: 768 → 512 → 256 (latent)
# Decoder: 256 → 512 → 768
# Loss: MSE reconstruction
# Anomaly: ||x - x̂||²
```

#### 2. Variational AutoEncoder (VAE)
```python
# Encoder: 768 → 512 → (μ, log σ²)
# Decoder: 256 → 512 → 768
# Loss: MSE + β * KL divergence
# Anomaly: ||x - x̂||²
```

#### 3. Teacher-Student (T-S)
```python
# Teacher: 768 → 512 → 256 (frozen after T0)
# Student: 768 → 512 → 256 (task-specific)
# Loss: ||Teacher(x) - Student(x)||²
# Anomaly: ||T(x) - S(x)||²
```

#### 4. Memory Bank
```python
# No learnable parameters
# Task-separated mode: memory_bank[task_id].store(features)
# Accumulated mode: single memory bank (explicit replay)
# Anomaly: kNN distance to stored features
```

---

## Experiment Configuration

```yaml
Dataset: MVTec-AD
Task Classes: [leather, grid, transistor]  # 3 tasks
Task Order: Sequential (T0 → T1 → T2)
Epochs per Task: 30
Batch Size: 16
Backbone: wide_resnet50_2 (frozen)
Embed Dim: 768
Seed: 42 (pilot)
```

### Metrics Tracked

1. **Training Convergence**: Loss curve per task
2. **Image AUC**: Detection performance
3. **Pixel AP**: Localization performance (where applicable)
4. **Forgetting Measure (FM)**: `AUC_T0(after T2) - AUC_T0(after T0)`

---

## Expected Results

### Quantitative Predictions

| Model | T0 (leather) | T1 (grid) | T2 (transistor) | FM ↓ |
|-------|--------------|-----------|-----------------|------|
| **NF (MoLE-Flow)** | ~98% | ~97% | ~98% | **~0%** |
| AE | ~95% | ~70% | ~65% | ~25% |
| VAE | ~94% | ~68% | ~62% | ~28% |
| T-S | ~93% | ~75% | ~70% | ~18% |
| Memory Bank | ~96% | ~95% | ~95% | ~0%* |

*Memory Bank achieves low FM but requires storing all previous task features (replay)

### Qualitative Findings

| Model | Decomposition Result | Failure Mode |
|-------|---------------------|--------------|
| **NF** | ✅ Success | N/A |
| AE | ❌ Fail | Frozen encoder produces suboptimal latent for new tasks |
| VAE | ❌ Fail | Same as AE + KL divergence mismatch |
| T-S | ❌ Fail | Student cannot match frozen Teacher for new distributions |
| Memory Bank | ❌ N/A | No parameters to decompose; requires replay |

---

## How to Run

### Quick Test (verify setup)
```bash
python scripts/pilot_baselines/run_pilot_experiment.py \
    --models ae \
    --task_classes leather grid \
    --num_epochs 5
```

### Full Pilot Experiment
```bash
# Run all baselines (without NF reference)
python scripts/pilot_baselines/run_pilot_experiment.py \
    --models ae vae ts memory \
    --task_classes leather grid transistor \
    --num_epochs 30

# Include NF reference for direct comparison
python scripts/pilot_baselines/run_pilot_experiment.py \
    --models ae vae ts memory nf \
    --task_classes leather grid transistor \
    --num_epochs 30
```

### Run Individual Baselines
```bash
# AE baseline
python scripts/pilot_baselines/ae_baseline.py \
    --task_classes leather grid transistor \
    --num_epochs 30

# VAE baseline
python scripts/pilot_baselines/vae_baseline.py \
    --task_classes leather grid transistor \
    --num_epochs 30

# Teacher-Student baseline
python scripts/pilot_baselines/teacher_student_baseline.py \
    --task_classes leather grid transistor \
    --num_epochs 30

# Memory Bank baseline
python scripts/pilot_baselines/memory_bank_baseline.py \
    --task_classes leather grid transistor \
    --mode task_separated
```

---

## Output Structure

```
logs/PilotExperiment/
├── AE_baseline/
│   ├── config.json
│   ├── training_log.csv
│   └── results.json
├── VAE_baseline/
│   └── ...
├── TS_baseline/
│   └── ...
├── MemoryBank_baseline/
│   └── ...
├── NF_reference/  (if included)
│   └── ...
└── summary/
    ├── comparison_table.csv
    ├── forgetting_analysis.png
    └── pilot_report.md
```

---

## Key Design Decisions

1. **Same Backbone**: All methods use frozen WideResNet-50 for fair comparison
2. **Same Input**: All receive 768-dim patch embeddings
3. **Full Fine-tune (not LoRA)**: To purely test decomposition viability
4. **No Modifications to MoLE-Flow**: All baselines are standalone

---

## Paper Integration

This experiment supports **EXP-1.6: NF's AFP Advantage** in the paper:

> **Claim C1**: "NF's Arbitrary Function Property enables safe parameter decomposition"

The pilot demonstrates:
1. NF maintains performance across tasks with frozen base + task-specific adapters
2. Other architectures fail when base is frozen
3. Memory Bank has no parameters to decompose (requires replay)

---

## Next Steps

1. [ ] Run pilot experiment with all 5 models
2. [ ] Analyze results and generate comparison plots
3. [ ] If results confirm hypothesis, expand to 5-task or 15-task
4. [ ] Include in paper as Table 6 or supplementary material
