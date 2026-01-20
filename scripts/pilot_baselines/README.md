# Pilot Baselines for MoLE-Flow Paper

This directory contains baseline implementations for the pilot experiment demonstrating that **parameter decomposition (Base frozen + Task-specific trainable) works for Normalizing Flows but NOT for reconstruction-based methods**.

## Hypothesis

### Why it works for Normalizing Flows (MoLE-Flow)
- **NF Base**: Captures the shared invertible transformation structure
- **LoRA Adapters**: Fine-tune the density estimation for task-specific distributions
- **Key Insight**: The frozen base still provides a meaningful bijective mapping; LoRA only needs to adjust the "shape" of the density

### Why it FAILS for AutoEncoders
- **Encoder**: Maps input distribution to latent space (distribution-specific)
- **Decoder**: Reconstructs from latent space (depends on encoder's mapping)
- **Problem**: Freezing encoder after Task 0 means:
  1. Task 1+ features are encoded using Task 0's learned mapping
  2. This mapping is suboptimal for the new task's distribution
  3. Decoder cannot compensate for poor latent representations
  4. Result: High reconstruction error = Poor anomaly detection

## AutoEncoder Baseline

### Architecture
```
Input (768-dim patch embedding)
    |
    v
Encoder (768 -> 512 -> 256)  <- FROZEN after Task 0
    |
    v
Latent z (256-dim)
    |
    v
Decoder (256 -> 512 -> 768)  <- TRAINABLE for all tasks
    |
    v
Reconstruction x_hat (768-dim)

Anomaly Score = ||x - x_hat||^2
```

### Continual Learning Setup
```
Task 0: leather
  - Train: Encoder + Decoder (full AE)
  - Result: Good performance

[FREEZE ENCODER]

Task 1: grid
  - Train: Decoder only
  - Result: Poor (encoder cannot adapt to grid's texture patterns)

Task 2: transistor
  - Train: Decoder only
  - Result: Poor (even worse, encoder stuck on leather's distribution)
```

## Running the Experiment

### Basic Run
```bash
cd /Volume/MoLeFlow
python scripts/pilot_baselines/ae_baseline.py \
    --task_classes leather grid transistor \
    --num_epochs 40 \
    --latent_dim 256 \
    --experiment_name "pilot_ae_test"
```

### Full MVTec Run (5 tasks)
```bash
python scripts/pilot_baselines/ae_baseline.py \
    --task_classes leather grid transistor carpet zipper \
    --num_epochs 60 \
    --latent_dim 256 \
    --experiment_name "pilot_ae_5tasks"
```

### Compare with MoLE-Flow
```bash
# Run AE baseline
python scripts/pilot_baselines/ae_baseline.py \
    --task_classes leather grid transistor \
    --experiment_name "pilot_ae"

# Run MoLE-Flow with same tasks
python run_moleflow.py \
    --task_classes leather grid transistor \
    --experiment_name "pilot_moleflow"
```

## Expected Results

| Task | AE Baseline (I-AUC) | MoLE-Flow (I-AUC) |
|------|---------------------|-------------------|
| leather (T0) | ~95% | ~98% |
| grid (T1) | ~60-70% | ~96% |
| transistor (T2) | ~55-65% | ~97% |

### Forgetting Analysis
- **AE**: After Task 2, Task 0 (leather) may degrade by 5-15%
- **MoLE-Flow**: Near-zero forgetting (<1% degradation)

## Output Files

Results are saved to `logs/PilotExperiment/AE_baseline/<experiment_name>/`:
- `config.json`: Experiment configuration
- `results_after_task_0.json`: Performance after Task 0
- `results_after_task_1.json`: Performance after Task 1 (includes forgetting)
- `final_results.json`: Complete results with all metrics
- `training_curves.json`: Loss curves for each task

## Key Metrics

1. **Image AUC**: Image-level anomaly detection performance
2. **Pixel AUC**: Pixel-level localization performance
3. **Forgetting Measure**: `initial_performance - current_performance` for previous tasks

## Citation

This experiment supports the claims in the MoLE-Flow paper regarding the suitability of Normalizing Flows for continual anomaly detection.
