# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MoLE-Flow (Mixture of LoRA Experts for Normalizing Flow) is a continual anomaly detection framework. It uses Normalizing Flows with LoRA (Low-Rank Adaptation) to learn anomaly detection across multiple tasks (product classes) sequentially without catastrophic forgetting.

**Key Problem**: Learn anomaly detection for Task 0 (e.g., leather), then Task 1 (e.g., grid), then Task 2 (e.g., transistor), while maintaining performance on all previous tasks. At inference time, the task ID is unknown - a router predicts which LoRA expert to use.

## Commands

### Training
```bash
# Basic training with defaults (recommended)
python run_moleflow.py

# Custom configuration
python run_moleflow.py \
    --task_classes leather grid transistor \
    --num_epochs 40 \
    --num_coupling_layers 16 \
    --lora_rank 64 \
    --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --img_size 224 \
    --experiment_name my_experiment

# Run with diagnostics (generates flow analysis plots)
python run_moleflow.py --run_diagnostics

# Ablation studies
python run_moleflow.py --no_lora           # Disable LoRA
python run_moleflow.py --no_router         # Use oracle task ID
python run_moleflow.py --ablation_preset wo_adapter  # Predefined ablation
```

### Quick run script
```bash
./run.sh
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

### Pipeline Flow
```
Image → ViT Extractor (frozen) → Positional Embedding → TaskInputAdapter → Normalizing Flow (with LoRA) → Latent z
                                                                                   ↓
                                                              Anomaly Score = -log p(z) - log|det J|
```

### Core Components (`moleflow/`)

1. **Feature Extraction** (`extractors/`)
   - `ViTPatchCoreExtractor`: Extracts patch-level features from DINOv2/ViT backbone (frozen)
   - Aggregates features from multiple transformer blocks

2. **Normalizing Flow** (`models/mole_nf.py`)
   - `MoLESpatialAwareNF`: Main NF model with LoRA adapters
   - Uses FrEIA library for coupling layers
   - Task-specific input adapters for distribution alignment

3. **LoRA Adaptation** (`models/lora.py`)
   - `LoRALinear`: Linear layer with low-rank adapters per task
   - `MoLESubnet`: Subnet for coupling blocks with LoRA
   - Base weights frozen after Task 0; only LoRA trained for subsequent tasks

4. **Router** (`models/routing.py`)
   - `PrototypeRouter`: Routes images to correct LoRA expert using Mahalanobis distance
   - `TaskPrototype`: Stores mean/covariance of each task's features

5. **Trainer** (`trainer/continual_trainer.py`)
   - `MoLEContinualTrainer`: Handles continual learning loop
   - FAST stage: Train LoRA + TaskAdapter (base frozen)
   - Optional SLOW stage: Fine-tune last K coupling blocks

6. **Configuration** (`config/ablation.py`)
   - `AblationConfig`: Controls which components are active
   - Presets: `full`, `wo_lora`, `wo_router`, `wo_adapter`, etc.

### Key Design Decisions

- **Task 0**: Trains base NF weights + LoRA adapter together
- **Task 1+**: Freezes base weights, only trains new LoRA adapter + TaskInputAdapter
- **Scaling**: `LoRA_output = (alpha/rank) * B @ A @ x`
- **TaskInputAdapter**: FiLM-style modulation with LayerNorm (aligns feature distributions)

## Data

Uses MVTec AD dataset. Set path via `--data_path /path/to/MVTecAD`.

Available classes: `leather grid transistor carpet zipper hazelnut toothbrush metal_nut screw wood tile capsule pill cable bottle`

## Logs

Training logs and results are saved to `--log_dir` (default: `./logs/`). Each experiment creates a timestamped folder containing:
- `config.json`: Experiment configuration
- `training.log`: Training progress
- `final_results.csv`: Per-class metrics
- `diagnostics/`: Flow analysis plots (if `--run_diagnostics`)

## Development Workflow

### Progress Tracking (IMPORTANT)

**Always record progress and analysis in `update_note.md`**:
- When analyzing experimental results, document findings in update_note.md
- When identifying problems or root causes, record them with evidence
- When proposing or implementing solutions, document the approach and rationale
- When completing version changes, summarize what was changed and why
- Include quantitative results (metrics, performance numbers) when available

This ensures continuity across sessions and maintains a clear history of the project's evolution.
