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

### Progress Tracking (CRITICAL - MUST FOLLOW)

**ALWAYS record progress and analysis in `update_note.md`**:

1. **실험 결과 분석 시**: 결과 테이블과 분석 내용을 기록
2. **문제 발견 시**: 문제 원인과 근거를 기록
3. **해결책 구현 시**: 접근법과 변경 내용을 기록
4. **버전 변경 완료 시**: 변경 사항 요약과 이유 기록
5. **버그 수정 시**: 에러 내용과 수정 방법 기록

**기록 형식**:
```markdown
## V{버전} - {기능명}

### 실험 결과
| Task | Metric | Value |
|------|--------|-------|
| ... | ... | ... |

### 분석
- 문제점: ...
- 원인: ...

### 해결책
- 변경 사항: ...
- 수정된 파일: ...
```

**중요**:
- 매 작업 세션마다 update_note.md에 진행 상황 기록
- 실험 실패도 기록 (왜 실패했는지, 무엇을 배웠는지)
- 다음 세션에서 컨텍스트를 잃지 않도록 충분히 상세하게 기록

This ensures continuity across sessions and maintains a clear history of the project's evolution.
