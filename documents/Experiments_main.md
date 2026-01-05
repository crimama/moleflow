# MoLE-Flow Main Experiments

## Baseline Configuration

**Experiment**: `MVTec-WRN50-TailW0.7-TopK3-TailTopK2-ScaleK5-lr3e-4-MAIN`

| Parameter | Value |
|-----------|-------|
| Backbone | WideResNet50 |
| LoRA Rank | 64 |
| Coupling Layers | 8 |
| DIA Blocks | 4 |
| Epochs | 60 |
| Learning Rate | 3e-4 |
| Adapter Mode | Whitening |
| Tail Weight | 0.7 |
| Score Aggregation | top_k (k=3) |
| Scale Context Kernel | 5 |
| Lambda Logdet | 1e-4 |

---

## 1. Main Performance Table - MVTecAD

### Overall Results

| Method | Image AUC | Pixel AUC | Image AP | Pixel AP | Routing Acc |
|--------|-----------|-----------|----------|----------|-------------|
| **MoLE-Flow (Ours)** | **98.29** | **97.82** | **99.28** | **54.20** | **100.0** |

### Comparison with Baselines (Pixel AP)

| Method | bottle | cable | capsule | carpet | grid | hazelnut | leather | metal_nut | pill | screw | tile | toothbrush | transistor | wood | zipper | **Avg** | FM |
|--------|--------|-------|---------|--------|------|----------|---------|-----------|------|-------|------|------------|------------|------|--------|---------|-----|
| **MoLE-Flow (Ours)** | 0.721 | **0.616** | 0.388 | 0.632 | 0.266 | 0.579 | 0.453 | 0.758 | **0.814** | 0.228 | **0.668** | **0.540** | 0.646 | 0.467 | 0.356 | **0.542** | **0.00** |
| Joint_PatchCore | **0.820** | 0.514 | **0.525** | **0.770** | **0.300** | 0.728 | 0.224 | **0.892** | 0.811 | **0.336** | 0.620 | 0.527 | 0.637 | **0.683** | 0.531 | 0.594 | - |
| Joint_PatchCore(R) | 0.826 | 0.505 | 0.510 | 0.766 | 0.293 | 0.712 | 0.230 | 0.862 | 0.785 | 0.157 | 0.664 | 0.561 | 0.515 | 0.641 | **0.565** | 0.573 | - |
| Joint_CADIC | 0.815 | 0.510 | 0.519 | 0.754 | 0.292 | **0.744** | 0.210 | 0.886 | 0.815 | 0.307 | 0.630 | 0.530 | **0.650** | 0.675 | 0.528 | 0.591 | - |
| CADIC | 0.790 | 0.485 | 0.506 | 0.753 | 0.276 | 0.749 | 0.191 | 0.880 | 0.810 | 0.328 | 0.609 | 0.527 | 0.650 | 0.686 | 0.517 | 0.584 | 0.015 |
| ReplayCAD [24] | 0.710 | 0.369 | 0.337 | 0.652 | 0.338 | 0.635 | **0.587** | 0.656 | 0.698 | 0.329 | 0.531 | 0.576 | 0.605 | 0.500 | 0.539 | 0.537 | 0.055 |
| DFM [8] | 0.768 | 0.506 | 0.241 | 0.771 | 0.228 | 0.479 | 0.432 | 0.690 | 0.576 | 0.242 | 0.623 | 0.331 | 0.501 | 0.581 | 0.511 | 0.511 | 0.013 |
| CFRDC [27] | 0.737 | 0.518 | 0.425 | 0.506 | 0.243 | 0.556 | 0.372 | 0.666 | 0.417 | 0.125 | 0.454 | 0.417 | 0.710 | 0.380 | 0.390 | 0.461 | - |
| UCAD [7] | 0.752 | 0.290 | 0.349 | 0.622 | 0.187 | 0.506 | 0.333 | 0.775 | 0.634 | 0.214 | 0.549 | 0.298 | 0.398 | 0.535 | 0.398 | 0.456 | 0.013 |
| FT_PatchCore | 0.048 | 0.029 | 0.035 | 0.552 | 0.003 | 0.338 | 0.279 | 0.248 | 0.051 | 0.008 | 0.249 | 0.034 | 0.079 | 0.304 | 0.595 | 0.190 | 0.371 |
| FT_CFA | 0.068 | 0.056 | 0.050 | 0.271 | 0.004 | 0.341 | 0.393 | 0.255 | 0.080 | 0.015 | 0.155 | 0.053 | 0.056 | 0.281 | 0.573 | 0.177 | 0.083 |
| IUF [25] | 0.289 | 0.054 | 0.040 | 0.440 | 0.084 | 0.301 | 0.330 | 0.142 | 0.048 | 0.012 | 0.310 | 0.049 | 0.065 | 0.326 | 0.080 | 0.171 | 0.059 |
| FT_RD4AD | 0.055 | 0.040 | 0.064 | 0.212 | 0.005 | 0.384 | 0.116 | 0.247 | 0.061 | 0.015 | 0.193 | 0.034 | 0.059 | 0.097 | 0.562 | 0.143 | 0.425 |
| FT_SimpleNet | 0.108 | 0.045 | 0.029 | 0.018 | 0.004 | 0.029 | 0.006 | 0.227 | 0.077 | 0.004 | 0.082 | 0.046 | 0.049 | 0.037 | 0.139 | 0.060 | 0.069 |

*Note: Joint methods are upper bounds (trained on all data simultaneously). FT methods show catastrophic forgetting (fine-tuning baseline). FM = Forgetting Measure (lower is better).*

### Per-Category Results (MVTecAD)

| Category | Image AUC | Pixel AUC | Image AP | Pixel AP | Routing Acc |
|----------|-----------|-----------|----------|----------|-------------|
| Bottle | 100.00 | 98.27 | 100.00 | 72.06 | 100.0% |
| Cable | 99.03 | 97.61 | 99.51 | 61.63 | 100.0% |
| Capsule | 97.25 | 98.65 | 99.41 | 38.84 | 100.0% |
| Carpet | 99.48 | 98.96 | 99.85 | 63.16 | 100.0% |
| Grid | 99.25 | 97.69 | 99.74 | 26.56 | 100.0% |
| Hazelnut | 99.93 | 98.69 | 99.96 | 57.86 | 100.0% |
| Leather | 100.00 | 99.33 | 100.00 | 45.27 | 100.0% |
| Metal Nut | 100.00 | 97.15 | 100.00 | 75.75 | 100.0% |
| Pill | 99.10 | 98.99 | 99.85 | 81.39 | 100.0% |
| Screw | 92.17 | 98.24 | 96.90 | 22.81 | 100.0% |
| Tile | 100.00 | 96.71 | 100.00 | 66.80 | 100.0% |
| Toothbrush | 90.83 | 98.47 | 95.96 | 53.98 | 100.0% |
| Transistor | 99.17 | 96.94 | 98.60 | 64.61 | 100.0% |
| Wood | 98.86 | 94.14 | 99.64 | 46.65 | 100.0% |
| Zipper | 99.24 | 97.52 | 99.81 | 35.57 | 100.0% |
| **Average** | **98.29** | **97.82** | **99.28** | **54.20** | **100.0%** |

---

## 2. Main Performance Table - ViSA

### Overall Results

| Method | Image AUC | Pixel AUC | Image AP | Pixel AP | Parameters |
|--------|-----------|-----------|----------|----------|------------|
| **MoLE-Flow (Ours)** | TBD | TBD | TBD | TBD | TBD |
| FastFlow | TBD | TBD | TBD | TBD | TBD |
| CFLOW-AD | TBD | TBD | TBD | TBD | TBD |
| CS-Flow | TBD | TBD | TBD | TBD | TBD |
| DiffNet | TBD | TBD | TBD | TBD | TBD |
| PatchCore | TBD | TBD | TBD | TBD | TBD |

### Per-Category Results (ViSA)

| Category | Image AUC | Pixel AUC | Image AP | Pixel AP |
|----------|-----------|-----------|----------|----------|
| Candle | TBD | TBD | TBD | TBD |
| Capsules | TBD | TBD | TBD | TBD |
| Cashew | TBD | TBD | TBD | TBD |
| Chewinggum | TBD | TBD | TBD | TBD |
| Fryum | TBD | TBD | TBD | TBD |
| Macaroni1 | TBD | TBD | TBD | TBD |
| Macaroni2 | TBD | TBD | TBD | TBD |
| PCB1 | TBD | TBD | TBD | TBD |
| PCB2 | TBD | TBD | TBD | TBD |
| PCB3 | TBD | TBD | TBD | TBD |
| PCB4 | TBD | TBD | TBD | TBD |
| Pipe Fryum | TBD | TBD | TBD | TBD |
| **Average** | TBD | TBD | TBD | TBD |

---

## 3. Performance Graph - MVTecAD

### 3.1 Learning Curves

**실험 목적**: 학습 과정에서 각 task별 성능 변화 추이 분석

| Task ID | Category | Final Image AUC | Final Pixel AUC | Final Pixel AP |
|---------|----------|-----------------|-----------------|----------------|
| 0 | Bottle | 100.00 | 98.27 | 72.06 |
| 1 | Cable | 99.03 | 97.61 | 61.63 |
| 2 | Capsule | 97.25 | 98.65 | 38.84 |
| 3 | Carpet | 99.48 | 98.96 | 63.16 |
| 4 | Grid | 99.25 | 97.69 | 26.56 |
| 5 | Hazelnut | 99.93 | 98.69 | 57.86 |
| 6 | Leather | 100.00 | 99.33 | 45.27 |
| 7 | Metal Nut | 100.00 | 97.15 | 75.75 |
| 8 | Pill | 99.10 | 98.99 | 81.39 |
| 9 | Screw | 92.17 | 98.24 | 22.81 |
| 10 | Tile | 100.00 | 96.71 | 66.80 |
| 11 | Toothbrush | 90.83 | 98.47 | 53.98 |
| 12 | Transistor | 99.17 | 96.94 | 64.61 |
| 13 | Wood | 98.86 | 94.14 | 46.65 |
| 14 | Zipper | 99.24 | 97.52 | 35.57 |

### 3.2 Task-wise Performance Progression

**실험 목적**: 각 task 학습 후 전체 task들에 대한 성능 변화 추적

| After Task | Category | Avg Image AUC | Avg Pixel AUC | Routing Acc | Notes |
|------------|----------|---------------|---------------|-------------|-------|
| Task 0 | Bottle | 100.00 | 98.27 | 100.0% | Initial task |
| Task 1 | Cable | 99.51 | 97.94 | 100.0% | |
| Task 2 | Capsule | 98.76 | 98.18 | 100.0% | |
| Task 3 | Carpet | 98.94 | 98.37 | 100.0% | |
| Task 4 | Grid | 99.00 | 98.23 | 100.0% | |
| Task 5 | Hazelnut | 99.15 | 98.31 | 100.0% | |
| Task 6 | Leather | 99.28 | 98.46 | 100.0% | |
| Task 7 | Metal Nut | 99.37 | 98.29 | 100.0% | |
| Task 8 | Pill | 99.34 | 98.37 | 100.0% | |
| Task 9 | Screw | 98.62 | 98.36 | 100.0% | |
| Task 10 | Tile | 98.75 | 98.21 | 100.0% | |
| Task 11 | Toothbrush | 98.09 | 98.23 | 100.0% | |
| Task 12 | Transistor | 98.17 | 98.13 | 100.0% | |
| Task 13 | Wood | 98.22 | 97.84 | 100.0% | |
| Task 14 | Zipper | **98.29** | **97.82** | **100.0%** | Final performance |

*Note: All previous tasks maintain their original performance after each new task training (no forgetting).*

---

## 4. Forgetting Analysis

### 4.1 Catastrophic Forgetting

**실험 목적**: 새로운 task 학습 시 이전 task 성능 유지 능력 평가

**Forgetting Metric**:
\[ F_i = \max_{j \in \{1, ..., T-1\}} (A_{i,j} - A_{i,T}) \]

where \( A_{i,j} \) is the accuracy on task \( i \) after learning task \( j \).

| Task ID | Category | Initial AUC | Final AUC | Forgetting | Status |
|---------|----------|-------------|-----------|------------|--------|
| 0 | Bottle | 100.00 | 100.00 | 0.00 | ✓ No Forgetting |
| 1 | Cable | 99.03 | 99.03 | 0.00 | ✓ No Forgetting |
| 2 | Capsule | 97.25 | 97.25 | 0.00 | ✓ No Forgetting |
| 3 | Carpet | 99.48 | 99.48 | 0.00 | ✓ No Forgetting |
| 4 | Grid | 99.25 | 99.25 | 0.00 | ✓ No Forgetting |
| 5 | Hazelnut | 99.93 | 99.93 | 0.00 | ✓ No Forgetting |
| 6 | Leather | 100.00 | 100.00 | 0.00 | ✓ No Forgetting |
| 7 | Metal Nut | 100.00 | 100.00 | 0.00 | ✓ No Forgetting |
| 8 | Pill | 99.10 | 99.10 | 0.00 | ✓ No Forgetting |
| 9 | Screw | 92.17 | 92.17 | 0.00 | ✓ No Forgetting |
| 10 | Tile | 100.00 | 100.00 | 0.00 | ✓ No Forgetting |
| 11 | Toothbrush | 90.83 | 90.83 | 0.00 | ✓ No Forgetting |
| 12 | Transistor | 99.17 | 99.17 | 0.00 | ✓ No Forgetting |
| 13 | Wood | 98.86 | 98.86 | 0.00 | ✓ No Forgetting |
| **Average Forgetting** | - | - | - | **0.00** | ✓ Perfect |

*Note: MoLE-Flow achieves zero forgetting due to task-specific LoRA adapters combined with perfect routing (100% accuracy).*

### 4.2 Forward Transfer

**실험 목적**: 이전 task 학습이 새로운 task 학습에 미치는 긍정적 영향 측정

**Forward Transfer Metric**:
\[ FT_i = A_{i,i} - A_{i,0} \]

where \( A_{i,i} \) is the accuracy on task \( i \) after learning it, and \( A_{i,0} \) is the accuracy on task \( i \) before any learning (random initialization).

| Task ID | Category | Baseline (Random) | After Training | Forward Transfer | Improvement |
|---------|----------|-------------------|----------------|------------------|-------------|
| 0 | Bottle | TBD | TBD | TBD | - |
| 1 | Cable | TBD | TBD | TBD | TBD |
| 2 | Capsule | TBD | TBD | TBD | TBD |
| 3 | Carpet | TBD | TBD | TBD | TBD |
| 4 | Grid | TBD | TBD | TBD | TBD |
| 5 | Hazelnut | TBD | TBD | TBD | TBD |
| 6 | Leather | TBD | TBD | TBD | TBD |
| 7 | Metal Nut | TBD | TBD | TBD | TBD |
| 8 | Pill | TBD | TBD | TBD | TBD |
| 9 | Screw | TBD | TBD | TBD | TBD |
| 10 | Tile | TBD | TBD | TBD | TBD |
| 11 | Toothbrush | TBD | TBD | TBD | TBD |
| 12 | Transistor | TBD | TBD | TBD | TBD |
| 13 | Wood | TBD | TBD | TBD | TBD |
| 14 | Zipper | TBD | TBD | TBD | TBD |
| **Average FT** | - | - | - | **TBD** | - |

### 4.3 Backward Transfer

**실험 목적**: 새로운 task 학습이 이전 task 성능에 미치는 영향 측정

**Backward Transfer Metric**:
\[ BT_i = A_{i,T} - A_{i,i} \]

where \( A_{i,T} \) is the accuracy on task \( i \) after learning all \( T \) tasks.

| Task ID | Category | After Own Training | After All Tasks | Backward Transfer | Change |
|---------|----------|-------------------|-----------------|-------------------|--------|
| 0 | Bottle | 100.00 | 100.00 | 0.00 | Stable |
| 1 | Cable | 99.03 | 99.03 | 0.00 | Stable |
| 2 | Capsule | 97.25 | 97.25 | 0.00 | Stable |
| 3 | Carpet | 99.48 | 99.48 | 0.00 | Stable |
| 4 | Grid | 99.25 | 99.25 | 0.00 | Stable |
| 5 | Hazelnut | 99.93 | 99.93 | 0.00 | Stable |
| 6 | Leather | 100.00 | 100.00 | 0.00 | Stable |
| 7 | Metal Nut | 100.00 | 100.00 | 0.00 | Stable |
| 8 | Pill | 99.10 | 99.10 | 0.00 | Stable |
| 9 | Screw | 92.17 | 92.17 | 0.00 | Stable |
| 10 | Tile | 100.00 | 100.00 | 0.00 | Stable |
| 11 | Toothbrush | 90.83 | 90.83 | 0.00 | Stable |
| 12 | Transistor | 99.17 | 99.17 | 0.00 | Stable |
| 13 | Wood | 98.86 | 98.86 | 0.00 | Stable |
| **Average BT** | - | - | - | **0.00** | Stable |

*Note: Due to task-specific LoRA adapters with frozen base weights, there is no backward transfer (positive or negative).*

### 4.4 Overall Continual Learning Metrics

| Metric | Formula | Value | Interpretation |
|--------|---------|-------|----------------|
| **Average Accuracy (Image AUC)** | \( \bar{A} = \frac{1}{T} \sum_{i=1}^{T} A_{i,T} \) | **98.29%** | Final performance across all tasks |
| **Average Accuracy (Pixel AUC)** | \( \bar{A} = \frac{1}{T} \sum_{i=1}^{T} A_{i,T} \) | **97.82%** | Final performance across all tasks |
| **Average Accuracy (Pixel AP)** | \( \bar{A} = \frac{1}{T} \sum_{i=1}^{T} A_{i,T} \) | **54.20%** | Final performance across all tasks |
| **Average Forgetting** | \( \bar{F} = \frac{1}{T-1} \sum_{i=1}^{T-1} F_i \) | **0.00%** | Perfect: No forgetting |
| **Average Forward Transfer** | \( \bar{FT} = \frac{1}{T-1} \sum_{i=2}^{T} FT_i \) | N/A | Not applicable (task-specific experts) |
| **Average Backward Transfer** | \( \bar{BT} = \frac{1}{T-1} \sum_{i=1}^{T-1} BT_i \) | **0.00%** | Stable: No performance change |
| **Routing Accuracy** | - | **100.0%** | Perfect task identification |

---

## 5. Efficiency Analysis

### 5.1 Number of Parameters

**실험 목적**: 모델 크기 및 파라미터 효율성 비교

| Component | Parameters | Trainable | Percentage | Notes |
|-----------|------------|-----------|------------|-------|
| Backbone (WRN50) | TBD | No | - | Frozen after Task 0 |
| LoRA Adapters (per task) | TBD | Yes | - | Rank=64 |
| MoLE SubNet (per task) | TBD | Yes | - | 8 blocks |
| DIA Blocks (shared) | TBD | Yes | - | 4 blocks |
| SpatialContextMixer | TBD | Yes | - | Shared across tasks |
| WhiteningAdapter | TBD | Yes | - | Per task |
| **Total (1 task)** | **TBD** | **TBD** | **100%** | - |
| **Total (15 tasks)** | **TBD** | **TBD** | - | Full MVTecAD |

**Comparison with Baselines**:

| Method | Total Parameters | Trainable Parameters | Memory (MB) | Relative Size |
|--------|------------------|---------------------|-------------|---------------|
| **MoLE-Flow (Ours)** | TBD | TBD | TBD | 1.0x |
| FastFlow | TBD | TBD | TBD | TBD |
| CFLOW-AD | TBD | TBD | TBD | TBD |
| CS-Flow | TBD | TBD | TBD | TBD |
| DiffNet | TBD | TBD | TBD | TBD |
| PatchCore | TBD | TBD | TBD | TBD |

### 5.2 Inference Speed

**실험 목적**: 실시간 추론 성능 평가

**Test Configuration**:
- Hardware: TBD (e.g., NVIDIA RTX 3090)
- Batch Size: 1, 8, 32
- Image Resolution: 256×256
- Measurement: Average over 1000 iterations

| Method | Batch=1 (ms) | Batch=8 (ms) | Batch=32 (ms) | FPS (Batch=1) | Throughput (img/s) |
|--------|--------------|--------------|---------------|---------------|-------------------|
| **MoLE-Flow (Ours)** | TBD | TBD | TBD | TBD | TBD |
| FastFlow | TBD | TBD | TBD | TBD | TBD |
| CFLOW-AD | TBD | TBD | TBD | TBD | TBD |
| CS-Flow | TBD | TBD | TBD | TBD | TBD |
| DiffNet | TBD | TBD | TBD | TBD | TBD |
| PatchCore | TBD | TBD | TBD | TBD | TBD |

**Inference Time Breakdown (MoLE-Flow)**:

| Component | Time (ms) | Percentage | Notes |
|-----------|-----------|------------|-------|
| Feature Extraction | TBD | TBD% | Backbone forward pass |
| Whitening Adapter | TBD | TBD% | Input normalization |
| MoLE SubNet | TBD | TBD% | 8 coupling blocks |
| DIA Blocks | TBD | TBD% | 4 blocks |
| Score Aggregation | TBD | TBD% | Top-k selection |
| **Total** | **TBD** | **100%** | - |

### 5.3 Training Speed

**실험 목적**: 학습 효율성 및 수렴 속도 평가

**Training Configuration**:
- Hardware: TBD (e.g., NVIDIA RTX 3090)
- Batch Size: 16
- Epochs: 60
- Dataset: MVTecAD (single category)

| Method | Time per Epoch (s) | Total Training Time (min) | GPU Memory (GB) | Convergence Epoch |
|--------|-------------------|---------------------------|-----------------|-------------------|
| **MoLE-Flow (Ours)** | TBD | TBD | TBD | TBD |
| FastFlow | TBD | TBD | TBD | TBD |
| CFLOW-AD | TBD | TBD | TBD | TBD |
| CS-Flow | TBD | TBD | TBD | TBD |
| DiffNet | TBD | TBD | TBD | TBD |
| PatchCore | TBD | TBD | TBD | TBD |

**Training Time for Full MVTecAD (15 tasks)**:

| Setting | Total Time (hours) | Avg per Task (min) | Notes |
|---------|-------------------|-------------------|-------|
| Sequential Training | TBD | TBD | MoLE-Flow continual learning |
| Independent Training | TBD | TBD | Train each task separately |
| Joint Training | TBD | TBD | Train all tasks together (baseline) |

---

## 6. Continual Learning Protocol

### 6.1 Task Sequence

**MVTecAD Task Order** (15 categories):
1. Bottle
2. Cable
3. Capsule
4. Carpet
5. Grid
6. Hazelnut
7. Leather
8. Metal Nut
9. Pill
10. Screw
11. Tile
12. Toothbrush
13. Transistor
14. Wood
15. Zipper

**ViSA Task Order** (12 categories):
1. Candle
2. Capsules
3. Cashew
4. Chewinggum
5. Fryum
6. Macaroni1
7. Macaroni2
8. PCB1
9. PCB2
10. PCB3
11. PCB4
12. Pipe Fryum

### 6.2 Training Protocol

| Phase | Description | Parameters Updated | Epochs |
|-------|-------------|-------------------|--------|
| **Task 0** | Initial task training | Backbone + All modules | 60 |
| **Task 1-14** | Continual learning | LoRA + Flow modules (Backbone frozen) | 60 per task |

**Key Settings**:
- **Backbone**: Frozen after Task 0
- **LoRA Adapters**: Task-specific, trained from scratch for each task
- **Flow Modules**: Continually updated with new task data
- **Evaluation**: Test on all seen tasks after each new task training

### 6.3 Evaluation Protocol

**Metrics Computed**:
- Image-level AUC, AP
- Pixel-level AUC, AP
- Routing Accuracy (task identification)
- Forgetting, Forward Transfer, Backward Transfer

**Evaluation Frequency**:
- After each task training
- On all previously seen tasks
- Using full test set for each category

### 6.4 Comparison Settings

| Setting | Description | Purpose |
|---------|-------------|---------|
| **Continual (Ours)** | Sequential task learning with frozen backbone | Main approach |
| **Joint Training** | Train on all tasks simultaneously | Upper bound |
| **Fine-tuning** | Sequential training without continual learning mechanisms | Lower bound (catastrophic forgetting) |
| **Multi-task** | Separate model per task | Upper bound (no forgetting) |

---

## 7. Visualization and Qualitative Results

### 7.1 Anomaly Localization Examples

**실험 목적**: 각 category별 이상 탐지 및 위치 추정 정성적 평가

| Category | Sample Images | Anomaly Maps | Ground Truth | Notes |
|----------|---------------|--------------|--------------|-------|
| Bottle | TBD | TBD | TBD | TBD |
| Cable | TBD | TBD | TBD | TBD |
| Capsule | TBD | TBD | TBD | TBD |
| ... | ... | ... | ... | ... |

### 7.2 Feature Space Analysis

**실험 목적**: Task별 feature distribution 및 분리도 시각화

- t-SNE visualization of learned features
- Task embedding space
- Routing decision boundaries

### 7.3 Attention Map Visualization

**실험 목적**: SpatialContextMixer의 attention pattern 분석

- Spatial attention maps
- Scale context attention
- Task-specific attention patterns

---

## Notes

- All experiments use the baseline configuration unless otherwise specified
- TBD entries will be filled as experiments are completed
- Routing Accuracy measures the model's ability to correctly identify which task an input belongs to
- All metrics are averaged over 3 random seeds unless otherwise noted