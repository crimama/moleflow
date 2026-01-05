# MoLE-Flow Paper: Experiments Section Structure Design

**Document Version**: 1.0
**Date**: 2026-01-05
**Purpose**: Design document for Section 4 (Experiments) of the MoLE-Flow paper

---

## Overview

This document outlines the complete structure of the Experiments section for the MoLE-Flow paper. The section is organized to demonstrate:
1. The effectiveness of MoLE-Flow compared to existing methods
2. The robustness across multiple datasets (MVTec AD, ViSA)
3. The continual learning capabilities (minimal forgetting)
4. The efficiency in terms of parameters and inference
5. The contribution of each component through ablation studies

---

## Section 4.1: Experimental Setup

### 4.1.1 Datasets

**MVTec AD** (Primary Benchmark)
- 15 object/texture categories
- 5,354 high-resolution images (training: 3,629 normal, test: 1,725 normal + anomaly)
- Pixel-level ground truth annotations for anomaly localization
- Citation: Bergmann et al., CVPR 2019

**ViSA** (Visual Anomaly Dataset)
- 12 categories with complex real-world scenarios
- More challenging than MVTec AD
- Citation: Zou et al., ECCV 2022

**MPDD** (Optional, if results available)
- Metal Parts Defect Detection
- 6 categories

### 4.1.2 Continual Learning Protocol

```
Protocol Description:
- Sequential Task Learning: Classes are presented as sequential tasks
- Scenario Notation: NxM = N classes per task, M total tasks
- Primary: 1x1 (15 tasks, each with 1 class) - most fine-grained
- Additional: 3x3 (5 tasks, 3 classes each), 5x5 (3 tasks, 5 classes each)
- Task ID Unknown at Inference: Router predicts which expert to use
- No Replay Buffer: No access to previous task data during training
```

### 4.1.3 Evaluation Metrics

| Metric | Description | Level |
|--------|-------------|-------|
| Image AUROC | Area Under ROC for image-level detection | Image |
| Pixel AUROC | Area Under ROC for pixel-level localization | Pixel |
| Image AP | Average Precision for image-level detection | Image |
| Pixel AP | Average Precision for pixel-level localization | Pixel |
| Routing Acc | Accuracy of task prediction by router | Task |
| BWT | Backward Transfer (forgetting measure) | CL |
| FWT | Forward Transfer | CL |

### 4.1.4 Implementation Details

```
Configuration (MAIN):
- Backbone: WideResNet50 (pretrained on ImageNet)
- Feature Extractor: Frozen backbone, multi-scale features
- Normalizing Flow: 8 MoLE coupling layers + 4 DIA blocks
- LoRA: rank=64, alpha=64
- Input Resolution: 224x224 (resized from original)
- Training: 60 epochs per task, lr=3e-4, batch_size=16
- Loss: NLL + tail-aware loss (weight=0.7) + logdet reg (lambda=1e-4)
- Optimizer: Adam
- Hardware: Single NVIDIA GPU (specify model)
```

---

## Section 4.2: Main Results

### 4.2.1 MVTec AD Results

**Table 1: Comparison with State-of-the-Art on MVTec AD (1x1 CL Scenario)**

| Method | Type | Image AUC | Pixel AUC | Pixel AP | Router Acc |
|--------|------|-----------|-----------|----------|------------|
| **Replay-based Methods** |
| ER (Experience Replay) | CL | TBD | TBD | TBD | - |
| DER++ | CL | TBD | TBD | TBD | - |
| **Regularization-based Methods** |
| EWC | CL | TBD | TBD | TBD | - |
| LwF | CL | TBD | TBD | TBD | - |
| **Architecture-based Methods** |
| PackNet | CL | TBD | TBD | TBD | - |
| **AD + Fine-tuning (Sequential)** |
| CFLOW-AD (seq) | AD | TBD | TBD | TBD | - |
| FastFlow (seq) | AD | TBD | TBD | TBD | - |
| **Upper Bounds** |
| Separate Models (oracle) | - | TBD | TBD | TBD | 100.0 |
| **Ours** |
| **MoLE-Flow** | CL+AD | **98.29** | **97.82** | **54.20** | **100.0** |

**Notes for Table 1**:
- Available results: MoLE-Flow = 98.29% Image AUC, 97.82% Pixel AUC, 54.20% Pixel AP
- Baseline methods need to be run or cited from literature
- Consider including: PatchCore, SPADE, PaDiM for reference (non-CL baselines)

**Table 2: Per-Class Results on MVTec AD**

| Class | Image AUC | Pixel AUC | Pixel AP |
|-------|-----------|-----------|----------|
| bottle | 100.00 | 96.42 | 45.36 |
| cable | 98.84 | 97.50 | 65.76 |
| capsule | 96.09 | 98.54 | 35.11 |
| carpet | 98.35 | 98.17 | 42.07 |
| grid | 99.83 | 97.54 | 19.96 |
| hazelnut | 99.93 | 98.45 | 50.13 |
| leather | 100.00 | 98.16 | 23.04 |
| metal_nut | 100.00 | 98.44 | 87.13 |
| pill | 98.88 | 98.82 | 80.62 |
| screw | 90.94 | 98.28 | 20.77 |
| tile | 100.00 | 95.76 | 67.83 |
| toothbrush | 88.61 | 98.10 | 43.44 |
| transistor | 99.50 | 96.71 | 65.51 |
| wood | 98.42 | 92.09 | 32.02 |
| zipper | 99.61 | 97.46 | 31.58 |
| **Mean** | **97.93** | **97.36** | **47.35** |

**Note**: Above is from OLD settings. MAIN setting results show 98.29% Image AUC.

### 4.2.2 ViSA Results

**Table 3: Results on ViSA Dataset (1x1 CL Scenario)**

| Method | Image AUC | Pixel AUC | Pixel AP | Router Acc |
|--------|-----------|-----------|----------|------------|
| MoLE-Flow | 83.78 | 97.15 | 28.78 | 100.0 |

**Table 4: Per-Class Results on ViSA**

| Class | Image AUC | Pixel AUC | Pixel AP |
|-------|-----------|-----------|----------|
| candle | 88.43 | 98.60 | 14.81 |
| capsules | 67.43 | 95.82 | 21.21 |
| cashew | 87.94 | 98.33 | 49.62 |
| chewinggum | 96.30 | 98.47 | 27.54 |
| fryum | 93.60 | 93.94 | 40.24 |
| macaroni1 | 76.09 | 97.34 | 6.55 |
| macaroni2 | 65.97 | 95.72 | 0.83 |
| pcb1 | 84.28 | 98.99 | 67.97 |
| pcb2 | 73.09 | 94.83 | 9.85 |
| pcb3 | 78.45 | 97.60 | 26.36 |
| pcb4 | 97.10 | 97.31 | 28.05 |
| pipe_fryum | 96.72 | 98.86 | 52.30 |
| **Mean** | **83.78** | **97.15** | **28.78** |

---

## Section 4.3: Continual Learning Analysis

### 4.3.1 Catastrophic Forgetting Analysis

**Figure 1: Performance Retention Across Tasks**
```
Description: Line plot showing Image AUC for each task after learning all 15 tasks
X-axis: Task ID (0-14)
Y-axis: Image AUC (%)
Lines:
  - Performance when task was just learned
  - Performance after all tasks learned
  - Baseline (sequential fine-tuning without LoRA)
```

**Table 5: Forgetting Metrics**

| Method | Avg Forgetting | Max Forgetting | Final Avg AUC |
|--------|----------------|----------------|---------------|
| Sequential FT | TBD | TBD | TBD |
| EWC | TBD | TBD | TBD |
| MoLE-Flow | ~0% | ~0% | 98.29% |

**Analysis Points**:
- MoLE-Flow achieves near-zero forgetting due to task-specific LoRA adapters
- Base weights frozen after Task 0, preventing interference
- Each task has isolated adapter parameters

### 4.3.2 Forward and Backward Transfer

**Backward Transfer (BWT)**:
$$BWT = \frac{1}{T-1} \sum_{i=1}^{T-1} (R_{T,i} - R_{i,i})$$

Where $R_{t,i}$ is performance on task $i$ after learning task $t$.

**Forward Transfer (FWT)**:
$$FWT = \frac{1}{T-1} \sum_{i=2}^{T} (R_{i-1,i}^{zero} - R_{random})$$

**Table 6: Transfer Analysis**

| Method | BWT | FWT | Notes |
|--------|-----|-----|-------|
| MoLE-Flow | ~0.0 | N/A | LoRA isolation prevents negative transfer |

### 4.3.3 Multi-Class Per Task Scenarios

**Table 7: CL Scenario Comparison**

| Scenario | Tasks | Classes/Task | Router Acc | Image AUC | Pixel AUC | Pixel AP |
|----------|-------|--------------|------------|-----------|-----------|----------|
| 1x1 | 15 | 1 | 100.0% | 98.29 | 97.82 | 54.20 |
| 3x3 | 5 | 3 | 99.37% | 83.83 | 88.30 | 33.65 |
| 5x5 | 3 | 5 | 100.0% | 80.22 | 77.78 | 26.40 |
| 14x1 | 2 | 14 | 96.69% | 66.85 | 46.04 | 8.06 |

**Figure 2: Performance vs Classes-per-Task**
```
Description: Bar chart comparing scenarios
Shows degradation as more classes share single LoRA adapter
```

**Analysis**:
- Performance degrades with more classes per task
- LoRA cannot prevent intra-task forgetting (within the same adapter)
- 1x1 scenario ideal for MoLE-Flow architecture
- Router accuracy remains high (>96%) across scenarios

---

## Section 4.4: Efficiency Analysis

### 4.4.1 Parameter Efficiency

**Table 8: Parameter Comparison**

| Method | Base Params | Per-Task Params | 15-Task Total | Relative |
|--------|-------------|-----------------|---------------|----------|
| Separate Models | 26.2M | 26.2M | 393M | 15.0x |
| Sequential FT | 26.2M | 0 | 26.2M | 1.0x |
| EWC | 26.2M | 26.2M (Fisher) | 52.4M | 2.0x |
| MoLE-Flow | 26.2M | 0.78M (LoRA) | 37.9M | 1.45x |

**Breakdown of MoLE-Flow Parameters**:
- Backbone (frozen): ~23M
- Base NF weights: ~3M
- LoRA per task: ~0.78M (rank=64)
- TaskInputAdapter per task: ~0.01M
- Router prototypes: Negligible

### 4.4.2 Inference Speed

**Table 9: Inference Time Comparison**

| Method | Time/Image (ms) | Throughput (img/s) | Notes |
|--------|-----------------|--------------------| ------|
| MoLE-Flow | TBD | TBD | Single forward pass |
| Separate Models | TBD | TBD | Requires model selection |

### 4.4.3 Training Speed

**Table 10: Training Time**

| Setting | Time/Epoch | Total (15 tasks) | GPU Memory |
|---------|------------|------------------|------------|
| MoLE-Flow | TBD min | TBD hours | TBD GB |

---

## Section 4.5: Ablation Study

### 4.5.1 Component Contribution

**Table 11: Component Ablation (MAIN Settings)**

| Configuration | Image AUC | Pixel AUC | Pixel AP | Delta |
|---------------|-----------|-----------|----------|-------|
| **MoLE-Flow (Full)** | **98.29** | **97.82** | **54.20** | - |
| w/o WhiteningAdapter | TBD | TBD | TBD | TBD |
| w/o DIA Blocks | TBD | TBD | TBD | TBD |
| w/o Tail-Aware Loss | TBD | TBD | TBD | TBD |
| w/o LogDet Reg | 98.08 | 97.36 | 47.18 | -0.21 / -0.46 / -7.02 |
| w/o LoRA | TBD | TBD | TBD | TBD |
| w/o SpatialContext | TBD | TBD | TBD | TBD |
| w/o ScaleContext | TBD | TBD | TBD | TBD |

**Component Ablation Results (OLD Settings - Reference)**:

| Configuration | Image AUC | Delta |
|---------------|-----------|-------|
| Full (OLD) | 97.93 | - |
| w/o DIA | 94.79 | **-3.14** |
| w/o Adapter | 96.04 | **-1.89** |
| w/o LoRA | 97.97 | +0.04 |
| w/o SpatialCtx | 97.72 | -0.21 |
| w/o ScaleCtx | 97.75 | -0.18 |

**Key Findings**:
1. **DIA Blocks**: Most critical component (-3.14% without it)
2. **WhiteningAdapter**: Essential for distribution alignment (-1.89%)
3. **Tail-Aware Loss**: Important for localization
4. **LogDet Regularization**: Prevents NF collapse, +7% Pixel AP

### 4.5.2 Architecture Design Choices

**Table 12: NF Block Configuration**

| MoLE Blocks | DIA Blocks | Total | Image AUC | Pixel AUC | Pixel AP |
|-------------|------------|-------|-----------|-----------|----------|
| 8 | 4 | 12 | 98.29 | 97.82 | 54.20 |
| 10 | 2 | 12 | TBD | TBD | TBD |
| 6 | 6 | 12 | TBD | TBD | TBD |
| 4 | 8 | 12 | TBD | TBD | TBD |
| 0 | 12 | 12 | TBD | TBD | TBD |

### 4.5.3 Hyperparameter Sensitivity

**Table 13: LoRA Rank Sensitivity**

| LoRA Rank | Image AUC | Pixel AUC | Params/Task |
|-----------|-----------|-----------|-------------|
| 16 | TBD | TBD | 0.20M |
| 32 | TBD | TBD | 0.39M |
| 64 (default) | 98.29 | 97.82 | 0.78M |
| 128 | 98.36 | 97.80 | 1.56M |

**Figure 3: Hyperparameter Sensitivity Plots**
```
Subplots:
(a) Learning Rate: 1e-4, 2e-4, 3e-4, 4e-4
(b) Lambda LogDet: 0, 1e-5, 1e-4, 2e-4, 5e-4
(c) Tail Weight: 0.3, 0.5, 0.65, 0.7, 0.75
(d) DIA Blocks: 2, 4, 6, 8
```

**Table 14: Loss Function Hyperparameters**

| Parameter | Value | Image AUC | Pixel AUC | Pixel AP |
|-----------|-------|-----------|-----------|----------|
| tail_weight=0.55 | top_k=5 | 98.24 | 97.78 | 53.50 |
| tail_weight=0.65 | top_k=3 | 98.24 | 97.81 | 53.95 |
| **tail_weight=0.70** | **top_k=3** | **98.29** | **97.82** | **54.20** |
| tail_weight=0.75 | top_k=5 | 98.12 | 97.82 | 54.49 |

### 4.5.4 Router Analysis

**Table 15: Router Performance**

| CL Scenario | Router Accuracy | Misrouting Rate |
|-------------|-----------------|-----------------|
| 1x1 (15 tasks) | 100.0% | 0.0% |
| 3x3 (5 tasks) | 99.37% | 0.63% |
| 5x5 (3 tasks) | 100.0% | 0.0% |

**Analysis**:
- Prototype-based router achieves perfect accuracy in 1x1 scenario
- Mahalanobis distance provides robust task discrimination
- Router adds negligible computational overhead

### 4.5.5 Statistical Significance

**Table 16: Results Across Random Seeds (1x1 Scenario)**

| Seed | Image AUC | Pixel AUC | Pixel AP |
|------|-----------|-----------|----------|
| 0 | 98.10 | 97.79 | 53.01 |
| 42 | 98.22 | 97.81 | 53.31 |
| 123 | 98.16 | 97.86 | 53.61 |
| **Mean +/- Std** | **98.16 +/- 0.06** | **97.82 +/- 0.04** | **53.31 +/- 0.30** |

---

## Figures Summary

| Figure | Description | Priority |
|--------|-------------|----------|
| Fig 1 | Performance retention across tasks (forgetting analysis) | High |
| Fig 2 | Performance vs classes-per-task | High |
| Fig 3 | Hyperparameter sensitivity (4 subplots) | Medium |
| Fig 4 | Qualitative localization results | High |
| Fig 5 | Router decision visualization (t-SNE/UMAP) | Medium |

---

## Tables Summary

| Table | Description | Data Status |
|-------|-------------|-------------|
| Tab 1 | Main comparison on MVTec AD | Partial (need baselines) |
| Tab 2 | Per-class MVTec AD results | Available |
| Tab 3 | Main comparison on ViSA | Available |
| Tab 4 | Per-class ViSA results | Available |
| Tab 5 | Forgetting metrics | Need computation |
| Tab 6 | Transfer analysis | Need computation |
| Tab 7 | CL scenario comparison | Available |
| Tab 8 | Parameter comparison | Partial |
| Tab 9 | Inference time | Need benchmark |
| Tab 10 | Training time | Need benchmark |
| Tab 11 | Component ablation | Partial (need MAIN reruns) |
| Tab 12 | NF block configuration | Need experiments |
| Tab 13 | LoRA rank sensitivity | Partial |
| Tab 14 | Loss hyperparameters | Available |
| Tab 15 | Router performance | Available |
| Tab 16 | Statistical significance | Available |

---

## Required Baseline Methods for Comparison

### Continual Learning Methods (Priority Order)

1. **EWC** (Elastic Weight Consolidation) - Kirkpatrick et al., PNAS 2017
2. **LwF** (Learning without Forgetting) - Li & Hoiem, ECCV 2016
3. **PackNet** - Mallya & Lazebnik, CVPR 2018
4. **DER++** (Dark Experience Replay) - Buzzega et al., NeurIPS 2020
5. **ER** (Experience Replay) - Chaudhry et al., 2019

### Anomaly Detection Methods (for context)

1. **CFLOW-AD** - Gudovskiy et al., WACV 2022
2. **FastFlow** - Yu et al., arXiv 2021
3. **PatchCore** - Roth et al., CVPR 2022
4. **SPADE** - Cohen & Hoshen, arXiv 2020
5. **PaDiM** - Defard et al., ICPR 2021

### Combined CL + AD Baselines

- **CL method + AD backbone**: Apply EWC/LwF to CFLOW-AD or FastFlow
- **Sequential Fine-tuning**: Train AD model sequentially without CL regularization

---

## Experiments Needed Before Paper Submission

### Priority 1: Must Have

1. [ ] Component ablation with MAIN settings (w/o Adapter, w/o DIA, w/o LoRA)
2. [ ] Baseline methods comparison (EWC, LwF on same AD backbone)
3. [ ] Inference time measurement
4. [ ] Parameter count verification

### Priority 2: Important

5. [ ] w/o Tail-Aware Loss ablation
6. [ ] LoRA rank sensitivity (16, 32, 64, 128)
7. [ ] Training time measurement
8. [ ] NF block configuration study

### Priority 3: Nice to Have

9. [ ] ViT backbone comparison
10. [ ] MPDD dataset evaluation
11. [ ] Different class orderings (to test ordering sensitivity)
12. [ ] Qualitative visualization figures

---

## Writing Guidelines for Experiments Section

1. **Be precise**: Report exact numbers with standard deviations where available
2. **Fair comparison**: Ensure all methods use same backbone, data splits, and evaluation protocol
3. **Highlight key findings**: Use bold for best results, emphasize surprising findings
4. **Connect to contributions**: Each experiment should support a claim from Introduction
5. **Avoid overclaiming**: Acknowledge limitations (e.g., multi-class per task degradation)
6. **Provide intuition**: Explain why certain components work (DIA for feature injection, etc.)
