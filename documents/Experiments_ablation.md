# MoLE-Flow Ablation Study

## Baseline Configuration (MAIN)

**Experiment**: `MVTec-WRN50-TailW0.7-TopK3-TailTopK2-ScaleK5-lr3e-4-MAIN`

| Parameter | Value |
|-----------|-------|
| Backbone | WideResNet50 |
| LoRA Rank | 64 |
| Coupling Layers | 8 |
| DIA Blocks | 4 |
| Epochs | 60 |
| Learning Rate | **3e-4** |
| Adapter Mode | Whitening |
| Tail Weight | 0.7 |
| Score Aggregation | top_k (k=3) |
| Scale Context Kernel | **5** |
| Lambda Logdet | **1e-4** |

**Baseline Performance**:
| Metric | Value |
|--------|-------|
| Image AUC | **98.29%** |
| Pixel AUC | **97.82%** |
| Pixel AP | **54.20%** |
| Routing Accuracy | **100%** |

---


# Architecture Modular Analysis

## 1. Core Component Ablation

> MAIN 설정(lr=3e-4, logdet=1e-4, scale_k=5) 기준으로 실험 완료

### 실험 목록

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o SpatialContextMixer | Spatial Context Mixing의 기여도 | SpatialContextMixer 모듈 제거 (Positional/Spatial mixing off) | ✅ Done |
| w/o WhiteningAdapter | Whitening Adapter의 기여도 | InputAdapter(Whitening) 미적용, SoftLN 사용 | ✅ Done |
| w/o Tail Aware Loss | Tail Aware Loss의 기여도 | Tail Aware Loss 비활성화 (표준 손실 사용) | ✅ Done |
| w/o LogDet Regularization | LogDet Regularization 유무 | LogDet 정규화 항 제거 (lambda_logdet=0) | ✅ Done |
| w/o MoLE subnet | DIA만 사용 (LoRA 미사용) | MoLESubNet/LoRA 제거, DIA만 사용 | 🔄 Running |

### 결과 테이블

| Configuration | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP |
|---------------|---------|---------|--------|--------|--------|-----------|----------|
| **MoLE-Flow (Full)** | **98.29** | **97.82** | **99.31** | **54.20** | 100.0 | - | - |
| w/o SpatialContextMixer | 98.08 | 97.70 | 99.23 | 52.24 | 100.0 | -0.21 | -1.96 |
| w/o WhiteningAdapter | 98.06 | 97.60 | 99.23 | 47.14 | 100.0 | -0.23 | **-7.06** |
| w/o Tail Aware Loss | 96.62 | 97.20 | 98.66 | 45.86 | 100.0 | **-1.67** | **-8.34** |
| w/o LogDet Regularization | 98.29 | 97.66 | 99.31 | 51.06 | 100.0 | 0.00 | -3.14 |
| w/o MoLE subnet | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 분석

1. **Tail Aware Loss**가 가장 큰 영향 (Img AUC -1.67%, Pix AP -8.34%)
   - 손실 함수에서 tail patch에 대한 집중이 성능에 핵심적

2. **WhiteningAdapter** 제거 시 Pix AP -7.06% 감소
   - 분포 정렬이 pixel-level anomaly detection에 중요

3. **LogDet Regularization** 영향 미미 (Img AUC 동일, Pix AP -3.14%)
   - 안정화 효과는 있으나 성능에 큰 영향 없음

4. **SpatialContextMixer** 제거 시 소폭 감소 (-0.21%, -1.96%)
   - 공간적 context mixing의 부가적 기여 확인


---

## 2. MoLE Subnet

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o Scale Context | scale_context 유/무 | Scale Context 모듈 미사용 (`--no_scale_context`) | 🔄 Running |
| w/o LoRA | LoRA 대신 Linear 사용 | LoRA 대신 Regular Linear 사용 (`--use_regular_linear`) | 🔄 Running (GPU 4) |


### 결과 테이블

| Configuration | Img AUC | Pix AUC | Pix AP | Δ Pix AP |
|---------------|---------|---------|--------|----------|
| **MoLE-Flow (Full)** | **98.29** | **97.82** | **54.20** | - |
| w/o Scale Context | TBD | TBD | TBD | TBD |
| w/o LoRA | TBD | TBD | TBD | TBD |

---

## 3. Normalizing Flow Block 구성 실험 (MoLE / DIA Block 조합)

전체 Coupling Block 수(=8)는 동일하게 고정하고, MoLE-SubNet과 DIA block의 구성 비율에 따라 성능이 어떻게 달라지는지 실험합니다.

| MoLE Blocks | DIA Blocks | Img AUC | Pix AUC | Pix AP | 비고 |
|-------------|-----------|---------|---------|--------|------|
| **8**       | 4         | 98.29   | 97.82   | 54.20  | MoLE-Flow(Full, 총 12블록; 기존 실험 결과) |
| 10          | 2         | 🔄 Running (GPU 5) | - | - | 총 12블록 |
| 6           | 6         | TBD     | TBD     | TBD    |  |
| 4           | 8         | TBD     | TBD     | TBD    |  |
| 0           | 12        | TBD     | TBD     | TBD    | DIA-only (총 12블록) |

- 실험 목적: Coupling Layer 내 MoLE/DIA 비중 변화가 모델 성능에 미치는 영향 조사
- 실험 세팅: 전체 coupling layer 수(8)는 고정, MoLE-SubNet 및 DIA block 개수만 조절



## 4. Base Weight Sharing vs. Sequential/Independent Training

Base backbone의 가중치 공유(sequential/independent) 방식에 따른 continual setting의 영향 분석을 위해 아래 3가지 설정을 비교합니다.

| 설정                      | Description                                                  | Img AUC | Pix AUC | Img AP | Pix AP | 비고          |
|---------------------------|-------------------------------------------------------------|---------|---------|--------|--------|---------------|
| (a) **Base Frozen(default)**       | Base Weight Task 0 학습 후 고정 (freeze), downstream만 학습          | TBD     | TBD     | TBD    | TBD    | 파라미터 최소화 |
| (b) **Sequential Training**| Base Weight는 모든 task에서 공유하되 순차적으로 학습 | TBD     | TBD     | TBD    | TBD    | catastrophic forgetting 현상 확인 |
| (c) **Complete Separated**| 각 task별로 base+flow 완전 독립 (multi-head, 파라미터 x15) | TBD     | TBD     | TBD    | TBD    | upper bound, 정보 공유 없음 |

**실험 목적:**  
- Base backbone의 동결, 순차 학습, 완전 독립 세팅 간 성능/일반화/forgetting trade-off 비교
- 실제 deployment scenario에 맞는 가중치 공유 전략 도출



### 4.2 Task-Separated vs Shared (Upper/Lower Bound)

| Design               | Img AUC | Pix AUC | Pix AP | Parameters |
|----------------------|---------|---------|--------|------------|
| **MoLE-Flow (MAIN)** | **98.29** | **97.82** | **54.20** | 1.0x       |
| Sequential Training  | TBD     | TBD     | TBD    | 1.0x       |
| Complete Separated   | TBD     | TBD     | TBD    | 15.0x      |

# Hyperparameter Analysis 

## lora_rank
> 기준: lr=3e-4, logdet=1e-4, scale_k=5

| lora_rank | Img AUC | Pix AUC | Pix AP | 비고 |
|-----------|---------|---------|--------|------|
| 16        | TBD     | TBD     | TBD    | 파라미터 최소 |
| 32        | TBD     | TBD     | TBD    | 균형 |
| **64**    | **98.30** | **97.83** | **54.04** | **MAIN 기준** |
| 128       | 98.36   | 97.80   | 52.42  | 80ep, DIA5, C10 실험 결과 |

## lambda_logdet
> 기준: lr=3e-4, scale_k=5

| lambda_logdet | Img AUC | Pix AUC | Pix AP | 비고 |
|---------------|---------|---------|--------|------|
| 5e-4          | 97.70   | 97.55   | 52.35  | 과도한 정규화 |
| 2e-4          | 98.19   | 97.79   | 54.18  | Pix AP 최고 |
| **1e-4**      | **98.29** | **97.82** | **54.20** | **MAIN 기준, 권장값** |
| 1e-5          | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |
| 0             | 98.29   | 97.66   | 51.06  | Ablation-Core 실험 |

## scale_context_kernel
> 기준: lr=3e-4, logdet=1e-4

| scale_context_kernel | Img AUC | Pix AUC | Pix AP | 비고 |
|---------------------|---------|---------|--------|------|
| 3                   | 98.36   | 97.71   | 52.16  | 기본값 |
| **5**               | **98.36** | **97.80** | **52.42** | **MAIN 기준, 권장값** |
| 7                   | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |

## spatial_context_kernel
> 기준: lr=3e-4, logdet=1e-4, scale_k=5
> Note: 모든 실험이 spatial_k=3으로 고정되어 MAIN 기반 비교 불가

| spatial_context_kernel | Img AUC | Pix AUC | Pix AP | 비고 |
|-----------------------|---------|---------|--------|------|
| **3**                 | **98.29** | **97.82** | **54.20** | **MAIN 기준** |
| 5                     | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |
| 7                     | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |

## Tail Aware Loss weight (tail_weight)
> 기준: lr=3e-4, logdet=1e-4, scale_k=5, topk=3

| tail_weight | Img AUC | Pix AUC | Pix AP | 비고 |
|-------------|---------|---------|--------|------|
| 0 (off)     | 96.62   | 97.20   | 45.86  | Ablation-Core 실험 (**-8.34 Pix AP**) |
| 0.3         | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |
| 0.5         | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |
| 0.65        | 98.24   | 97.81   | 53.95  | topk=3, tail_topk=3 |
| **0.7**     | **98.29** | **97.82** | **54.20** | **MAIN 기준, 권장값** |
| 1.0         | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |

## Image Anomaly Score Aggregation K (score_aggregation_top_k)
> 기준: lr=3e-4, logdet=1e-4, scale_k=5, tw=0.7

| top_k | Img AUC | Pix AUC | Pix AP | 비고 |
|-------|---------|---------|--------|------|
| 1     | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |
| **3** | **98.29** | **97.82** | **54.20** | **MAIN 기준, 권장값** |
| 5     | 98.30   | 97.83   | 54.04  | tail_topk=3 |
| 7     | TBD     | TBD     | TBD    | MAIN 기반 실험 필요 |

