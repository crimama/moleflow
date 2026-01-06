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
| w/o MoLE subnet | DIA만 사용 (LoRA 미사용) | MoLESubNet/LoRA 제거, DIA만 사용 | ✅ Done |

### 결과 테이블

| Configuration | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP |
|---------------|---------|---------|--------|--------|--------|-----------|----------|
| **MoLE-Flow (Full)** | **98.29** | **97.82** | **99.31** | **54.20** | 100.0 | - | - |
| w/o SpatialContextMixer | 98.08 | 97.70 | 99.23 | 52.24 | 100.0 | -0.21 | -1.96 |
| w/o WhiteningAdapter | 98.06 | 97.60 | 99.23 | 47.14 | 100.0 | -0.23 | **-7.06** |
| w/o Tail Aware Loss | 96.62 | 97.20 | 98.66 | 45.86 | 100.0 | **-1.67** | **-8.34** |
| w/o LogDet Regularization | 98.29 | 97.66 | 99.31 | 51.06 | 100.0 | 0.00 | -3.14 |
| w/o MoLE subnet (DIA only) | 98.37 | 97.84 | 99.32 | 54.16 | 100.0 | +0.08 | -0.04 |

### 분석

1. **Tail Aware Loss**가 가장 큰 영향 (Img AUC -1.67%, Pix AP -8.34%)
   - 손실 함수에서 tail patch에 대한 집중이 성능에 핵심적

2. **WhiteningAdapter** 제거 시 Pix AP -7.06% 감소
   - 분포 정렬이 pixel-level anomaly detection에 중요

3. **LogDet Regularization** 영향 미미 (Img AUC 동일, Pix AP -3.14%)
   - 안정화 효과는 있으나 성능에 큰 영향 없음

4. **SpatialContextMixer** 제거 시 소폭 감소 (-0.21%, -1.96%)
   - 공간적 context mixing의 부가적 기여 확인

5. **MoLE subnet 제거 (DIA only)** 시 성능 유지 (+0.08%, -0.04%)
   - DIA만으로도 유사한 성능 달성 가능
   - LoRA 기반 MoLE subnet의 기여도가 예상보다 낮음
   - **주의**: Continual learning 시나리오에서 LoRA의 역할 재검토 필요

6. **Scale Context** 제거 시 소폭 감소 (-0.21%, -0.27%)
   - s-network의 local context 주입이 미세하게 기여


---

## 2. MoLE Subnet

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o Scale Context | scale_context 유/무 | Scale Context 모듈 미사용 (`--no_scale_context`) | ✅ Done |
| w/o LoRA | LoRA 대신 Linear 사용 | LoRA 대신 Regular Linear 사용 (`--use_regular_linear`) | ✅ Done |


### 결과 테이블

| Configuration | Img AUC | Pix AUC | Pix AP | Δ Pix AP |
|---------------|---------|---------|--------|----------|
| **MoLE-Flow (Full)** | **98.29** | **97.82** | **54.20** | - |
| w/o Scale Context | 98.08 | 97.84 | 53.93 | -0.27 |
| w/o LoRA (Regular Linear) | 98.29 | 97.82 | 54.20 | 0.00 |

### 분석
- **w/o LoRA**: LoRA 대신 Regular Linear를 사용해도 성능 동일
  - LoRA의 low-rank constraint가 성능에 영향을 주지 않음
  - 파라미터 효율성 관점에서 LoRA 유지 권장

---

## 3. Normalizing Flow Block 구성 실험 (MoLE / DIA Block 조합)

전체 Coupling Block 수(=8)는 동일하게 고정하고, MoLE-SubNet과 DIA block의 구성 비율에 따라 성능이 어떻게 달라지는지 실험합니다.

| MoLE Blocks | DIA Blocks | Img AUC | Pix AUC | Img AP | Pix AP | 비고 |
|-------------|-----------|---------|---------|--------|--------|------|
| **8**       | 4         | **98.29** | 97.82 | 99.31  | **54.20** | MoLE-Flow(Full, 총 12블록) |
| 10          | 2         | 98.27   | 97.73   | 99.31  | **54.70** | 총 12블록, **Pix AP 최고** |
| 6           | 6         | 98.19   | 97.79   | 99.16  | 51.62  | 총 12블록 |
| 4           | 8         | 98.09   | 97.74   | 99.14  | 50.27  | 총 12블록 |
| 0           | 12        | 98.37   | 97.84   | 99.32  | 54.16  | DIA-only (총 12블록) |

- 실험 목적: Coupling Layer 내 MoLE/DIA 비중 변화가 모델 성능에 미치는 영향 조사
- 실험 세팅: 전체 coupling layer 수(8)는 고정, MoLE-SubNet 및 DIA block 개수만 조절

### 분석

1. **MoLE 8 + DIA 4 (MAIN)**: 가장 균형 잡힌 성능 (Img AUC 98.29%, Pix AP 54.20%)
2. **MoLE 10 + DIA 2**: Pix AP 54.70%로 최고 - MoLE block이 많을수록 pixel-level 성능 향상
3. **MoLE 6 + DIA 6**: Pix AP 51.62%로 하락 - DIA 비중 증가 시 성능 저하
4. **MoLE 4 + DIA 8**: Pix AP 50.27%로 가장 낮음 - DIA 비중이 너무 높으면 성능 저하
5. **DIA only (0 + 12)**: Pix AP 54.16%로 양호 - 순수 DIA도 효과적

**결론**: MoLE block 비중이 높을수록 Pix AP 성능이 향상되는 경향. 최적 구성은 **MoLE 10 + DIA 2** (Pix AP 54.70%)



## 4. Base Weight Sharing vs. Sequential/Independent Training

Base backbone의 가중치 공유(sequential/independent) 방식에 따른 continual setting의 영향 분석을 위해 아래 3가지 설정을 비교합니다.

| 설정                      | Description                                                  | Img AUC | Pix AUC | Img AP | Pix AP | 비고          |
|---------------------------|-------------------------------------------------------------|---------|---------|--------|--------|---------------|
| (a) **Base Frozen(default)**       | Base Weight Task 0 학습 후 고정 (freeze), downstream만 학습          | **98.29** | **97.82** | **99.31** | **54.20** | MoLE-Flow MAIN |
| (b) **Sequential Training**| Base Weight는 모든 task에서 공유하되 순차적으로 학습 (`--no_freeze_base`) | 57.47 | 55.81 | 77.38 | 7.90 | **Catastrophic Forgetting** |
| (c) **Complete Separated**| 각 task별로 base+flow 완전 독립 (multi-head, 파라미터 x15) | 55.40 | 55.65 | 77.69 | 6.22 | ⚠️ 재실험 필요 |

**실험 목적:**  
- Base backbone의 동결, 순차 학습, 완전 독립 세팅 간 성능/일반화/forgetting trade-off 비교
- 실제 deployment scenario에 맞는 가중치 공유 전략 도출



### 4.2 Task-Separated vs Shared (Upper/Lower Bound)

| Design               | Img AUC | Img AP | Pix AUC | Pix AP | Parameters |
|----------------------|---------|--------|---------|--------|------------|
| **MoLE-Flow (MAIN)** | **98.29** | **99.31** | **97.82** | **54.20** | 1.0x       |
| Sequential Training  | 57.47   | 77.38  | 55.81   | 7.90   | 1.0x       |
| Complete Separated   | 55.40   | 77.69  | 55.65   | 6.22   | 15.0x (⚠️) |

### 4.3 분석

**Sequential Training 결과 (Catastrophic Forgetting)**:
- Base NF를 freeze하지 않고 모든 task에서 순차적으로 학습한 결과
- Img AUC 57.47%로 MAIN 대비 **-40.82%p** 하락
- Pix AP 7.90%로 MAIN 대비 **-46.30%p** 하락
- **결론**: Base NF freeze 없이 순차 학습 시 심각한 catastrophic forgetting 발생

**Complete Separated 결과**:
- ⚠️ 실험 설정 오류로 예상과 다른 결과 (재실험 필요)
- 원래 의도: 각 task별 완전 독립 NF로 upper bound 측정
- 실제: Sequential Training과 유사한 설정으로 실행됨

**핵심 인사이트**:
- MoLE-Flow의 "Task 0 base freeze + LoRA adaptation" 전략이 catastrophic forgetting 방지에 핵심
- Base NF weights를 Task 0 이후 freeze하는 것이 continual learning 성능의 핵심 요소

# Hyperparameter Analysis 

## lora_rank
> 기준: lr=3e-4, logdet=1e-4, scale_k=5

| lora_rank | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-----------|---------|--------|---------|--------|------|
| 16        | TBD     | TBD    | TBD     | TBD    | 파라미터 최소 |
| 32        | TBD     | TBD    | TBD     | TBD    | 균형 |
| **64**    | **98.30** | **97.10** | **97.83** | **54.04** | **MAIN 기준** |
| 128       | 98.36   | 97.12  | 97.80   | 52.42  | 80ep, DIA5, C10 실험 결과 |

## lambda_logdet
> 기준: lr=3e-4, scale_k=5

| lambda_logdet | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|---------------|---------|--------|---------|--------|------|
| 5e-4          | 97.70   | 97.00  | 97.55   | 52.35  | 과도한 정규화 |
| 2e-4          | 98.19   | 97.09  | 97.79   | 54.18  | Pix AP 최고 |
| **1e-4**      | **98.29** | **97.10** | **97.82** | **54.20** | **MAIN 기준, 권장값** |
| 1e-5          | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |
| 0             | 98.29   | 97.03  | 97.66   | 51.06  | Ablation-Core 실험 |

## scale_context_kernel
> 기준: lr=3e-4, logdet=1e-4

| scale_context_kernel | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|---------------------|---------|--------|---------|--------|------|
| 3                   | 98.36   | 97.00  | 97.71   | 52.16  | 기본값 |
| **5**               | **98.36** | **97.12** | **97.80** | **52.42** | **MAIN 기준, 권장값** |
| 7                   | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |

## spatial_context_kernel
> 기준: lr=3e-4, logdet=1e-4, scale_k=5
> Note: 모든 실험이 spatial_k=3으로 고정되어 MAIN 기반 비교 불가

| spatial_context_kernel | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-----------------------|---------|--------|---------|--------|------|
| **3**                 | **98.29** | **97.10** | **97.82** | **54.20** | **MAIN 기준** |
| 5                     | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |
| 7                     | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |

## Tail Aware Loss weight (tail_weight)
> 기준: lr=3e-4, logdet=1e-4, scale_k=5, topk=3

| tail_weight | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-------------|---------|--------|---------|--------|------|
| 0 (off)     | 96.62   | 96.10  | 97.20   | 45.86  | Ablation-Core 실험 (**-8.34 Pix AP**) |
| 0.3         | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |
| 0.5         | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |
| 0.65        | 98.24   | 97.07  | 97.81   | 53.95  | topk=3, tail_topk=3 |
| **0.7**     | **98.29** | **97.10** | **97.82** | **54.20** | **MAIN 기준, 권장값** |
| 1.0         | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |

## Image Anomaly Score Aggregation K (score_aggregation_top_k)
> 기준: lr=3e-4, logdet=1e-4, scale_k=5, tw=0.7

| top_k | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-------|---------|--------|---------|--------|------|
| 1     | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |
| **3** | **98.29** | **97.10** | **97.82** | **54.20** | **MAIN 기준, 권장값** |
| 5     | 98.30   | 97.09  | 97.83   | 54.04  | tail_topk=3 |
| 7     | TBD     | TBD    | TBD     | TBD    | MAIN 기반 실험 필요 |

