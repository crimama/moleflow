# MoLE-Flow Ablation Study

## Baseline Configuration (MAIN)

**Experiment**: `MVTec-NoDIA-NCL6` (MoLE-Only, No DIA)

| Parameter | Value |
|-----------|-------|
| Backbone | WideResNet50 |
| LoRA Rank | 64 |
| Coupling Layers | **6** |
| DIA Blocks | **0 (disabled)** |
| Epochs | 60 |
| Learning Rate | 3e-4 |
| Adapter Mode | Whitening |
| Tail Weight | 0.7 |
| Score Aggregation | top_k (k=3) |
| Tail Top-K Ratio | 0.02 |
| Scale Context Kernel | 5 |
| Lambda Logdet | 1e-4 |

**Baseline Performance**:
| Metric | Value |
|--------|-------|
| Image AUC | **98.05%** |
| Pixel AUC | **97.81%** |
| Pixel AP | **55.80%** |
| Routing Accuracy | **100%** |

> **Note**: MoLE-Only (No DIA) 아키텍처가 더 단순하면서도 Pix AP가 더 높음 (55.80% vs 54.20%)

---


# Architecture Modular Analysis

## 1. Core Component Ablation (MoLE-Only NCL=6 기준)

> MAIN 설정: MoLE-Only NCL=6, lr=3e-4, logdet=1e-4, scale_k=5
> ⚠️ 아래 실험들은 이전 MAIN(MoLE+DIA) 기준 결과. NCL=6 기준 재실험 예정

### 실험 목록

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o SpatialContextMixer | Spatial Context Mixing의 기여도 | SpatialContextMixer 모듈 제거 | 🔄 NCL6 재실험 필요 |
| w/o WhiteningAdapter | Whitening Adapter의 기여도 | InputAdapter(Whitening) 미적용 | 🔄 NCL6 재실험 필요 |
| w/o Tail Aware Loss | Tail Aware Loss의 기여도 | Tail Aware Loss 비활성화 | 🔄 NCL6 재실험 필요 |
| w/o LogDet Regularization | LogDet Regularization 유무 | lambda_logdet=0 | 🔄 NCL6 재실험 필요 |
| w/o Scale Context | Scale Context의 기여도 | `--no_scale_context` | 🔄 실험 중 |
| w/o LoRA | LoRA vs Regular Linear | `--use_regular_linear` | 🔄 실험 중 |

### 결과 테이블 (이전 MoLE+DIA 기준, 참고용)

| Configuration | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP |
|---------------|---------|---------|--------|--------|--------|-----------|----------|
| **MoLE+DIA (Old MAIN)** | 98.29 | 97.82 | 99.31 | 54.20 | 100.0 | - | - |
| w/o SpatialContextMixer | 98.08 | 97.70 | 99.23 | 52.24 | 100.0 | -0.21 | -1.96 |
| w/o WhiteningAdapter | 98.06 | 97.60 | 99.23 | 47.14 | 100.0 | -0.23 | **-7.06** |
| w/o Tail Aware Loss | 96.62 | 97.20 | 98.66 | 45.86 | 100.0 | **-1.67** | **-8.34** |
| w/o LogDet Regularization | 98.29 | 97.66 | 99.31 | 51.06 | 100.0 | 0.00 | -3.14 |
| w/o MoLE subnet (DIA only) | 98.37 | 97.84 | 99.32 | 54.16 | 100.0 | +0.08 | -0.04 |

### 분석 (이전 결과 기반)

1. **Tail Aware Loss**가 가장 큰 영향 (Img AUC -1.67%, Pix AP -8.34%)
   - 손실 함수에서 tail patch에 대한 집중이 성능에 핵심적

2. **WhiteningAdapter** 제거 시 Pix AP -7.06% 감소
   - 분포 정렬이 pixel-level anomaly detection에 중요

3. **LogDet Regularization** 영향 미미 (Img AUC 동일, Pix AP -3.14%)
   - 안정화 효과는 있으나 성능에 큰 영향 없음

4. **SpatialContextMixer** 제거 시 소폭 감소 (-0.21%, -1.96%)
   - 공간적 context mixing의 부가적 기여 확인

5. **MoLE subnet 제거 (DIA only)** 시 성능 유지 (+0.08%, -0.04%)
   - → 이 결과를 기반으로 MoLE-Only (No DIA) 아키텍처 채택


---

## 2. MoLE Subnet Ablation (NCL=6 기준, 실험 중)

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o Scale Context | scale_context 유/무 | Scale Context 모듈 미사용 (`--no_scale_context`) | 🔄 실험 중 (GPU 0) |
| w/o LoRA | LoRA 대신 Linear 사용 | LoRA 대신 Regular Linear 사용 (`--use_regular_linear`) | 🔄 실험 중 (GPU 1) |
| Complete Separated | Task별 완전 분리 | 각 Task별 독립 NF (`--use_task_separated`) | 🔄 실험 중 (GPU 4) |
| LoRA Rank=16 | LoRA rank 영향 | `--lora_rank 16` | 🔄 실험 중 (GPU 5) |

### 결과 테이블 (NCL=6 기준, 업데이트 예정)

| Configuration | Img AUC | Pix AUC | Pix AP | Δ Pix AP | Status |
|---------------|---------|---------|--------|----------|--------|
| **MoLE-Only NCL=6 (MAIN)** | **98.05** | **97.81** | **55.80** | - | ✅ |
| w/o Scale Context | TBD | TBD | TBD | TBD | 🔄 실험 중 |
| w/o LoRA (Regular Linear) | TBD | TBD | TBD | TBD | 🔄 실험 중 |
| Complete Separated | TBD | TBD | TBD | TBD | 🔄 실험 중 |
| LoRA Rank=16 | TBD | TBD | TBD | TBD | 🔄 실험 중 |

### 이전 결과 (MoLE+DIA 기준, 참고용)
| Configuration | Img AUC | Pix AUC | Pix AP | Δ Pix AP |
|---------------|---------|---------|--------|----------|
| MoLE+DIA (Old MAIN) | 98.29 | 97.82 | 54.20 | - |
| w/o Scale Context | 98.08 | 97.84 | 53.93 | -0.27 |
| w/o LoRA (Regular Linear) | 98.29 | 97.82 | 54.20 | 0.00 |

### 분석 (이전 결과 기반)
- **w/o LoRA**: LoRA 대신 Regular Linear를 사용해도 성능 동일
  - LoRA의 low-rank constraint가 성능에 영향을 주지 않음
  - Continual Learning에서 파라미터 효율성 관점으로 LoRA 유지 권장

---

## 3. Normalizing Flow Block 구성 실험 (MoLE / DIA Block 조합)

> ⚠️ 이 실험은 이전 MoLE+DIA 아키텍처 기준. 현재 MAIN은 MoLE-Only (No DIA)

전체 Coupling Block 수(=8)는 동일하게 고정하고, MoLE-SubNet과 DIA block의 구성 비율에 따라 성능이 어떻게 달라지는지 실험합니다.

| MoLE Blocks | DIA Blocks | Img AUC | Pix AUC | Img AP | Pix AP | 비고 |
|-------------|-----------|---------|---------|--------|--------|------|
| 8           | 4         | 98.29 | 97.82 | 99.31  | 54.20 | Old MAIN (MoLE+DIA) |
| 10          | 2         | 98.27   | 97.73   | 99.31  | 54.70 | 총 12블록 |
| 6           | 6         | 98.19   | 97.79   | 99.16  | 51.62  | 총 12블록 |
| 4           | 8         | 98.09   | 97.74   | 99.14  | 50.27  | 총 12블록 |
| 0           | 12        | 98.37   | 97.84   | 99.32  | 54.16  | DIA-only |
| **6**       | **0**     | **98.05** | **97.81** | **99.25** | **55.80** | **NEW MAIN (MoLE-Only)** |

### 분석

1. **MoLE-Only NCL=6 (NEW MAIN)**: Pix AP **55.80%**로 최고 성능
   - DIA 없이 더 단순한 아키텍처로 더 높은 성능 달성
2. **MoLE+DIA 조합**: DIA 비중이 높을수록 Pix AP 감소 경향
3. **DIA only**: Pix AP 54.16%로 양호하나 MoLE-Only보다 낮음

**결론**: DIA를 제거하고 MoLE-Only로 단순화하면 오히려 성능 향상 (Pix AP 54.20% → 55.80%)


### 3.2 MoLE-Only (No DIA) Depth Scaling

DIA 없이 MoLE(LoRA) subnet만으로 구성할 때, num_coupling_layers(NCL) 증가에 따른 성능 변화를 실험합니다.

> **실험 조건**: `--use_dia False` (DIA 비활성화), backbone=WRN50, lr=3e-4, logdet=1e-4, scale_k=5, epochs=60

| NCL | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | 비고 |
|-----|---------|---------|--------|--------|--------|------|
| **4** | **97.84** | **97.80** | **99.12** | **55.90** | 100.0 | **Pix AP 최고**, 얕은 네트워크 |
| 6 | 98.05 | 97.81 | 99.25 | 55.80 | 100.0 | 양호, MAIN 설정 |
| 8 | 97.99 | 97.74 | 99.23 | 54.92 | 100.0 | 안정적 |
| 12 | 94.20 | 94.16 | 97.81 | 51.82 | 100.0 | ⚠️ 성능 하락 시작 |
| 16 | 60.43 | 53.50 | 81.20 | 10.67 | 100.0 | ❌ 심각한 성능 저하 |
| 20 | 58.60 | 52.68 | 80.40 | 9.60 | 100.0 | ❌ 학습 실패 |

**분석**:
1. **NCL=4~8**: 모두 안정적 학습, Img AUC 97.8~98.0%, Pix AP 54.9~55.9%
   - **NCL=4가 Pix AP 55.90%로 가장 높음** (shallow network의 이점)
   - NCL=6: Img AUC 98.05%로 가장 높음, Pix AP 55.80%로 균형 잡힌 성능
   - NCL=8도 유사한 성능 유지
2. **NCL=12**: Img AUC 94.20%로 하락 시작 - gradient flow 문제 징후
3. **NCL=16, 20**: 심각한 성능 저하 (Img AUC ~60%, Pix AP ~10%) - **학습 불안정/실패**

**핵심 인사이트**:
- DIA 없이 MoLE-only: **NCL=4~8 범위에서 안정적** (그 이상은 학습 불안정)
- **NCL=6 권장**: Img AUC 98.05%, Pix AP 55.80%로 최적 균형점
- **NCL=4 대안**: 파라미터 효율성이 중요할 경우 (Pix AP 55.90% 최고)
- DIA의 역할: 깊은 NF(NCL>8)에서 gradient flow 안정화에 필수
- MoLE-Flow(Full)의 MoLE 8 + DIA 4 조합이 **깊이 확장과 안정성**을 동시 달성

**MAIN 설정 (NCL=6) 상세 결과**:
| Metric | Value |
|--------|-------|
| Image AUC | 98.05% |
| Pixel AUC | 97.81% |
| Image AP | 99.25% |
| Pixel AP | 55.80% |
| Routing Accuracy | 100% |


---

## 4. Base Weight Sharing vs. Sequential/Independent Training

> ⚠️ 이 실험들은 이전 MoLE+DIA 기준. NCL=6 기준 재실험 진행 중 (GPU 4)

Base backbone의 가중치 공유(sequential/independent) 방식에 따른 continual setting의 영향 분석을 위해 아래 3가지 설정을 비교합니다.

| 설정                      | Description                                                  | Img AUC | Pix AUC | Img AP | Pix AP | 비고          |
|---------------------------|-------------------------------------------------------------|---------|---------|--------|--------|---------------|
| (a) **Base Frozen(default)**       | Base Weight Task 0 학습 후 고정 (freeze), downstream만 학습          | **98.05** | **97.81** | **99.25** | **55.80** | MoLE-Only MAIN |
| (b) **Sequential Training**| Base Weight는 모든 task에서 공유하되 순차적으로 학습 | TBD | TBD | TBD | TBD | 🔄 NCL6 재실험 필요 |
| (c) **Complete Separated**| 각 task별로 base+flow 완전 독립 (multi-head) | TBD | TBD | TBD | TBD | 🔄 실험 중 (GPU 4) |

**실험 목적:**
- Base backbone의 동결, 순차 학습, 완전 독립 세팅 간 성능/일반화/forgetting trade-off 비교
- 실제 deployment scenario에 맞는 가중치 공유 전략 도출

### 4.2 이전 결과 (MoLE+DIA 기준, 참고용)

| Design               | Img AUC | Img AP | Pix AUC | Pix AP | Parameters |
|----------------------|---------|--------|---------|--------|------------|
| MoLE+DIA (Old MAIN)  | 98.29   | 99.31  | 97.82   | 54.20  | 1.0x       |
| Sequential Training  | 57.47   | 77.38  | 55.81   | 7.90   | 1.0x       |
| Complete Separated   | 55.40   | 77.69  | 55.65   | 6.22   | 15.0x (⚠️) |

### 4.3 분석 (이전 결과 기반)

**Sequential Training 결과 (Catastrophic Forgetting)**:
- Base NF를 freeze하지 않고 모든 task에서 순차적으로 학습한 결과
- Img AUC 57.47%로 심각한 catastrophic forgetting 발생
- **결론**: Base NF freeze 없이 순차 학습 시 심각한 catastrophic forgetting 발생

**핵심 인사이트**:
- MoLE-Flow의 "Task 0 base freeze + LoRA adaptation" 전략이 catastrophic forgetting 방지에 핵심
- Base NF weights를 Task 0 이후 freeze하는 것이 continual learning 성능의 핵심 요소

# Hyperparameter Analysis (NCL=6 기준)

> ⚠️ 대부분의 하이퍼파라미터 실험은 이전 MoLE+DIA 기준. NCL=6 기준 재실험 필요

## lora_rank
> 기준: NCL=6, lr=3e-4, logdet=1e-4, scale_k=5

| lora_rank | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-----------|---------|--------|---------|--------|------|
| 16        | TBD     | TBD    | TBD     | TBD    | 🔄 실험 중 (GPU 5) |
| 32        | TBD     | TBD    | TBD     | TBD    | GPU 5 순차 실험 예정 |
| **64**    | **98.05** | **99.25** | **97.81** | **55.80** | **MAIN 기준** |
| 128       | TBD     | TBD    | TBD     | TBD    | GPU 5 순차 실험 예정 |

### 이전 결과 (MoLE+DIA 기준, 참고용)
| lora_rank | Img AUC | Pix AUC | Pix AP | 비고 |
|-----------|---------|---------|--------|------|
| 64        | 98.30   | 97.83   | 54.04  | Old MAIN |
| 128       | 98.36   | 97.80   | 52.42  | 80ep, DIA5 |

## lambda_logdet
> 기준: NCL=6, lr=3e-4, scale_k=5

| lambda_logdet | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|---------------|---------|--------|---------|--------|------|
| **1e-4**      | **98.05** | **99.25** | **97.81** | **55.80** | **MAIN 기준** |
| 기타          | TBD     | TBD    | TBD     | TBD    | NCL6 재실험 필요 |

## scale_context_kernel
> 기준: NCL=6, lr=3e-4, logdet=1e-4

| scale_context_kernel | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|---------------------|---------|--------|---------|--------|------|
| **5**               | **98.05** | **99.25** | **97.81** | **55.80** | **MAIN 기준** |
| 0 (disabled)        | TBD     | TBD    | TBD     | TBD    | 🔄 실험 중 (GPU 0) |

## spatial_context_kernel
> 기준: NCL=6, lr=3e-4, logdet=1e-4, scale_k=5

| spatial_context_kernel | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-----------------------|---------|--------|---------|--------|------|
| **3**                 | **98.05** | **99.25** | **97.81** | **55.80** | **MAIN 기준** |

## Tail Aware Loss weight (tail_weight)
> 기준: NCL=6, lr=3e-4, logdet=1e-4, scale_k=5, topk=3

| tail_weight | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-------------|---------|--------|---------|--------|------|
| **0.7**     | **98.05** | **99.25** | **97.81** | **55.80** | **MAIN 기준** |
| 기타        | TBD     | TBD    | TBD     | TBD    | NCL6 재실험 필요 |

## Image Anomaly Score Aggregation K (score_aggregation_top_k)
> 기준: NCL=6, lr=3e-4, logdet=1e-4, scale_k=5, tw=0.7

| top_k | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-------|---------|--------|---------|--------|------|
| **3** | **98.05** | **99.25** | **97.81** | **55.80** | **MAIN 기준** |
| 기타  | TBD     | TBD    | TBD     | TBD    | NCL6 재실험 필요 |

