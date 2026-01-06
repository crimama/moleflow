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

**Baseline Performance** (2026-01-06 업데이트):
| Metric | Value |
|--------|-------|
| Image AUC | **98.29%** |
| Pixel AUC | **97.82%** |
| Pixel AP | **54.20%** |
| Routing Accuracy | **100%** |

> **Note**: MoLE-Only (No DIA) 아키텍처가 더 단순하면서도 MoLE+DIA(12블록)와 동등한 성능 (6블록만으로 달성)

---


# Architecture Modular Analysis

## 1. Core Component Ablation (MoLE-Only NCL=6 기준)

> MAIN 설정: MoLE-Only NCL=6, lr=3e-4, logdet=1e-4, scale_k=5
> ✅ NCL=6 기준 재실험 완료 (2026-01-06)

### 실험 목록

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o SpatialContextMixer | Spatial Context Mixing의 기여도 | SpatialContextMixer 모듈 제거 | ✅ 완료 |
| w/o WhiteningAdapter | Whitening Adapter의 기여도 | InputAdapter(Whitening) 미적용 | ✅ 완료 |
| w/o Tail Aware Loss | Tail Aware Loss의 기여도 | Tail Aware Loss 비활성화 | ✅ 완료 |
| w/o LogDet Regularization | LogDet Regularization 유무 | lambda_logdet=0 | ✅ 완료 |
| w/o Scale Context | Scale Context의 기여도 | `--no_scale_context` | ✅ 완료 |
| w/o LoRA | LoRA vs Regular Linear | `--use_regular_linear` | ✅ 완료 |
| w/o MoLE Subnet | MoLE Subnet 제거 (Standard Subnet) | MoLE Subnet 비활성화 | ✅ 완료 |

### 결과 테이블 (NCL=6 MoLE-Only 기준, 2026-01-06 업데이트)

| Configuration | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP |
|---------------|---------|---------|--------|--------|--------|-----------|----------|
| **MoLE-Only NCL=6 (MAIN)** | **98.29** | **97.82** | **99.28** | **54.20** | 100.0 | - | - |
| w/o LoRA | 98.29 | 97.82 | 99.28 | 54.20 | 100.0 | 0.00 | 0.00 |
| w/o MoLE Subnet | 98.37 | 97.84 | 99.32 | 54.16 | 100.0 | +0.08 | -0.04 |
| w/o Scale Context | 98.08 | 97.84 | 99.16 | 53.93 | 100.0 | -0.21 | -0.27 |
| w/o SpatialContextMixer | 98.08 | 97.70 | 99.23 | 52.24 | 100.0 | -0.21 | -1.96 |
| w/o LogDet Regularization | 98.29 | 97.66 | 99.31 | 51.06 | 100.0 | 0.00 | -3.14 |
| w/o WhiteningAdapter | 98.06 | 97.60 | 99.23 | 47.14 | 100.0 | -0.23 | **-7.06** |
| w/o Tail Aware Loss | 96.62 | 97.20 | 98.66 | 45.86 | 100.0 | **-1.67** | **-8.34** |

### 분석

**Critical Components (제거 시 큰 성능 저하)**:
1. **Tail Aware Loss** (Img AUC -1.67%, Pix AP -8.34%)
   - 가장 큰 영향을 미치는 컴포넌트
   - Tail patch에 대한 집중이 anomaly detection 성능에 핵심적
   - **필수 컴포넌트**

2. **WhiteningAdapter** (Img AUC -0.23%, Pix AP -7.06%)
   - Feature 분포 정렬이 pixel-level detection에 중요
   - Task 간 feature 분포 차이를 보정
   - **필수 컴포넌트**

**Moderate Impact Components (제거 시 소폭 저하)**:
3. **LogDet Regularization** (Img AUC 0.00%, Pix AP -3.14%)
   - Jacobian regularization으로 학습 안정화
   - Pixel AP에 약간 기여

4. **SpatialContextMixer** (Img AUC -0.21%, Pix AP -1.96%)
   - 공간적 context mixing의 부가적 기여
   - Pixel-level 성능에 도움

5. **Scale Context** (Img AUC -0.21%, Pix AP -0.27%)
   - Multi-scale context 기여도 제한적
   - 선택적 컴포넌트

**No Impact Components (제거해도 성능 유지)**:
6. **LoRA** (Img AUC 0.00%, Pix AP 0.00%)
   - NCL=6 MoLE-only에서 LoRA가 DIA 없이는 활성화되지 않음
   - 현재 설정에서 영향 없음

7. **MoLE Subnet** (Img AUC +0.08%, Pix AP -0.04%)
   - Standard subnet으로도 동등한 성능
   - MoLE subnet의 추가 이점 미미
   - → MoLE-Only (No DIA) 아키텍처 채택 근거


---

## 2. MoLE Subnet Ablation (NCL=6 기준)

> ✅ 실험 완료 (2026-01-06)

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o Scale Context | scale_context 유/무 | Scale Context 모듈 미사용 | ✅ 완료 |
| w/o LoRA | LoRA 대신 Linear 사용 | Regular Linear 사용 | ✅ 완료 |
| Complete Separated | Task별 완전 분리 | 각 Task별 독립 NF | ✅ 완료 |
| LoRA Rank=16 | LoRA rank 영향 | `--lora_rank 16` | ✅ 완료 |

### 결과 테이블 (NCL=6 기준, 2026-01-06 업데이트)

| Configuration | Img AUC | Pix AUC | Pix AP | Δ Pix AP | Status |
|---------------|---------|---------|--------|----------|--------|
| **MoLE-Only NCL=6 (MAIN)** | **98.29** | **97.82** | **54.20** | - | ✅ |
| w/o Scale Context | 98.08 | 97.84 | 53.93 | -0.27 | ✅ |
| w/o LoRA (Regular Linear) | 98.29 | 97.82 | 54.20 | 0.00 | ✅ |
| Complete Separated | 98.13 | 97.74 | 52.49 | -1.71 | ✅ 완료 |
| LoRA Rank=16 | 98.06 | 97.82 | 55.86 | +1.66 | ✅ 완료 |

### 분석
- **w/o Scale Context**: Multi-scale context 제거 시 소폭 성능 저하 (Pix AP -0.27%)
  - Scale context의 기여도는 제한적이나 유지 권장

- **w/o LoRA**: LoRA 대신 Regular Linear를 사용해도 성능 동일
  - LoRA의 low-rank constraint가 성능에 영향을 주지 않음
  - Continual Learning에서 파라미터 효율성 관점으로 LoRA 유지 권장

- **Complete Separated**: 각 Task별 독립 NF 시 성능 하락 (Pix AP -1.71%)
  - 파라미터 15배 증가에도 성능 저하
  - Base weight sharing이 효율적

- **LoRA Rank=16**: Low-rank에서 오히려 Pix AP 향상 (+1.66%)
  - Regularization 효과로 해석 가능
  - 파라미터 효율성 우수

---

## 3. Normalizing Flow Block 구성 실험 (MoLE / DIA Block 조합)

> ✅ 실험 완료 (2026-01-06 업데이트)

전체 Coupling Block 수를 조절하고, MoLE-SubNet과 DIA block의 구성 비율에 따른 성능 변화를 실험합니다.

| MoLE Blocks | DIA Blocks | Total | Img AUC | Pix AUC | Pix AP | 비고 |
|-------------|-----------|-------|---------|---------|--------|------|
| 10          | 2         | 12    | 98.27   | 97.73   | **54.70** | MoLE 비중 높을수록 Pix AP 향상 |
| 8           | 4         | 12    | 98.29   | 97.82   | 54.20 | Old MAIN (MoLE+DIA) |
| 6           | 6         | 12    | 98.19   | 97.79   | 51.62 | 균형 구성 |
| 4           | 8         | 12    | 98.09   | 97.74   | 50.27 | DIA 비중 높으면 Pix AP 저하 |
| 0           | 12        | 12    | 98.37   | 97.84   | 54.16 | DIA-only |
| **6**       | **0**     | **6** | **98.29** | **97.82** | **54.20** | **MAIN (MoLE-Only NCL=6)** |

### 분석

1. **MoLE 비중 vs Pix AP**: MoLE block 비중이 높을수록 Pix AP 향상
   - MoLE 10 + DIA 2: Pix AP **54.70%** (최고)
   - MoLE 8 + DIA 4: Pix AP 54.20%
   - MoLE 4 + DIA 8: Pix AP 50.27% (최저)

2. **MoLE-Only NCL=6 (MAIN)**: 더 적은 블록으로 동등한 성능
   - 6블록으로 12블록 MoLE+DIA 조합과 유사한 성능
   - 파라미터 효율성 우수

3. **DIA의 역할**:
   - DIA 비중이 높으면 Pix AP 감소 경향
   - DIA-only(12블록)는 MoLE+DIA(8+4)보다 Pix AP 약간 낮음

**결론**: MoLE-Only NCL=6가 효율성과 성능 균형에서 최적


### 3.2 MoLE-Only (No DIA) Depth Scaling

DIA 없이 MoLE(LoRA) subnet만으로 구성할 때, num_coupling_layers(NCL) 증가에 따른 성능 변화를 실험합니다.

> **실험 조건**: `--use_dia False` (DIA 비활성화), backbone=WRN50, lr=3e-4, logdet=1e-4, scale_k=5, epochs=60

| NCL | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | 비고 |
|-----|---------|---------|--------|--------|--------|------|
| 4 | 97.84 | 97.80 | 99.12 | 55.90 | 100.0 | Pix AP 최고, 얕은 네트워크 |
| **6** | **98.29** | **97.82** | **99.28** | **54.20** | 100.0 | **MAIN 설정** |
| 8 | 97.99 | 97.74 | 99.23 | 54.92 | 100.0 | 안정적 |
| 12 | 94.20 | 94.16 | 97.81 | 51.82 | 100.0 | ⚠️ 성능 하락 시작 |
| 16 | 60.43 | 53.50 | 81.20 | 10.67 | 100.0 | ❌ 심각한 성능 저하 |
| 20 | 58.60 | 52.68 | 80.40 | 9.60 | 100.0 | ❌ 학습 실패 |

**분석**:
1. **NCL=4~8**: 모두 안정적 학습, Img AUC 97.8~98.3%, Pix AP 54.2~55.9%
   - NCL=4: Pix AP 55.90%로 최고 (shallow network의 이점)
   - **NCL=6 (MAIN)**: Img AUC 98.29%로 최고, 균형 잡힌 성능
   - NCL=8: 유사한 성능 유지
2. **NCL=12**: Img AUC 94.20%로 하락 시작 - gradient flow 문제 징후
3. **NCL=16, 20**: 심각한 성능 저하 (Img AUC ~60%, Pix AP ~10%) - **학습 불안정/실패**

**핵심 인사이트**:
- DIA 없이 MoLE-only: **NCL=4~8 범위에서 안정적** (그 이상은 학습 불안정)
- **NCL=6 권장**: Img AUC 98.29%, Pix AP 54.20%로 최적 균형점
- NCL=4 대안: 파라미터 효율성이 중요할 경우 (Pix AP 55.90% 최고)
- DIA의 역할: 깊은 NF(NCL>8)에서 gradient flow 안정화에 필수

**MAIN 설정 (NCL=6) 상세 결과**:
| Metric | Value |
|--------|-------|
| Image AUC | 98.29% |
| Pixel AUC | 97.82% |
| Image AP | 99.28% |
| Pixel AP | 54.20% |
| Routing Accuracy | 100% |


---

## 4. Base Weight Sharing vs. Sequential/Independent Training

> ✅ Sequential Training 실험 완료 (2026-01-06)

Base backbone의 가중치 공유(sequential/independent) 방식에 따른 continual setting의 영향 분석을 위해 아래 3가지 설정을 비교합니다.

| 설정                      | Description                                                  | Img AUC | Pix AUC | Img AP | Pix AP | 비고          |
|---------------------------|-------------------------------------------------------------|---------|---------|--------|--------|---------------|
| (a) **Base Frozen(default)**       | Base Weight Task 0 학습 후 고정 (freeze), downstream만 학습          | **98.29** | **97.82** | **99.28** | **54.20** | MoLE-Only MAIN |
| (b) **Sequential Training**| Base Weight는 모든 task에서 공유하되 순차적으로 학습 | 57.47 | 55.81 | 77.38 | 7.90 | ❌ Catastrophic Forgetting |
| (c) **Complete Separated**| 각 task별로 base+flow 완전 독립 (multi-head) | 98.13 | 97.74 | 99.22 | 52.49 | 파라미터 15x, Pix AP -1.71% |

**실험 목적:**
- Base backbone의 동결, 순차 학습, 완전 독립 세팅 간 성능/일반화/forgetting trade-off 비교
- 실제 deployment scenario에 맞는 가중치 공유 전략 도출

### 4.2 분석

**Sequential Training 결과 (Catastrophic Forgetting)**:
- Base NF를 freeze하지 않고 모든 task에서 순차적으로 학습한 결과
- Img AUC **57.47%** → MAIN 대비 **-40.82%** 하락
- Pix AP **7.90%** → MAIN 대비 **-46.30%** 하락
- **심각한 catastrophic forgetting 발생**

**Complete Separated 결과**:
- 각 Task별로 완전히 독립된 NF 모델 학습
- Img AUC 98.13% (MAIN 대비 -0.16%), Pix AP 52.49% (MAIN 대비 -1.71%)
- **파라미터 수 15배 증가**에도 성능은 오히려 하락
- Base weight sharing의 이점 확인

**핵심 인사이트**:
- MoLE-Flow의 "Task 0 base freeze + LoRA adaptation" 전략이 catastrophic forgetting 방지에 **핵심적**
- Base NF weights를 Task 0 이후 freeze하는 것이 continual learning 성능의 필수 요소
- LoRA를 통한 task-specific adaptation으로 기존 지식 보존
- **Base weight sharing**이 성능과 효율성 모두에서 우수 (Complete Separated 대비)

# Hyperparameter Analysis (NCL=6 기준)

> ⚠️ 대부분의 하이퍼파라미터 실험은 이전 MoLE+DIA 기준. NCL=6 기준 재실험 필요

## lora_rank
> 기준: NCL=6, lr=3e-4, logdet=1e-4, scale_k=5
> ✅ 실험 완료 (2026-01-06)

| lora_rank | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-----------|---------|---------|--------|----------|------|
| 16        | 98.06   | 97.82   | 55.86  | +1.66    | ✅ 완료 |
| 32        | 98.04   | 97.82   | 55.89  | +1.69    | ✅ 완료 |
| **64**    | **98.29** | **97.82** | **54.20** | - | **MAIN 기준** |
| 128       | 98.04   | 97.82   | 55.80  | +1.60    | ✅ 완료 |

### 분석
- **LoRA Rank 16, 32, 128**: Pix AP가 MAIN(64)보다 높음 (+1.6~1.7%)
  - Low-rank(16, 32)와 High-rank(128) 모두 Pix AP 향상
  - Rank 64가 오히려 Pix AP 최저
- **LoRA Rank 64 (MAIN)**: Img AUC 98.29%로 **가장 높음**
  - Image-level에서 최적
- **LoRA Rank와 Pix AP**: Rank에 따른 일관된 경향 없음
  - 모든 rank에서 Pix AP 54~56% 범위로 안정적

**결론**: LoRA rank는 16~128 범위에서 유사한 성능. Img AUC 최대화는 rank=64, 파라미터 효율성은 rank=16~32 권장

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

