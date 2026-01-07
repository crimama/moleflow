# MoLE-Flow Ablation Study

## Baseline Configuration (MAIN)

**Experiment**: `MVTec-MoLE6-DIA2`

| Parameter | Value |
|-----------|-------|
| Backbone | WideResNet50 |
| LoRA Rank | 64 |
| Coupling Layers (MoLE) | **6** |
| DIA Blocks | **2** |
| Total Blocks | 8 |
| Epochs | 60 |
| Learning Rate | 3e-4 |
| Adapter Mode | Whitening |
| Tail Weight | 0.7 |
| Score Aggregation | top_k (k=3) |
| Tail Top-K Ratio | 0.02 |
| Scale Context Kernel | 5 |
| Lambda Logdet | 1e-4 |

**Baseline Performance** (MVTec AD, 15 classes, 1x1 CL):
| Metric | Value |
|--------|-------|
| Image AUC | **98.05%** |
| Pixel AUC | **97.81%** |
| Pixel AP | **55.80%** |
| Routing Accuracy | **100%** |

> **✅ Note**: MoLE6+DIA2가 최고 Pixel AP (55.80%)를 달성. DIA가 학습 안정화에 기여하고, MoLE가 Task-specific adaptation으로 Pixel-level 정밀도 향상.

---


# Architecture Modular Analysis

## 1. Core Component Ablation (MoLE6+DIA2 기준)

> **MAIN**: MoLE6+DIA2 (NCL=6, DIA=2) 기준으로 각 컴포넌트 제거 시 성능 변화 측정

### 실험 목록

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| w/o LoRA | LoRA vs Regular Linear | `--no_lora` | ⚠️ 재실험 필요 |
| w/o Scale Context | Scale Context의 기여도 | `--no_scale_context` | ⚠️ 재실험 필요 |
| w/o Spatial Context | Spatial Context의 기여도 | `--no_spatial_context` | ⚠️ 재실험 필요 |
| w/o Whitening Adapter | Whitening Adapter의 기여도 | `--no_whitening_adapter` | ⚠️ 재실험 필요 |
| w/o Tail Aware Loss | Tail Aware Loss의 기여도 | `--tail_weight 0` | ⚠️ 재실험 필요 |
| w/o LogDet Reg | LogDet Regularization 유무 | `--lambda_logdet 0` | ⚠️ 재실험 필요 |
| w/o Pos Embedding | 위치 정보의 기여도 | `--no_pos_embedding` | ⚠️ 재실험 필요 |
| w/o DIA | DIA 제거 (MoLE-Only) | `--no_dia` | ✅ 완료 (Section 3.3) |

### 결과 테이블 (MoLE6+DIA2 기준)

| Configuration | Img AUC | Pix AUC | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP | Status |
|---------------|---------|---------|--------|--------|-----------|----------|--------|
| **MAIN (MoLE6+DIA2)** | **98.05** | **97.81** | **55.80** | 100.0 | - | - | ✅ |
| w/o DIA (MoLE6-Only) | 92.08 | 94.27 | 49.85 | 100.0 | -5.97 | -5.95 | ✅ |
| w/o LoRA | TBD | TBD | TBD | TBD | TBD | TBD | ⚠️ |
| w/o Scale Context | TBD | TBD | TBD | TBD | TBD | TBD | ⚠️ |
| w/o Spatial Context | TBD | TBD | TBD | TBD | TBD | TBD | ⚠️ |
| w/o Whitening Adapter | TBD | TBD | TBD | TBD | TBD | TBD | ⚠️ |
| w/o Tail Aware Loss | TBD | TBD | TBD | TBD | TBD | TBD | ⚠️ |
| w/o LogDet Reg | TBD | TBD | TBD | TBD | TBD | TBD | ⚠️ |
| w/o Pos Embedding | TBD | TBD | TBD | TBD | TBD | TBD | ⚠️ |

### 분석

**w/o DIA 분석** (완료):
- DIA 제거 시 Img AUC -5.97%p, Pix AP -5.95%p 하락
- DIA가 학습 안정화 및 성능 향상에 핵심 역할

> 나머지 ablation은 재실험 필요


---

## 2. Training Strategy Ablation (MoLE6+DIA2 기준)

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| Sequential Training | Base freeze 없이 순차 학습 | `--no_freeze_base` | ⚠️ 재실험 필요 |
| Complete Separated | Task별 완전 분리 | `--use_task_separated` | ⚠️ 재실험 필요 |

### 결과 테이블

| Configuration | Img AUC | Pix AUC | Pix AP | Δ Pix AP | Status |
|---------------|---------|---------|--------|----------|--------|
| **MAIN (Base Frozen)** | 98.05 | 97.81 | 55.80 | - | ✅ |
| Sequential Training | TBD | TBD | TBD | TBD | ⚠️ 재실험 필요 |
| Complete Separated | TBD | TBD | TBD | TBD | ⚠️ 재실험 필요 |

### 분석

> 재실험 후 업데이트 필요

---

## 3. Normalizing Flow Block 구성 실험 (MoLE / DIA Block 조합)

> ✅ 실험 완료 (2026-01-07 업데이트)

전체 Coupling Block 수를 조절하고, MoLE-SubNet과 DIA block의 구성 비율에 따른 성능 변화를 실험합니다.

### 3.1 Architecture 비교 요약

| MoLE | DIA | Total | Img AUC | Pix AUC | Pix AP | 비고 |
|------|-----|-------|---------|---------|--------|------|
| **6** | **2** | **8** | **98.05** | **97.81** | **55.80** | **✅ MAIN (추천)** |
| 10 | 2 | 12 | 98.27 | 97.73 | 54.70 | MoLE 비중 높음 |
| 8 | 4 | 12 | 98.29 | 97.82 | 54.20 | Old MAIN |
| 6 | 6 | 12 | 98.19 | 97.79 | 51.62 | 균형 구성 |
| 4 | 8 | 12 | 98.09 | 97.74 | 50.27 | DIA 비중 높음 |
| 0 | 4 | 4 | 98.13 | 97.86 | 53.28 | DIA-Only (Best) |
| 8 | 0 | 8 | 92.74 | 94.55 | 50.06 | MoLE-Only (Best) |

### 분석

1. **MoLE6+DIA2가 최고 Pix AP (55.80%)**
   - DIA의 안정화 + MoLE의 Task-specific 장점 결합
   - 8 blocks로 효율적

2. **MoLE-Only는 불안정**
   - DIA 없이는 Img AUC 92.74%로 크게 하락
   - NCL≥10에서 학습 실패

3. **DIA-Only는 안정적이나 Pix AP 제한적**
   - 모든 depth에서 98%+ 안정
   - Task-specific 학습 부재로 Pix AP 53.28%

**결론**: MoLE6+DIA2가 최고 Pix AP와 안정성 균형에서 최적


### 3.2 MoLE + DIA2 Depth Scaling (실제 실험 설정)

> ⚠️ **주의**: 이전에 "MoLE-Only (No DIA)"로 기록되었으나, 실제로는 `use_dia=true, dia_n_blocks=2`로 DIA가 활성화된 상태였음

MoLE(NCL) + DIA(2 blocks) 구성에서 NCL 증가에 따른 성능 변화:

> **실제 실험 조건**: `use_dia=true, dia_n_blocks=2`, backbone=WRN50, lr=3e-4, logdet=1e-4, scale_k=5, epochs=60

| NCL | DIA | Total | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | 비고 |
|-----|-----|-------|---------|---------|--------|--------|--------|------|
| 4 | 2 | 6 | 97.84 | 97.80 | 99.12 | 55.90 | 100.0 | MoLE4+DIA2 |
| 6 | 2 | 8 | 98.05 | 97.81 | 99.25 | 55.80 | 100.0 | MoLE6+DIA2 |
| 8 | 2 | 10 | 97.99 | 97.74 | 99.23 | 54.92 | 100.0 | MoLE8+DIA2 |
| 10 | 2 | 12 | 98.27 | 97.73 | - | 54.70 | 100.0 | MoLE10+DIA2 |
| 12 | 2 | 14 | 94.20 | 94.16 | 97.81 | 51.82 | 100.0 | ⚠️ 성능 하락 |
| 16 | 2 | 18 | 60.43 | 53.50 | 81.20 | 10.67 | 100.0 | ❌ 학습 실패 |
| 20 | 2 | 22 | 58.60 | 52.68 | 80.40 | 9.60 | 100.0 | ❌ 학습 실패 |

**분석**:
1. **NCL=4~10 + DIA2**: 안정적 학습, Img AUC 97.8~98.3%
2. **NCL=12+**: 성능 하락 시작
3. **NCL=16, 20**: 학습 실패

### 3.3 MoLE-Only (No DIA) Depth Scaling

> ✅ **실험 완료** (2026-01-07): 진정한 MoLE-Only (`use_dia=false`) 실험 수행 완료

| NCL | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | 비고 |
|-----|---------|---------|--------|--------|--------|------|
| 4 | 90.04 | 93.31 | 95.77 | 49.76 | 100.0 | |
| 6 | 92.08 | 94.27 | 96.77 | 49.85 | 100.0 | |
| **8** | **92.74** | **94.55** | **97.09** | **50.06** | 100.0 | **최적** |
| 10 | 86.28 | 88.79 | 94.47 | 43.65 | 100.0 | 성능 하락 시작 |
| 12 | 62.19 | 61.98 | 82.24 | 14.23 | 100.0 | 학습 불안정 |
| 14 | 59.52 | 61.25 | 79.41 | 11.85 | 100.0 | 학습 실패 |
| 16 | 55.97 | 57.24 | 78.61 | 9.64 | 100.0 | 학습 실패 |
| 18 | 58.93 | 58.37 | 80.04 | 9.42 | 100.0 | 학습 실패 |

**분석**:
1. **NCL=8이 최적**: Img AUC 92.74%, Pix AP 50.06%
2. **NCL=10부터 성능 급락**: Img AUC 86.28%로 하락
3. **NCL≥12: 학습 실패**: Img AUC 60% 이하
4. **DIA 없이는 전체적으로 낮은 성능**: MoLE+DIA2(NCL=6)의 98.05% 대비 MoLE-Only(NCL=8)는 92.74%

**결론**: DIA가 안정적인 학습에 중요한 역할 수행. MoLE-Only는 깊은 모델에서 불안정


### 3.4 DIA-Only (No MoLE) Depth Scaling

> ✅ **실험 완료** (2026-01-07): DIA-Only (`num_coupling_layers=0, use_dia=true`) 실험 수행 완료

| DIA | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | 비고 |
|-----|---------|---------|--------|--------|--------|------|
| **4** | **98.13** | **97.86** | **99.26** | **53.28** | 100.0 | **최적** |
| 6 | 98.15 | 97.81 | 99.22 | 51.39 | 100.0 | |
| 8 | 98.19 | 97.78 | 99.23 | 50.74 | 100.0 | |
| 10 | 98.17 | 97.73 | 99.22 | 49.61 | 100.0 | |

**분석**:
1. **모든 DIA depth에서 안정적**: Img AUC 98.1%+ 유지
2. **DIA=4가 Pix AP 최적**: 53.28%
3. **깊어질수록 Pix AP 감소**: DIA 4→10에서 53.28%→49.61%
4. **Task-Specific 학습 부재**: MoLE+DIA2(55.80%) 대비 Pix AP 2.5%p 낮음

**결론**: DIA는 학습 안정화에 효과적이나, Task-specific adaptation 없이는 Pixel-level 정밀도 한계


### 3.5 MoLE vs DIA 종합 비교

> ✅ **분석 완료** (2026-01-07)

#### Architecture 별 Best Configuration

| Architecture | Best Config | Img AUC | Pix AUC | Pix AP | Rt Acc | 특징 |
|--------------|-------------|---------|---------|--------|--------|------|
| **MoLE+DIA2** | NCL=6, DIA=2 | 98.05% | 97.81% | **55.80%** | 100% | Task-specific + 안정성 |
| **DIA-Only** | DIA=4 | **98.13%** | **97.86%** | 53.28% | 100% | 안정적 학습 |
| **MoLE-Only** | NCL=8 | 92.74% | 94.55% | 50.06% | 100% | 불안정, 낮은 성능 |

#### Depth Scaling 안정성 비교

| Architecture | 안정 학습 범위 | 최대 Depth | 비고 |
|--------------|---------------|-----------|------|
| **MoLE-Only** | NCL 4~8 | NCL=8 | NCL≥10에서 급격한 성능 하락 |
| **DIA-Only** | DIA 4~10+ | DIA=10+ | 모든 depth에서 98%+ 안정 |
| **MoLE+DIA2** | NCL 4~10 | NCL=10 | DIA가 MoLE 학습 안정화 |

#### 핵심 Findings

1. **MoLE의 역할**: Task-specific LoRA adaptation
   - Pixel-level 정밀도 향상 (Pix AP: +2.5~5.7%p)
   - 단점: 깊은 모델에서 학습 불안정

2. **DIA의 역할**: 학습 안정화
   - 모든 depth에서 Img AUC 98%+ 유지
   - MoLE의 학습 가능 범위 확장 (NCL 8→10)
   - 단점: Task-specific 학습 없어 Pix AP 제한적

3. **시너지 효과 (MoLE+DIA)**:
   - MoLE의 Task-specific 장점 + DIA의 안정화 장점 결합
   - 최적: MoLE6+DIA2 (총 8 blocks)

#### 권장 구성

| 우선순위 | Configuration | Img AUC | Pix AP | 용도 |
|---------|---------------|---------|--------|------|
| 1 (추천) | **MoLE6+DIA2** | 98.05% | 55.80% | Production (최고 Pix AP) |
| 2 | DIA-Only (DIA=4) | 98.13% | 53.28% | 빠른 학습, 안정성 우선 |
| 3 | MoLE-Only (NCL=8) | 92.74% | 50.06% | ⚠️ 비권장 (불안정)


---

## 4. Base Weight Sharing vs. Sequential/Independent Training

> ⚠️ **주의**: 이전 실험들이 실제로는 `use_dia=true`로 DIA가 활성화된 상태였음. 결과는 참고용.

Base backbone의 가중치 공유(sequential/independent) 방식에 따른 continual setting의 영향 분석:

| 설정                      | Description                                                  | Img AUC | Pix AUC | Img AP | Pix AP | 비고          |
|---------------------------|-------------------------------------------------------------|---------|---------|--------|--------|---------------|
| (a) **Base Frozen(default)**       | Base Weight Task 0 학습 후 고정 (freeze), downstream만 학습          | TBD | TBD | TBD | TBD | ⚠️ 재실험 필요 |
| (b) **Sequential Training**| Base Weight는 모든 task에서 공유하되 순차적으로 학습 | 57.47 | 55.81 | 77.38 | 7.90 | ❌ Catastrophic Forgetting (MoLE8+DIA4 기준) |
| (c) **Complete Separated**| 각 task별로 base+flow 완전 독립 (multi-head) | 98.13 | 97.74 | 99.22 | 52.49 | MoLE6+DIA2 기준 |

**실험 목적:**
- Base backbone의 동결, 순차 학습, 완전 독립 세팅 간 성능/일반화/forgetting trade-off 비교
- 실제 deployment scenario에 맞는 가중치 공유 전략 도출

### 4.2 분석

> ⚠️ 이전 분석은 config 불일치로 인해 유효하지 않음. 재실험 후 업데이트 필요.

# Hyperparameter Analysis

> ⚠️ **주의**: 아래 실험들은 실제로 `use_dia=true, dia_n_blocks=2`로 수행됨 (MoLE+DIA2 기준)

## lora_rank (MoLE6+DIA2 기준)

> **실제 설정**: NCL=6, DIA=2, lr=3e-4, logdet=1e-4, scale_k=5

| lora_rank | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-----------|---------|---------|--------|----------|------|
| 16        | 98.06   | 97.82   | 55.86  | +0.06    | MoLE6+DIA2 |
| 32        | 98.04   | 97.82   | 55.89  | +0.09    | MoLE6+DIA2 |
| **64**    | **98.05** | **97.81** | **55.80** | - | **MoLE6+DIA2 기준** |
| 128       | 98.04   | 97.82   | 55.80  | 0.00    | MoLE6+DIA2 |

### 분석
- 모든 LoRA rank(16~128)에서 유사한 성능
- Pix AP: 55.80~55.89% 범위로 안정적
- **결론**: LoRA rank는 성능에 큰 영향 없음. 파라미터 효율성을 위해 rank=16~32 권장

## lambda_logdet
> ⚠️ MoLE+DIA2 기준 (NCL=6, DIA=2)

| lambda_logdet | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|---------------|---------|--------|---------|--------|------|
| **1e-4**      | 98.05 | 99.25 | 97.81 | 55.80 | MoLE6+DIA2 |
| 기타          | TBD     | TBD    | TBD     | TBD    | 재실험 필요 |

## scale_context_kernel
> ⚠️ MoLE+DIA2 기준 (NCL=6, DIA=2)

| scale_context_kernel | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|---------------------|---------|--------|---------|--------|------|
| **5**               | 98.05 | 99.25 | 97.81 | 55.80 | MoLE6+DIA2 |
| 0 (disabled)        | TBD     | TBD    | TBD     | TBD    | 재실험 필요 |

## spatial_context_kernel
> ⚠️ MoLE+DIA2 기준 (NCL=6, DIA=2)

| spatial_context_kernel | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-----------------------|---------|--------|---------|--------|------|
| **3**                 | 98.05 | 99.25 | 97.81 | 55.80 | MoLE6+DIA2 |

## Tail Aware Loss weight (tail_weight)
> ⚠️ MoLE+DIA2 기준 (NCL=6, DIA=2)

| tail_weight | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-------------|---------|--------|---------|--------|------|
| **0.7**     | 98.05 | 99.25 | 97.81 | 55.80 | MoLE6+DIA2 |
| 기타        | TBD     | TBD    | TBD     | TBD    | 재실험 필요 |

## Image Anomaly Score Aggregation K (score_aggregation_top_k)
> ⚠️ MoLE+DIA2 기준 (NCL=6, DIA=2)

| top_k | Img AUC | Img AP | Pix AUC | Pix AP | 비고 |
|-------|---------|--------|---------|--------|------|
| **3** | 98.05 | 99.25 | 97.81 | 55.80 | MoLE6+DIA2 |
| 기타  | TBD     | TBD    | TBD     | TBD    | 재실험 필요 |

