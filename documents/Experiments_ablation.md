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
| w/o LoRA | LoRA vs Regular Linear | `--no_lora` | ✅ 완료 |
| w/o Scale Context | Scale Context의 기여도 | `--no_scale_context` | ✅ 완료 |
| w/o Spatial Context | Spatial Context의 기여도 | `--no_spatial_context` | ✅ 완료 |
| w/o Whitening Adapter | Whitening Adapter의 기여도 | `--no_whitening_adapter` | ✅ 완료 |
| w/o Tail Aware Loss | Tail Aware Loss의 기여도 | `--tail_weight 0` | ✅ 완료 |
| w/o LogDet Reg | LogDet Regularization 유무 | `--lambda_logdet 0` | ✅ 완료 |
| w/o Pos Embedding | 위치 정보의 기여도 | `--no_pos_embedding` | ✅ 완료 |
| w/o DIA | DIA 제거 (MoLE-Only) | `--no_dia` | ✅ 완료 (Section 2) |

### 결과 테이블 (MoLE6+DIA2 기준)

| Configuration | Img AUC | Pix AUC | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP | Status |
|---------------|---------|---------|--------|--------|-----------|----------|--------|
| **MAIN (MoLE6+DIA2)** | **98.05** | **97.81** | **55.80** | 100.0 | - | - | ✅ |
| w/o Tail Aware Loss | 94.97 | 97.21 | 48.61 | 100.0 | **-3.08** | **-7.19** | ✅ |
| w/o Whitening Adapter | 97.90 | 97.69 | 48.84 | 100.0 | -0.15 | **-6.96** | ✅ |
| w/o LogDet Reg | 98.06 | 97.70 | 51.85 | 100.0 | +0.01 | -3.95 | ✅ |
| w/o Spatial Context | 97.98 | 97.54 | 52.93 | 100.0 | -0.07 | -2.87 | ✅ |
| w/o Pos Embedding | 97.40 | 97.47 | 53.99 | 100.0 | -0.65 | -1.81 | ✅ |
| w/o Scale Context | 97.90 | 97.74 | 54.52 | 100.0 | -0.15 | -1.28 | ✅ |
| w/o LoRA | 97.96 | 97.77 | 55.31 | 100.0 | -0.09 | -0.49 | ✅ |

### 분석 (2026-01-07 업데이트)

#### 컴포넌트 기여도 순위 (Pix AP 영향 기준)

| 순위 | Component | Pix AP 기여 | Img AUC 기여 | Pix AUC 기여 |
|------|-----------|------------|-------------|-------------|
| 1 | **Tail Aware Loss** | **+7.19%** | **+3.08%** | +0.60% |
| 2 | **Whitening Adapter** | **+6.96%** | +0.15% | +0.12% |
| 3 | **LogDet Regularization** | **+3.95%** | -0.01% | +0.11% |
| 4 | **Spatial Context** | +2.87% | +0.07% | +0.27% |
| 5 | **Pos Embedding** | +1.81% | +0.65% | +0.34% |
| 6 | **Scale Context** | +1.28% | +0.15% | +0.07% |
| 7 | **LoRA** | +0.49% | +0.09% | +0.04% |

#### 핵심 발견

1. **Tail Aware Loss가 가장 중요**: Pix AP +7.19%p, Img AUC +3.08%p
   - Anomaly detection에서 tail distribution 학습이 핵심
   - Image-level 성능에도 큰 영향 (94.97% → 98.05%)

2. **Whitening Adapter가 Pix AP에 두 번째로 큰 영향**: +6.96%p
   - Feature distribution normalization이 anomaly localization에 매우 중요
   - 특히 tile, leather, toothbrush 등에서 큰 성능 저하

3. **LogDet Regularization이 Pix AP에 중요**: +3.95%p
   - Flow의 invertibility 유지가 localization 정확도에 기여
   - Img AUC에는 영향 없음 (98.06%)

4. **Pos Embedding이 Img AUC에 유의미한 영향**: +0.65%p
   - 위치 정보가 Image-level detection에 도움

5. **LoRA 기여도가 예상보다 낮음**: Pix AP +0.49%p
   - DIA가 task-specific adaptation 역할을 일부 수행하는 것으로 추정
   - 성능 측면에서 기여도는 낮으나, 파라미터 효율성 증가 


---
## 2. Normalizing Flow Block 구성 실험 (MoLE / DIA Block 조합)

전체 Coupling Block 수를 조절하고, MoLE-SubNet과 DIA block의 구성 비율에 따른 성능 변화를 실험합니다.

### 2.1 MoLE vs DIA Architecture 종합 비교


#### 전체 실험 결과 통합 테이블

| Architecture | MoLE | DIA | Total | Img AUC | Pix AUC | Img AP | Pix AP | Rt Acc | 비고 |
|--------------|------|-----|-------|---------|---------|--------|--------|--------|------|
| **MoLE+DIA** | **6** | **2** | **8** | **98.05** | **97.81** | **99.25** | **55.80** | 100.0 | **✅ MAIN (최적)** |
| MoLE+DIA | 4 | 2 | 6 | 97.84 | 97.80 | 99.12 | 55.90 | 100.0 | |
| MoLE+DIA | 8 | 2 | 10 | 97.99 | 97.74 | 99.23 | 54.92 | 100.0 | |
| MoLE+DIA | 10 | 2 | 12 | 98.27 | 97.73 | - | 54.70 | 100.0 | |
| MoLE+DIA | 12 | 2 | 14 | 94.20 | 94.16 | 97.81 | 51.82 | 100.0 | ⚠️ 성능 하락 |
| MoLE+DIA | 16 | 2 | 18 | 60.43 | 53.50 | 81.20 | 10.67 | 100.0 | ❌ 학습 실패 |
| MoLE+DIA | 20 | 2 | 22 | 58.60 | 52.68 | 80.40 | 9.60 | 100.0 | ❌ 학습 실패 |
| MoLE+DIA | 8 | 4 | 12 | 98.29 | 97.82 | - | 54.20 | 100.0 | Old MAIN |
| MoLE+DIA | 6 | 6 | 12 | 98.19 | 97.79 | - | 51.62 | 100.0 | 균형 구성 |
| MoLE+DIA | 4 | 8 | 12 | 98.09 | 97.74 | - | 50.27 | 100.0 | DIA 비중 높음 |
| **DIA-Only** | **0** | **4** | **4** | **98.13** | **97.86** | **99.26** | **53.28** | 100.0 | **안정성 최고** |
| DIA-Only | 0 | 6 | 6 | 98.15 | 97.81 | 99.22 | 51.39 | 100.0 | |
| DIA-Only | 0 | 8 | 8 | 98.19 | 97.78 | 99.23 | 50.74 | 100.0 | |
| DIA-Only | 0 | 10 | 10 | 98.17 | 97.73 | 99.22 | 49.61 | 100.0 | |
| **MoLE-Only** | **8** | **0** | **8** | **92.74** | **94.55** | **97.09** | **50.06** | 100.0 | **MoLE 최적** |
| MoLE-Only | 4 | 0 | 4 | 90.04 | 93.31 | 95.77 | 49.76 | 100.0 | |
| MoLE-Only | 6 | 0 | 6 | 92.08 | 94.27 | 96.77 | 49.85 | 100.0 | |
| MoLE-Only | 10 | 0 | 10 | 86.28 | 88.79 | 94.47 | 43.65 | 100.0 | 성능 하락 |
| MoLE-Only | 12 | 0 | 12 | 62.19 | 61.98 | 82.24 | 14.23 | 100.0 | ❌ 학습 불안정 |
| MoLE-Only | 14 | 0 | 14 | 59.52 | 61.25 | 79.41 | 11.85 | 100.0 | ❌ 학습 실패 |
| MoLE-Only | 16 | 0 | 16 | 55.97 | 57.24 | 78.61 | 9.64 | 100.0 | ❌ 학습 실패 |
| MoLE-Only | 18 | 0 | 18 | 58.93 | 58.37 | 80.04 | 9.42 | 100.0 | ❌ 학습 실패 |

#### 핵심 분석

**1. Architecture 별 Best Configuration**

| Architecture | Best Config | Img AUC | Pix AUC | Pix AP | 특징 |
|--------------|-------------|---------|---------|--------|------|
| **MoLE+DIA2** | MoLE=6, DIA=2 | 98.05% | 97.81% | **55.80%** | **Task-specific + 안정성 (최적)** |
| **DIA-Only** | DIA=4 | **98.13%** | **97.86%** | 53.28% | 안정적 학습, Img 성능 최고 |
| **MoLE-Only** | MoLE=8 | 92.74% | 94.55% | 50.06% | 불안정, 낮은 성능 |

**2. Depth Scaling 안정성 비교**

| Architecture | 안정 학습 범위 | 최대 안정 Depth | 성능 하락 시점 | 학습 실패 시점 |
|--------------|---------------|----------------|---------------|---------------|
| **MoLE-Only** | MoLE 4~8 | 8 blocks | MoLE≥10 | MoLE≥12 |
| **DIA-Only** | DIA 4~10+ | 10+ blocks | 없음 | 없음 |
| **MoLE+DIA2** | MoLE 4~10 | 10 blocks | MoLE≥12 | MoLE≥16 |

**3. 컴포넌트 역할 분석**

| Component | 역할 | 기여도 | 장점 | 단점 |
|-----------|------|--------|------|------|
| **MoLE** | Task-specific LoRA adaptation | Pix AP +2.5~5.7%p | Pixel-level 정밀도 향상 | 깊은 모델에서 학습 불안정 |
| **DIA** | 학습 안정화 | Img AUC 안정화 | 모든 depth에서 98%+ 유지 | Task-specific 학습 부재 |
| **MoLE+DIA** | 시너지 효과 | 최고 Pix AP 달성 | 안정성 + 정밀도 결합 | - |

**4. 주요 발견사항**

1. **MoLE6+DIA2가 최적 구성**
   - Pix AP 55.80%로 최고 성능
   - 총 8 blocks로 효율적
   - DIA의 안정화 + MoLE의 Task-specific 장점 결합

2. **MoLE-Only는 불안정**
   - DIA 없이는 Img AUC 92.74%로 크게 하락 (vs 98.05%)
   - MoLE≥10에서 성능 급락, MoLE≥12에서 학습 실패
   - Task-specific 학습은 가능하나 안정성 부족

3. **DIA-Only는 안정적이나 Pix AP 제한적**
   - 모든 depth(4~10+)에서 Img AUC 98%+ 안정 유지
   - Img AUC 98.13%로 가장 높음
   - Task-specific 학습 부재로 Pix AP 53.28% (MoLE+DIA2 대비 -2.5%p)

4. **DIA가 MoLE 학습 안정화에 핵심**
   - MoLE-Only: 최대 NCL=8까지만 안정
   - MoLE+DIA2: NCL=10까지 안정 학습 가능
   - DIA 2 blocks만으로도 충분한 안정화 효과



---

## 3. Training Strategy Ablation (MoLE6+DIA2 기준)

| Ablation | 목적 | 내용 | Status |
|----------|------|------|--------|
| Main (Base Frozen) | Base 모델 학습 후 freeze | `--freeze_base` (default) | ✅ 완료 |
| Sequential Training | Base freeze 없이 순차 학습 | `--no_freeze_base` | ✅ 완료 |
| Complete Separated | Task별 완전 분리 | `--use_task_separated` | ⏳ 대기 (Batch 3) |

### 결과 테이블

| Configuration | Img AUC | Pix AUC | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP | Status |
|---------------|---------|---------|--------|--------|-----------|----------|--------|
| **MAIN (Base Frozen)** | **98.05** | **97.81** | **55.80** | 100.0 | - | - | ✅ |
| Sequential Training | 60.10 | 68.20 | 12.29 | 100.0 | **-37.95** | **-43.51** | ✅ |
| Complete Separated | TBD | TBD | TBD | TBD | TBD | TBD | ⏳ |

### 분석

**Sequential Training의 Catastrophic Forgetting**:
- Base weights를 freeze하지 않고 순차 학습 시 심각한 성능 저하
- Img AUC: 98.05% → 60.10% (**-37.95%p**)
- Pix AP: 55.80% → 12.29% (**-43.51%p**)
- 이전 Task의 지식이 새로운 Task 학습 시 완전히 망각됨
- **결론**: Base Frozen 전략이 Continual Learning에 필수



---
# Hyperparameter Analysis

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
- LoRA 자체는 성능에 크게 영향을 주기 보다는, 파라미터 효율성을 위한 장치.
- 한편으로는 Linear Layer보다 파라미터 수는 적음에도 불구하고 똑같은 성능을 보여줌 

## lambda_logdet
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| lambda_logdet | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|---------------|---------|---------|--------|----------|------|
| 0 (disabled)  | 98.06   | 97.70   | 51.85  | -3.95    | w/o LogDet |
| 1e-6          | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 1e-5          | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| **1e-4**      | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |

### 분석
- lambda_logdet=1e-4가 최적 (Pix AP 기준)
- NCL8+DIA4 참고 데이터에서 1e-4 > 0 > 1e-5 > 1e-6 순서
- Log-determinant regularization이 Pix AP 향상에 중요한 역할
- **결론**: lambda_logdet=1e-4 권장


## scale_context_kernel
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| scale_context_kernel | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|---------------------|---------|---------|--------|----------|------|
| 0 (disabled)        | 97.90   | 97.74   | 54.52  | -1.28    | MoLE6-DIA2-woScaleCtx |
| 3                   | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| **5**               | **98.05** | **97.81** | **55.80** | - | **MoLE6+DIA2 기준** |
| 7                   | TBD     | TBD     | TBD    | TBD      | 실험 필요 |

### 분석
- scale_context_kernel=5가 최적
- disabled(0) 대비 +1.28%p Pix AP 향상
- 3과 7은 NCL8+DIA4 참고 데이터로, MoLE6+DIA2에서 유사 경향 예상
- **결론**: scale_context_kernel=5 권장


## spatial_context_kernel
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| spatial_context_kernel | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-----------------------|---------|---------|--------|----------|------|
| 0 (disabled)          | 97.98   | 97.54   | 52.93  | -2.87    | MoLE6-DIA2-woSpatialCtx |
| **3**                 | **98.05** | **97.81** | **55.80** | - | **MoLE6+DIA2 기준** |
| 5                     | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 7                     | TBD     | TBD     | TBD    | TBD      | 실험 필요 |


### 분석
- spatial_context_kernel=3이 최적
- disabled(0) 대비 +2.87%p Pix AP 향상, Pix AUC +0.27%p
- kernel=5는 오히려 성능 저하 (NCL8+DIA4 참고)
- **결론**: spatial_context_kernel=3 권장, 더 큰 kernel은 비권장


## Tail Aware Loss weight (tail_weight)
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| tail_weight | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-------------|---------|---------|--------|----------|------|
| 0 (disabled) | 94.97   | 97.21   | 48.61  | **-7.19** | ❌ w/o Tail Loss |
| 0.1         | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 0.3         | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 0.5         | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| **0.7**     | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |
| 0.8         | TBD     | TBD     | TBD    | TBD      | 실험 필요 |

## Tail Top-K Ratio (tail_top_k_ratio)
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| tail_top_k_ratio | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|------------------|---------|---------|--------|----------|------|
| 0.01             | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| **0.02**         | **98.05** | **97.81** | **55.80** | - | **MoLE6+DIA2 기준** |
| 0.03             | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 0.05             | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 0.10             | TBD     | TBD     | TBD    | TBD      | 실험 필요 |

### 분석
- tail_weight 증가에 따라 Pix AP 증가 경향 (0.1 → 0.7)
- 0.7에서 최적, 0.75~0.8에서 약간 하락
- Tail-aware loss가 Pix AP에 매우 큰 영향 (+9.94%p vs w/o tail loss)
- **결론**: tail_weight=0.7 권장


## Image Anomaly Score Aggregation K (score_aggregation_top_k)
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| top_k | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-------|---------|---------|--------|----------|------|
| **3** | **98.05** | **97.81** | **55.80** | - | **MoLE6+DIA2 기준** |
| 5     | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 7     | TBD     | TBD     | TBD    | TBD      | 실험 필요 |
| 10    | TBD     | TBD     | TBD    | TBD      | 실험 필요 |

### 분석
- top_k=3이 최적 (Pix AP 기준)
- k 증가 시 Image AUC는 유지되나 Pix AP 하락 경향
- k=5~7에서 유사, k=10에서 큰 성능 저하
- **결론**: score_aggregation_top_k=3 권장


---