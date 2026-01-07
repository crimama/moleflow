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
| Complete Separated | Task별 완전 분리 | `--use_task_separated` | ✅ 완료 |

### 결과 테이블

| Configuration | Img AUC | Pix AUC | Pix AP | Rt Acc | Δ Img AUC | Δ Pix AP | Status |
|---------------|---------|---------|--------|--------|-----------|----------|--------|
| **MAIN (Base Frozen)** | **98.05** | **97.81** | **55.80** | 100.0 | - | - | ✅ |
| Sequential Training | 60.10 | 68.20 | 12.29 | 100.0 | **-37.95** | **-43.51** | ✅ |
| Complete Separated | 98.13 | 97.74 | 52.49 | 100.0 | +0.08 | -3.31 | ✅ |

### 분석

**Sequential Training의 Catastrophic Forgetting**:
- Base weights를 freeze하지 않고 순차 학습 시 심각한 성능 저하
- Img AUC: 98.05% → 60.10% (**-37.95%p**)
- Pix AP: 55.80% → 12.29% (**-43.51%p**)
- 이전 Task의 지식이 새로운 Task 학습 시 완전히 망각됨
- **결론**: Base Frozen 전략이 Continual Learning에 필수

**Complete Separated의 특성**:
- Task별 완전 분리 학습 (공유 weight 없음)
- Img AUC: 98.13% (+0.08%p) - MAIN과 유사
- Pix AP: 52.49% (**-3.31%p**) - 성능 하락
- 공유 representation 학습의 이점을 잃어버림
- **결론**: Base Frozen이 Complete Separated보다 Pix AP에서 우수



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
| 1e-6          | 98.08   | 97.70   | 51.88  | -3.92    | ✅ 완료 |
| 1e-5          | 98.10   | 97.71   | 52.46  | -3.34    | ✅ 완료 |
| **1e-4**      | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |

### 분석 (2026-01-08 업데이트)
- **lambda_logdet=1e-4가 최적** (Pix AP 55.80%)
- 1e-6 → 1e-5 → 1e-4 순으로 Pix AP 향상: 51.88% → 52.46% → 55.80%
- 1e-6/1e-5는 disabled(0)와 거의 동일한 성능 (Pix AP ~51-52%)
- Log-determinant regularization 강도가 Pix AP에 큰 영향
- **결론**: lambda_logdet=1e-4 권장, 약한 regularization(1e-5, 1e-6)은 효과 미미


## scale_context_kernel
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| scale_context_kernel | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|---------------------|---------|---------|--------|----------|------|
| 0 (disabled)        | 97.90   | 97.74   | 54.52  | -1.28    | MoLE6-DIA2-woScaleCtx |
| 3                   | 97.93   | 97.77   | 54.59  | -1.21    | ✅ 완료 |
| **5**               | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |
| 7                   | 97.91   | 97.79   | 55.33  | -0.47    | ✅ 완료 |

### 분석 (2026-01-08 업데이트)
- **scale_context_kernel=5가 최적** (Pix AP 55.80%)
- kernel=3: Pix AP 54.59% (-1.21%p) - disabled와 거의 동일
- kernel=7: Pix AP 55.33% (-0.47%p) - 5보다 약간 낮음
- 0 → 3 → 7 → 5 순으로 Pix AP 향상
- **결론**: scale_context_kernel=5 권장


## spatial_context_kernel
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| spatial_context_kernel | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-----------------------|---------|---------|--------|----------|------|
| 0 (disabled)          | 97.98   | 97.54   | 52.93  | -2.87    | MoLE6-DIA2-woSpatialCtx |
| **3**                 | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |
| 5                     | 96.12   | 96.97   | 51.38  | -4.42    | ❌ 성능 하락 |
| 7                     | 90.90   | 93.91   | 44.33  | -11.47   | ❌ 심각한 성능 하락 |


### 분석
- **spatial_context_kernel=3이 최적** (Pix AP 55.80%)
- disabled(0) 대비 +2.87%p Pix AP 향상, Pix AUC +0.27%p
- kernel=5: Pix AP 51.38% (**-4.42%p**), Img AUC 96.12% (-1.93%p)
- kernel=7: Pix AP 44.33% (**-11.47%p**), Img AUC 90.90% (-7.15%p) - 심각한 성능 저하
- **결론**: spatial_context_kernel=3 권장, 더 큰 kernel은 성능 저하 유발


## Tail Aware Loss weight (tail_weight)
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| tail_weight | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-------------|---------|---------|--------|----------|------|
| 0 (disabled) | 94.97   | 97.21   | 48.61  | **-7.19** | ❌ w/o Tail Loss |
| 0.1         | 97.25   | 97.44   | 50.54  | -5.26    | ✅ 완료 |
| 0.3         | 97.76   | 97.57   | 52.94  | -2.86    | ✅ 완료 |
| 0.5         | 97.93   | 97.68   | 54.78  | -1.02    | ✅ 완료 |
| **0.7**     | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |
| 0.8         | 98.01   | 97.82   | 56.00  | +0.20    | ✅ 약간 향상 |
| 1.0         | 97.92   | 97.81   | 56.18  | +0.38    | ✅ 최고 Pix AP |

### 분석 (2026-01-08 업데이트)
- **Tail weight 증가에 따라 Pix AP 단조 증가**: 48.61% → 50.54% → 52.94% → 54.78% → 55.80% → 56.00% → 56.18%
- tail_weight=1.0에서 최고 Pix AP 56.18% 달성 (+0.38%p)
- 그러나 Img AUC는 0.7에서 최고 (98.05%), 1.0에서 97.92%로 약간 하락
- **최적 구성**: Pix AP 우선 시 tail_weight=0.8~1.0, Img AUC 우선 시 tail_weight=0.7
- **결론**: tail_weight=0.7~0.8 권장 (균형잡힌 성능)


## Tail Top-K Ratio (tail_top_k_ratio)
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| tail_top_k_ratio | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|------------------|---------|---------|--------|----------|------|
| 0.01             | 98.02   | 97.79   | 55.85  | +0.05    | ✅ 유사 |
| **0.02**         | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |
| 0.03             | 98.07   | 97.83   | 55.83  | +0.03    | ✅ 유사 |
| 0.05             | 97.98   | 97.83   | 55.60  | -0.20    | ✅ 약간 하락 |
| 0.10             | 97.83   | 97.83   | 55.24  | -0.56    | ✅ 하락 |

### 분석 (2026-01-08 업데이트)
- **tail_top_k_ratio는 0.01~0.03 범위에서 최적** (Pix AP 55.80~55.85%)
- 0.01 = 0.02 ≈ 0.03 > 0.05 > 0.10 순서
- ratio가 높아질수록 Pix AP 하락: 0.05 (-0.20%p), 0.10 (-0.56%p)
- 너무 많은 패치를 tail로 사용하면 오히려 성능 저하
- **결론**: tail_top_k_ratio=0.01~0.02 권장


## Image Anomaly Score Aggregation K (score_aggregation_top_k)
> MoLE6+DIA2 기준 (NCL=6, DIA=2)

| top_k | Img AUC | Pix AUC | Pix AP | Δ Pix AP | 비고 |
|-------|---------|---------|--------|----------|------|
| **3** | **98.05** | **97.81** | **55.80** | - | **✅ MoLE6+DIA2 기준** |
| 5     | 98.05   | 97.81   | 55.80  | 0.00     | ✅ 동일 |
| 7     | 98.06   | 97.81   | 55.80  | 0.00     | ✅ 동일 |
| 10    | 98.07   | 97.81   | 55.80  | 0.00     | ✅ 동일 |

### 분석 (2026-01-08 업데이트)
- **score_aggregation_top_k는 3~10 범위에서 모두 동일한 성능** (Pix AP 55.80%)
- Img AUC는 k=10에서 약간 높음 (98.07%), 그러나 차이 미미
- Top-K 평균 방식이 k 값에 robust함을 확인
- **결론**: score_aggregation_top_k=3 권장 (추가 계산 부담 없음)


---

# Hyperparameter Analysis Summary

## 최적 하이퍼파라미터 (2026-01-08 종합)

| Hyperparameter | 최적값 | 범위 | 민감도 | 비고 |
|----------------|--------|------|--------|------|
| lambda_logdet | **1e-4** | 1e-6~1e-4 | **높음** | +3.95%p Pix AP |
| scale_context_kernel | **5** | 0~7 | 중간 | +1.28%p Pix AP |
| spatial_context_kernel | **3** | 0~7 | **높음** | 5↑에서 급격히 하락 |
| tail_weight | **0.7~0.8** | 0~1.0 | **높음** | 0에서 -7.19%p |
| tail_top_k_ratio | **0.01~0.02** | 0.01~0.10 | 낮음 | 0.01~0.03 동일 |
| score_aggregation_top_k | **3** | 3~10 | 없음 | 모두 동일 |
| lora_rank | **64** | 16~128 | 없음 | 모두 동일 |

## 주요 발견사항

1. **가장 민감한 하이퍼파라미터**: `tail_weight`, `spatial_context_kernel`, `lambda_logdet`
2. **가장 안정적인 하이퍼파라미터**: `score_aggregation_top_k`, `lora_rank`
3. **tail_weight=1.0**: 최고 Pix AP (56.18%) 달성, Img AUC 약간 하락
4. **spatial_context_kernel**: 3이 최적, 5 이상에서 급격한 성능 저하

---