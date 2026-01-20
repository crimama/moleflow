# SVD Analysis: Low-Rank Adaptation의 실증적 검증

## 목적
"Low-rank adaptation이 충분한 이유"를 실증적으로 검증

## 배경
- MoLE-Flow는 coupling subnet을 Base + LoRA로 분해
- 이론적 주장: task 간 density transformation 구조가 유사하므로 low-rank adaptation으로 충분
- 검증 필요: 실제로 필요한 adaptation이 intrinsically low-rank인지 확인

---

## 실험 1: Trained LoRA Weight SVD Analysis

### 실험 설계
- **방법**: MoLE-Flow를 3개 task (leather, grid, transistor)에 대해 학습 후, 각 task의 LoRA weight (ΔW = B @ A)에 대해 SVD 분석
- **모델 설정**:
  - Backbone: wide_resnet50_2
  - LoRA Rank: 64
  - Coupling Layers: 6 (MoLE blocks)
  - DIA Blocks: 2
  - Training: 30 epochs per task
- **측정 지표**:
  - Effective Rank (95%): 전체 에너지의 95%를 설명하는 최소 rank
  - Effective Rank (99%): 전체 에너지의 99%를 설명하는 최소 rank
  - Energy at r=64: 상위 64개 singular value가 차지하는 에너지 비율

### 실험 결과 (2026-01-16)

#### Per-Task SVD 분석 결과

| Task | Class | Layers Analyzed | Eff. Rank (95%) | Eff. Rank (99%) | Energy at r=64 |
|------|-------|-----------------|-----------------|-----------------|----------------|
| Task 0 | leather | 24 | **14.5 ± 8.6** | 29.5 ± 12.9 | 100.0% |
| Task 1 | grid | 24 | **1.3 ± 0.7** | 1.9 ± 2.4 | 100.0% |
| Task 2 | transistor | 24 | **1.5 ± 1.2** | 3.5 ± 3.7 | 100.0% |

#### 핵심 발견

1. **Task 0 (Base + LoRA 동시 학습)**
   - Effective Rank (95%) = 14.5 ± 8.6
   - 설정된 LoRA rank (64) 대비 매우 낮은 intrinsic rank
   - Base와 LoRA가 함께 학습되어도 adaptation은 본질적으로 저차원

2. **Task 1, 2 (LoRA만 학습)**
   - Effective Rank (95%) ≈ 1-2
   - 극도로 낮은 intrinsic dimensionality
   - Base가 학습한 transformation 구조 위에서 최소한의 조정만 필요

3. **Energy at r=64 = 100%**
   - 현재 LoRA rank=64 설정이 완전히 충분함 (과잉 설정)
   - r=16 또는 r=32로도 충분할 가능성

### 시각화
- `./analysis_results/lora_svd_trained/svd_spectrum_by_task.png` - Singular value spectrum
- `./analysis_results/lora_svd_trained/effective_rank_analysis.png` - Effective rank 분포

---

## 실험 2: Independent Training SVD Analysis

### 실험 설계
- **방법**: 동일한 초기화에서 시작하여 두 개의 독립적인 모델을 각각 다른 task에서 학습 후, 두 모델의 weight 차이에 대해 SVD 분석
- **목적**: "처음부터 학습할 때 task 간 weight 차이가 얼마나 되는가?"를 측정
- **모델 설정**:
  - Backbone: wide_resnet50_2
  - Coupling Layers: 6
  - Training: 30 epochs per task
  - No LoRA (full parameters trained)
- **비교 대상**: leather vs grid

### 실험 결과 (2026-01-16)

#### 전체 결과 요약

| Metric | Value |
|--------|-------|
| Mean Effective Rank (95%) | **181.5 ± 45.7** |
| Mean Effective Rank (99%) | 309.1 ± 69.1 |
| Range (95%) | [108, 245] |
| Energy at r=64 | **74.4%** |
| Energy at r=128 | ~90% |
| Relative Weight Change | 43.2% ± 9.6% |

#### Per-Layer 분석 결과

| Layer | Shape | Eff. Rank (95%) | Energy at r=64 |
|-------|-------|-----------------|----------------|
| subnet0_layer1 | 1024×512 | 108 | 88.4% |
| subnet0_layer2 | 1024×1024 | 138 | 84.3% |
| subnet1_layer1 | 1024×512 | 123 | 85.7% |
| subnet1_layer2 | 1024×1024 | 180 | 75.7% |
| subnet2_layer1 | 1024×512 | 147 | 81.0% |
| subnet2_layer2 | 1024×1024 | 215 | 68.8% |
| subnet3_layer1 | 1024×512 | 162 | 77.4% |
| subnet3_layer2 | 1024×1024 | 245 | 62.7% |
| subnet4_layer1 | 1024×512 | 182 | 72.8% |
| subnet4_layer2 | 1024×1024 | 241 | 64.6% |
| subnet5_layer1 | 1024×512 | 193 | 68.5% |
| subnet5_layer2 | 1024×1024 | 244 | 63.5% |

### 시각화
- `./analysis_results/svd_full_finetune/independent_leather_grid/svd_spectrum_*.png`
- `./analysis_results/svd_full_finetune/independent_leather_grid/energy_at_ranks.png`
- `./analysis_results/svd_full_finetune/independent_leather_grid/effective_rank_histogram.png`

---

## 두 실험의 비교 및 해석

### 핵심 비교

| 분석 유형 | Effective Rank (95%) | Energy at r=64 | 의미 |
|-----------|---------------------|----------------|------|
| **Trained LoRA** (on frozen base) | **1.3 ~ 14.5** | **100%** | Frozen base 위에서 필요한 adaptation |
| **Independent Training** (from scratch) | **181.5** | **74.4%** | 처음부터 학습 시 task 간 차이 |

### 핵심 통찰

**두 실험이 답하는 질문이 다름:**

1. **Independent Training**: "두 task의 최적 모델이 얼마나 다른가?"
   - 답: 상당히 다름 (Effective Rank ≈ 180)
   - 처음부터 학습하면 각 task에 특화된 다른 weight로 수렴

2. **Trained LoRA**: "Frozen base 위에서 얼마나 적응이 필요한가?"
   - 답: 매우 적음 (Effective Rank ≈ 1-15)
   - Base가 일반적인 transformation 구조를 학습하면 task별 적응은 최소화

### 이론적 의미

이 차이가 **MoLE-Flow의 핵심 설계 근거**:

1. **Base NF의 역할**
   - Task 0에서 학습된 base가 "feature → Gaussian" 변환의 **일반적 구조**를 포착
   - 이 구조는 task에 관계없이 재사용 가능

2. **Low-rank Adaptation의 정당화**
   - 처음부터 학습하면 high-rank 차이가 발생하지만
   - 좋은 base 위에서는 **low-rank adaptation만으로 충분**
   - 이것이 LoRA가 효과적인 이유

3. **실용적 함의**
   - LoRA rank=64는 과잉 설정 (실제 필요 rank ≈ 15 이하)
   - r=32 또는 r=16으로도 충분할 가능성
   - 메모리/계산 효율성 향상 여지 존재

---

## 결론

**"Low-rank adaptation이 충분한 이유"에 대한 실증적 검증 완료:**

1. **Frozen base 위의 LoRA adaptation**은 intrinsically very low-rank (Eff. Rank ≈ 1-15)
2. **처음부터 학습**하면 task 간 weight 차이가 high-rank (Eff. Rank ≈ 180)
3. **핵심 통찰**: Base NF가 일반적인 density transformation 구조를 학습하므로, task-specific adaptation은 low-rank로 충분
4. 설정된 LoRA rank=64는 실제 필요한 것보다 4배 이상 과잉

## 스크립트 위치
- `/Volume/MoLeFlow/scripts/analyze_trained_lora_svd.py` - Trained LoRA SVD 분석
- `/Volume/MoLeFlow/scripts/analyze_svd_full_finetune.py` - Independent Training SVD 분석
