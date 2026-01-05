# MoLE-Flow 실험 결과 분석

## 1. 실험 개요

- **데이터셋**: MVTec AD (15 classes), VISA (12 classes), MPDD (6 classes)
- **Backbone**: Wide ResNet50
- **평가 지표**: Image AUROC, Pixel AUROC, Pixel AP
- **총 완료 실험**: 98개 (MVTec-WRN50 기준)

---

## 2. 하이퍼파라미터 설정 그룹

실험은 크게 두 가지 기본 설정 그룹으로 나뉩니다. **공정한 비교를 위해 동일 그룹 내에서만 비교합니다.**

| 그룹 | lr | lambda_logdet | scale_ctx_k | 실험 수 | 비고 |
|:----:|:--:|:-------------:|:-----------:|:-------:|:-----|
| **그룹 1** | 2e-4 | 1e-5 | 3 | 39개 | 초기 Baseline |
| **그룹 2** | 3e-4 | 1e-4 | 5 | 14개 | MAIN 설정 |

---

## 3. 그룹 1: 초기 Baseline 설정 실험 결과

### 공통 하이퍼파라미터
```
lr = 2e-4
lambda_logdet = 1e-5
scale_context_kernel = 3
epochs = 60
coupling_layers = 8
lora_rank = 64
```

### 3.1 전체 결과 (Image AUC 기준 정렬)

| Rank | Experiment | Img AUC | Pix AP | DIA | TailW | TopK | Ablation |
|:----:|:-----------|:-------:|:------:|:---:|:-----:|:----:|:---------|
| 1 | DIA7 | **0.9830** | 0.4580 | 7 | - | - | - |
| 2 | TopK5-TailW0.5-DIA8 | 0.9828 | 0.4656 | 8 | - | - | - |
| 3 | DIA8 | 0.9825 | 0.4546 | 8 | - | - | - |
| 4 | TopK5-TailW0.5-DIA6 | 0.9824 | 0.4723 | 6 | - | - | - |
| 5 | TopK7-TailW0.5 | 0.9821 | 0.4866 | 4 | - | - | - |
| 6 | TopK5-TailW0.5 | 0.9820 | **0.4866** | 4 | - | - | - |
| 7 | DIA6 | 0.9820 | 0.4606 | 6 | - | - | - |
| 8 | Coupling12 | 0.9802 | 0.4741 | 4 | - | - | - |
| 9 | TopK10 | 0.9802 | 0.4735 | 4 | - | - | - |
| 10 | TailTopK3 | 0.9801 | 0.4748 | 4 | - | - | - |
| - | **60ep-lr2e4-dia4 (Baseline)** | 0.9793 | 0.4735 | 4 | - | - | - |

### 3.2 DIA Blocks 효과 분석

| DIA Blocks | Img AUC | Pix AP | Δ Img AUC | Δ Pix AP |
|:----------:|:-------:|:------:|:---------:|:--------:|
| 4 (baseline) | 0.9793 | 0.4735 | - | - |
| 6 | 0.9820 | 0.4606 | **+0.27%** | -1.29% |
| 7 | **0.9830** | 0.4580 | **+0.37%** | -1.55% |
| 8 | 0.9825 | 0.4546 | +0.32% | -1.89% |

**분석**: DIA blocks 증가 → Image AUC 향상, Pixel AP 감소 (Trade-off)

### 3.3 Tail-Aware Loss + TopK 효과 분석

| 설정 | Img AUC | Pix AP | Δ Img AUC | Δ Pix AP |
|:-----|:-------:|:------:|:---------:|:--------:|
| Baseline | 0.9793 | 0.4735 | - | - |
| TopK5-TailW0.5 | 0.9820 | **0.4866** | **+0.27%** | **+1.31%** |
| TopK7-TailW0.5 | 0.9821 | 0.4866 | **+0.28%** | **+1.31%** |
| TailTopK3 | 0.9801 | 0.4748 | +0.08% | +0.13% |

**분석**: TopK5-TailW0.5 조합이 두 메트릭 모두 향상

### 3.4 Ablation Study (모듈 제거)

| 제거 모듈 | Img AUC | Pix AP | Δ Img AUC | Δ Pix AP | 중요도 |
|:----------|:-------:|:------:|:---------:|:--------:|:------:|
| Full Model | 0.9793 | 0.4735 | - | - | - |
| wo_Router | 0.9798 | 0.4684 | +0.05% | -0.51% | ★☆☆☆☆ |
| wo_LoRA | 0.9797 | 0.4753 | +0.04% | +0.18% | ★☆☆☆☆ |

**주의**: 이 ablation 결과는 그룹 1 설정 기준입니다. MAIN 설정과 다릅니다.

---

## 4. 그룹 2: MAIN 설정 실험 결과

### 공통 하이퍼파라미터
```
lr = 3e-4
lambda_logdet = 1e-4
scale_context_kernel = 5
epochs = 60
coupling_layers = 8
lora_rank = 64
dia_blocks = 4 (기본값)
```

### 4.1 전체 결과 (Image AUC 기준 정렬)

| Rank | Experiment | Img AUC | Pix AP | DIA | 비고 |
|:----:|:-----------|:-------:|:------:|:---:|:-----|
| 1 | FullBest-80ep-lr3e-4-LoRA128-C10-DIA5 | **0.9836** | 0.5242 | 5 | Extended config |
| 2 | TailW0.7-TopK5-TailTopK3 | 0.9830 | 0.5404 | 4 | - |
| 3 | **TailW0.7-TopK3-TailTopK2 (MAIN)** | **0.9829** | **0.5420** | 4 | **Best balanced** |
| 4 | TailW0.65-TopK5-TailTopK1 | 0.9828 | 0.5430 | 4 | - |
| 5 | TailW0.7-TopK3-TailTopK3 | 0.9826 | 0.5404 | 4 | - |
| 6 | TailW0.65-TopK3-TailTopK3 | 0.9824 | 0.5395 | 4 | - |
| 7 | TailW0.55-TopK5-LogdetReg1e-4 | 0.9824 | 0.5350 | 4 | - |
| 8 | CL-1x1-Seed42 | 0.9822 | 0.5331 | 4 | CL scenario |
| 9 | Ultimate-Combo1 | 0.9821 | 0.5066 | 6 | - |

### 4.2 MAIN 설정 성능 요약

**MAIN Experiment**: `MVTec-WRN50-TailW0.7-TopK3-TailTopK2-ScaleK5-lr3e-4-MAIN`

| 메트릭 | 값 |
|:-------|:--:|
| Image AUC | **98.29%** |
| Pixel AUC | **97.82%** |
| Pixel AP | **54.20%** |
| Routing Accuracy | **100%** |

### 4.3 CL 시나리오별 성능 비교

| 시나리오 | Tasks | Img AUC | Pix AP | 분석 |
|:---------|:-----:|:-------:|:------:|:-----|
| **1x1 (MAIN)** | 15 | **0.9829** | **0.5420** | 최적 |
| 3x3 | 5 | 0.8383 | 0.3365 | Multi-class task 영향 |
| 5x5 | 3 | 0.8022 | 0.2640 | 성능 저하 |
| 14x1 | 2 | 0.6685 | 0.0806 | 극단적 불균형 |

**분석**: 1x1 시나리오(각 task에 1개 클래스)가 가장 우수한 성능

---

## 5. 그룹 간 비교 시 주의사항

### ⚠️ 공정한 비교를 위한 규칙

1. **그룹 1 실험끼리만 비교** (lr=2e-4, logdet=1e-5, scale_k=3)
2. **그룹 2 실험끼리만 비교** (lr=3e-4, logdet=1e-4, scale_k=5)
3. 그룹 간 비교 시에는 하이퍼파라미터 차이를 명시해야 함

### 두 그룹 설정 비교

| 설정 | 그룹 1 | 그룹 2 (MAIN) | 변화 |
|:-----|:------:|:-------------:|:-----|
| lr | 2e-4 | **3e-4** | +50% |
| lambda_logdet | 1e-5 | **1e-4** | +10x |
| scale_ctx_k | 3 | **5** | +67% |

### 최고 성능 비교

| 그룹 | Best Img AUC | Best Pix AP | 대표 실험 |
|:----:|:------------:|:-----------:|:----------|
| 그룹 1 | 0.9830 | 0.4866 | DIA7 / TopK5-TailW0.5 |
| 그룹 2 | **0.9836** | **0.5420** | FullBest / MAIN |

**결론**: 그룹 2 (MAIN 설정)이 전반적으로 더 우수한 성능

---

## 6. 핵심 발견 사항

### 6.1 하이퍼파라미터 효과 (그룹 1 기준)

| 하이퍼파라미터 | Img AUC 효과 | Pix AP 효과 | 권장 |
|:---------------|:------------:|:-----------:|:----:|
| DIA blocks ↑ | ↑ (+0.37%) | ↓ (-1.5%) | 6-7 |
| TopK5-TailW0.5 | ↑ (+0.27%) | ↑ (+1.3%) | ✓ |
| Coupling12 | ↑ (+0.09%) | ↑ (+0.06%) | ✓ |

### 6.2 MAIN 설정 권장 사항

```python
config = {
    'lr': 3e-4,
    'lambda_logdet': 1e-4,
    'scale_context_kernel': 5,
    'dia_n_blocks': 4,
    'tail_weight': 0.7,
    'score_aggregation_top_k': 3,
    'tail_top_k_ratio': 0.02,
    'num_epochs': 60,
    'num_coupling_layers': 8,
    'lora_rank': 64,
}
```

**예상 성능**: Image AUC ~ 98.3%, Pixel AP ~ 54.2%

---

## 7. Ablation Study (TODO)

> **Note**: 기존 ablation 실험들은 그룹 1 설정(lr=2e-4, logdet=1e-5, scale_k=3)으로 진행되었습니다.
> MAIN 설정과 공정한 비교를 위해 그룹 2 설정으로 재실험이 필요합니다.

자세한 내용은 `documents/ablation.md` 참조.

---

## 8. 데이터셋별 성능 요약

| Dataset | Classes | Best Img AUC | Best Pix AP | Routing Acc |
|:--------|:-------:|:------------:|:-----------:|:-----------:|
| MVTec AD | 15 | **0.9836** | **0.5420** | 100% |
| VISA | 12 | 0.8566 | 0.2878 | 100% |
| MPDD | 6 | 0.9019 | 0.2890 | 98.12% |

---

*Updated: 2026-01-05*
*Note: 모든 비교는 동일 하이퍼파라미터 그룹 내에서만 수행*
