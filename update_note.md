# MoLE-Flow Experiment Analysis Report

## Analysis Date: 2026-01-03

## Executive Summary

This document provides a comprehensive analysis of 78 experiments conducted to optimize MoLE-Flow performance on MVTec AD dataset. The focus is on achieving Pixel AP in the range of 0.54-0.60 while maintaining or improving Image AUC.

**Key Finding**: The best configuration achieved **Pixel AP = 0.5350** (vs baseline 0.4640, +15.3% improvement) with **Image AUC = 0.9824**.

---

## 1. Top 20 Experiments by Pixel AP

| Rank | Experiment Name | Image AUC | Pixel AUC | Pixel AP |
|------|-----------------|-----------|-----------|----------|
| 1 | TailW0.55-TopK5-LogdetReg1e-4-ScaleCtxK5-lr3e-4 | 0.9824 | 0.9778 | **0.5350** |
| 2 | TailW0.65-TailTopK3-TopK5-LogdetReg1e-4 | 0.9827 | 0.9776 | 0.5324 |
| 3 | TopK3-TailW0.5-LogdetReg1e-4-ScaleCtxK5 | 0.9802 | 0.9772 | 0.5317 |
| 4 | TopK5-TailW0.5-LogdetReg1e-4-ScaleCtxK5 | 0.9809 | 0.9772 | 0.5317 |
| 5 | TailW0.6-TailTopK3-TopK5-LogdetReg1e-4-ScaleCtxK5-80ep | 0.9826 | 0.9777 | 0.5310 |
| 6 | TailW0.6-TopK5-LogdetReg1e-4 | 0.9827 | 0.9773 | 0.5290 |
| 7 | TailW0.55-TopK5-LogdetReg1e-4 | 0.9827 | 0.9770 | 0.5256 |
| 8 | TailW0.5-TailTopK3-TopK5-LogdetReg1e-4 | 0.9830 | 0.9767 | 0.5242 |
| 9 | FullBest-80ep-lr3e-4-LoRA128-C10-DIA5-TailW0.55-TailTopK3-ScaleCtxK5 | **0.9836** | **0.9780** | 0.5242 |
| 10 | TopK5-TailW0.5-LogdetReg1e-4 | 0.9826 | 0.9767 | 0.5221 |
| 11 | TopK3-TailW0.5-LogdetReg1e-4 | 0.9818 | 0.9767 | 0.5221 |
| 12 | TopK7-TailW0.5-LogdetReg1e-4 | 0.9826 | 0.9767 | 0.5221 |
| 13 | TopK5-TailW0.5-LogdetReg1e-4-LoRA128 | 0.9825 | 0.9767 | 0.5221 |
| 14 | TopK5-TailW0.5-LogdetReg1e-4-lr3e-4 | 0.9836 | 0.9771 | 0.5216 |
| 15 | TopK5-TailW0.5-LogdetReg1e-4-80ep | 0.9830 | 0.9768 | 0.5204 |
| 16 | TailW0.5-TailTopK7-TopK5-LogdetReg1e-4 | 0.9822 | 0.9766 | 0.5204 |
| 17 | TopK5-TailW0.5-LogdetReg1e-4-ScaleCtxK7 | 0.9822 | 0.9768 | 0.5194 |
| 18 | TopK5-TailW0.5-LogdetReg1e-4-Coupling12 | 0.9828 | 0.9764 | 0.5186 |
| 19 | LogdetReg1e-4-ScaleCtxK5 | 0.9796 | 0.9760 | 0.5168 |
| 20 | TopK3-TailW0.55-LogdetReg1e-4-Coupling12-lr3e-4 | 0.9833 | 0.9769 | 0.5153 |

---

## 2. Baseline Performance

| Experiment | Image AUC | Pixel AUC | Pixel AP |
|------------|-----------|-----------|----------|
| MVTec-WRN50-60ep-lr2e4-dia4 | 0.9793 | 0.9736 | 0.4735 |
| MVTec-WRN50-80ep | 0.9796 | 0.9736 | 0.4640 |

---

## 3. Ablation Studies

| Ablation | Image AUC | Pixel AUC | Pixel AP | Impact |
|----------|-----------|-----------|----------|--------|
| wo_ScaleCtx | 0.9775 | 0.9741 | 0.4776 | Minor loss |
| wo_LoRA | 0.9797 | 0.9739 | 0.4753 | Minor loss |
| wo_Router | 0.9798 | 0.9734 | 0.4684 | Minor loss |
| wo_SpatialCtx | 0.9772 | 0.9731 | 0.4659 | Moderate loss |
| wo_DIA | **0.9479** | 0.9702 | 0.4586 | **Significant ImgAUC drop** |
| wo_PosEmbed | 0.9767 | 0.9695 | 0.4564 | Moderate loss |
| wo_Adapter | **0.9604** | 0.9703 | 0.4461 | **Significant ImgAUC drop** |

**Key Insight**: DIA (Dense Input Adapter) and TaskInputAdapter are critical for Image AUC.

---

## 4. Hyperparameter Effect Analysis

### 4.1 Individual Component Effects (vs Baseline 0.4640)

| Component | Pixel AP | Delta |
|-----------|----------|-------|
| LogdetReg1e-4 | 0.5055 | **+0.0415** |
| ScaleCtxK5 | 0.4870 | +0.0230 |
| TopK5-TailW0.5 | 0.4866 | +0.0226 |
| lr3e-4 | 0.4718 | +0.0078 |
| DIA6 | 0.4606 | -0.0034 |

### 4.2 LogdetReg Effect
Log-determinant regularization with weight 1e-4 provides the **single largest improvement** (+4.15%).

| LogdetReg Weight | Pixel AP |
|------------------|----------|
| 1e-6 | 0.4700 |
| 1e-5 | (not tested) |
| 1e-4 | **0.5055** |

### 4.3 TailW (Tail Weight) Effect
Higher tail loss weights improve pixel-level localization:

| TailW | Pixel AP (with LogdetReg+TopK5) |
|-------|--------------------------------|
| 0.5 | 0.5221 |
| 0.55 | 0.5256 |
| 0.6 | 0.5290 |
| 0.65 | **0.5324** |

### 4.4 ScaleCtxK Effect
Scale context aggregation helps significantly:

| ScaleCtxK | Pixel AP |
|-----------|----------|
| None | 0.5221 |
| K=5 | **0.5317** |
| K=7 | 0.5194 |

K=5 is optimal; K=7 slightly worse.

### 4.5 DIA (Dense Input Adapter) Effect
Higher DIA values improve Image AUC but may hurt Pixel AP:

| DIA | Image AUC | Pixel AP |
|-----|-----------|----------|
| 2 | 0.9726 | 0.4845 |
| 4 | 0.9793 | 0.4735 |
| 6 | **0.9820** | 0.4606 |
| 7 | **0.9830** | 0.4580 |
| 8 | 0.9825 | 0.4546 |

### 4.6 LoRA Rank Effect
LoRA rank has minimal impact on performance:

| LoRA Rank | Image AUC | Pixel AP |
|-----------|-----------|----------|
| 32 | 0.9794 | 0.4737 |
| 64 (default) | 0.9793 | 0.4735 |
| 128 | 0.9794 | 0.4736 |
| 256 | 0.9796 | 0.4741 |

### 4.7 Coupling Layers Effect

| Coupling Layers | Image AUC | Pixel AP | Notes |
|-----------------|-----------|----------|-------|
| 10 (default) | 0.9796 | 0.4640 | Stable |
| 12 | 0.9802 | 0.4741 | Slightly better |
| 16 | **0.7341** | **0.2284** | **FAILED** - Training instability |

**Warning**: Coupling16 causes severe training instability.

---

## 5. Combination Synergies

| Combination | Pixel AP | Improvement |
|-------------|----------|-------------|
| Baseline | 0.4640 | - |
| + LogdetReg1e-4 | 0.5055 | +0.0415 |
| + TopK5 + TailW0.5 | 0.5221 | +0.0581 |
| + ScaleCtxK5 | 0.5317 | +0.0677 |
| + TailW0.55 + lr3e-4 | **0.5350** | **+0.0710** |

---

## 6. Per-Class Performance (Top Config vs Baseline)

| Class | Baseline | Top Config | Improvement |
|-------|----------|------------|-------------|
| carpet | 0.3601 | 0.6167 | **+0.2566** |
| bottle | 0.4551 | 0.6774 | **+0.2223** |
| leather | 0.2292 | 0.3970 | **+0.1678** |
| toothbrush | 0.4028 | 0.5619 | +0.1591 |
| wood | 0.3546 | 0.4453 | +0.0907 |
| hazelnut | 0.5110 | 0.5798 | +0.0688 |
| capsule | 0.3400 | 0.3940 | +0.0540 |
| zipper | 0.2948 | 0.3481 | +0.0533 |
| grid | 0.2051 | 0.2536 | +0.0485 |
| tile | 0.6409 | 0.6673 | +0.0264 |
| screw | 0.2009 | 0.2212 | +0.0203 |
| pill | 0.8035 | 0.8077 | +0.0042 |
| transistor | 0.6561 | 0.6442 | -0.0119 |
| cable | 0.6575 | 0.6339 | -0.0236 |
| metal_nut | 0.8491 | 0.7776 | **-0.0715** |
| **Mean** | **0.4640** | **0.5350** | **+0.0710** |

**Key Observations**:
- Textured classes (carpet, leather) benefit most
- Object-with-boundary classes (bottle, toothbrush) show large gains
- Some fine-grained classes (metal_nut, cable) show slight regression

---

## 7. Recommendations

### 7.1 Optimal Configuration for Balanced Performance
```bash
python run_moleflow.py \
    --tail_weight 0.55 \
    --topk 5 \
    --logdet_reg 1e-4 \
    --scale_context_k 5 \
    --learning_rate 3e-4 \
    --num_epochs 60 \
    --experiment_name optimal_balanced
```
**Expected**: Image AUC ~0.982, Pixel AP ~0.535

### 7.2 Configuration for Maximum Image AUC
```bash
python run_moleflow.py \
    --tail_weight 0.55 \
    --topk 5 \
    --logdet_reg 1e-4 \
    --learning_rate 3e-4 \
    --lora_rank 128 \
    --num_coupling_layers 10 \
    --dia 5 \
    --num_epochs 80 \
    --experiment_name max_img_auc
```
**Expected**: Image AUC ~0.984, Pixel AP ~0.524

### 7.3 To Reach 0.54+ Pixel AP (Recommended Next Experiments)

1. **Higher TailW exploration**:
```bash
python run_moleflow.py \
    --tail_weight 0.7 \
    --topk 5 \
    --logdet_reg 1e-4 \
    --scale_context_k 5 \
    --learning_rate 3e-4 \
    --experiment_name tailw0.7_exploration
```

2. **Stronger LogdetReg**:
```bash
python run_moleflow.py \
    --tail_weight 0.55 \
    --topk 5 \
    --logdet_reg 5e-4 \
    --scale_context_k 5 \
    --learning_rate 3e-4 \
    --experiment_name logdet5e-4_exploration
```

3. **Combined with longer training**:
```bash
python run_moleflow.py \
    --tail_weight 0.6 \
    --topk 5 \
    --logdet_reg 1e-4 \
    --scale_context_k 5 \
    --learning_rate 3e-4 \
    --num_epochs 100 \
    --experiment_name extended_training
```

---

## 8. Conclusions

1. **Best Overall Configuration**: TailW0.55 + TopK5 + LogdetReg1e-4 + ScaleCtxK5 + lr3e-4
   - Pixel AP: 0.5350 (target range: 0.54-0.60)
   - Image AUC: 0.9824 (maintains high performance)

2. **Critical Components**:
   - LogdetReg1e-4: Most impactful single hyperparameter
   - ScaleCtxK5: Important for pixel-level localization
   - TailW (0.55-0.65): Helps focus on difficult pixels

3. **Avoid**:
   - Coupling16: Causes training instability
   - High DIA (>6) without other optimizations: May hurt Pixel AP

4. **Trade-offs**:
   - Higher DIA improves Image AUC but may reduce Pixel AP
   - LoRA rank changes have minimal effect
   - TailW > 0.65 needs more exploration

5. **Gap to Target**:
   - Current best: 0.5350
   - Target: 0.54-0.60
   - Gap: 0.005-0.065
   - Status: Very close to lower target bound

---

*Report generated automatically from experiment results in /Volume/MoLeFlow/logs/Final/*

---

## 9. 추가 분석: Pixel AP 0.6 달성 전략 (2026-01-03 업데이트)

### 9.1 새로운 최고 성능 발견

| Rank | Experiment | Image AUC | Pixel AP | 핵심 차이점 |
|------|------------|-----------|----------|-------------|
| **1** | **TailW0.8-TopK5-TailTopK3-ScaleK5** | 0.9811 | **0.5447** | tail_weight=0.8, lr=2e-4 |
| 2 | TailW0.65-TopK5-TailTopK1-ScaleK5-lr3e-4 | 0.9828 | 0.5430 | tail_weight=0.65 |
| 3 | TailW0.7-TopK5-TailTopK3-ScaleK5-lr3e-4 | 0.9830 | 0.5404 | tail_weight=0.7 |

**핵심 발견**: tail_weight 0.8에서 Pixel AP가 0.5447로 향상되었지만 Image AUC가 0.9811로 약간 하락함.

### 9.2 하이퍼파라미터 영향도 순위

1. **tail_weight** (가장 중요): 0.65-0.8 범위에서 최고 성능
2. **logdet_reg**: 1e-4가 기본, 2e-4도 효과적
3. **scale_context_k**: K=5가 최적
4. **topk**: 5가 최적 (3-7 범위 양호)
5. **learning_rate**: 3e-4가 Image AUC 유지에 좋음
6. **dia_n_blocks**: 4-5가 균형 잡힌 선택

### 9.3 미시도 조합 및 권장 실험

**시도하지 않은 조합**:
- TailW 0.9, 1.0
- TailW 0.8 + lr=3e-4
- LogdetReg 3e-4
- TailW 0.85 (0.8과 0.9 사이)

**권장 실험**:
```bash
# 실험 1: TailW 0.85
python run_moleflow.py --tail_weight 0.85 --topk 5 --logdet_reg 1e-4 \
    --scale_context_k 5 --learning_rate 2e-4 --experiment_name TailW0.85

# 실험 2: TailW 0.8 + lr 3e-4
python run_moleflow.py --tail_weight 0.8 --topk 5 --logdet_reg 1e-4 \
    --scale_context_k 5 --learning_rate 3e-4 --experiment_name TailW0.8-lr3e-4

# 실험 3: LogdetReg 3e-4
python run_moleflow.py --tail_weight 0.7 --topk 5 --logdet_reg 3e-4 \
    --scale_context_k 5 --learning_rate 3e-4 --experiment_name LogdetReg3e-4
```

### 9.4 Pixel AP 0.6 달성 가능성

**현재**: 0.5447 (TailW0.8)
**목표**: 0.6
**갭**: 0.0553 (약 10% 추가 개선 필요)

**병목 클래스** (TailW0.8 기준):
- screw: 0.2105 (가장 어려움 - rotation variance)
- grid: 0.2660 (큰 개선 필요)
- zipper: 0.3513
- capsule: 0.3920
- leather: 0.4458

**결론**: 0.6 달성은 도전적이지만 가능할 수 있음. 병목 클래스 특화 전략 필요.

---

## 10. VisA 데이터셋 실험 분석 (2026-01-03)

### 10.1 VisA 데이터셋 개요

VisA (Visual Anomaly) 데이터셋은 MVTec-AD보다 더 다양하고 도전적인 산업 이상 탐지 벤치마크입니다.

| 특성 | MVTec-AD | VisA |
|------|----------|------|
| 클래스 수 | 15 | 12 |
| 결함 유형 | 단순 | 복잡/다양 |
| 이미지 크기 | 다양 (700-1024) | 다양 (>1000) |
| 주요 카테고리 | 텍스처/객체 | PCB, 식품, 공구 |

**VisA 12개 클래스**:
- PCB 계열: pcb1, pcb2, pcb3, pcb4 (복잡한 회로 결함)
- 식품 계열: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pipe_fryum

### 10.2 VisA 실험 결과 비교

| 실험명 | Backbone | Epochs | LoRA | DIA | lr | Image AUC | Pixel AUC | Pixel AP |
|--------|----------|--------|------|-----|-----|-----------|-----------|----------|
| **VISA-ViT-60ep** | ViT-Base | 60 | 64 | 2 | 1e-4 | **0.8801** | 0.9440 | 0.1982 |
| VISA-WRN50-60ep-lr2e4-dia4 | WRN50 | 60 | 64 | 4 | 2e-4 | 0.8378 | **0.9715** | **0.2878** |
| VISA-WRN50-80ep-lr3e4 | WRN50 | 80 | 64 | 4 | 3e-4 | 0.8272 | 0.9665 | 0.2698 |
| VISA-WRN50-DIA6-80ep | WRN50 | 80 | 64 | 6 | 2e-4 | 0.8376 | 0.9687 | 0.2750 |
| VISA-WRN50-LoRA128-80ep | WRN50 | 80 | 128 | 4 | 2e-4 | 0.8202 | 0.9634 | 0.2571 |
| **VISA-WRN50-LoRA128-DIA6** | WRN50 | 80 | 128 | 6 | 2e-4 | 0.8566 | 0.9687 | 0.2761 |

### 10.3 핵심 발견 (VisA vs MVTec)

#### Image AUC 분석
| 조건 | MVTec Image AUC | VisA Image AUC | 차이 |
|------|-----------------|----------------|------|
| 최고 성능 | 0.9836 | 0.8801 | -0.1035 |
| WRN50 기본 | 0.9793 | 0.8378 | -0.1415 |

**관찰**: VisA가 MVTec보다 Image-level 탐지에서 **10-14% 낮은 성능**을 보임.

#### Backbone 비교 (VisA)
| Backbone | Image AUC | Pixel AUC | Pixel AP |
|----------|-----------|-----------|----------|
| ViT-Base | **0.8801** | 0.9440 | 0.1982 |
| WideResNet50 | 0.8378 | **0.9715** | **0.2878** |

**결론**:
- **ViT-Base**: Image-level AUC에서 +4.2% 우수
- **WideResNet50**: Pixel-level 성능에서 압도적 우수 (Pixel AP +9.0%)

### 10.4 클래스별 성능 분석 (VISA-WRN50-LoRA128-DIA6)

| 클래스 | Image AUC | Pixel AUC | Pixel AP | 난이도 |
|--------|-----------|-----------|----------|--------|
| cashew | 0.8686 | 0.9759 | 0.4405 | 쉬움 |
| pipe_fryum | 0.9662 | 0.9860 | 0.5122 | 쉬움 |
| chewinggum | 0.9574 | 0.9868 | 0.3153 | 쉬움 |
| fryum | 0.9598 | 0.9479 | 0.4316 | 중간 |
| pcb4 | 0.9681 | 0.9722 | 0.2512 | 중간 |
| pcb1 | 0.8693 | 0.9880 | 0.5551 | 중간 |
| candle | 0.8619 | 0.9858 | 0.1622 | 어려움 |
| capsules | 0.7077 | 0.9380 | 0.2156 | 어려움 |
| pcb2 | 0.8066 | 0.9445 | 0.0916 | 어려움 |
| pcb3 | 0.8187 | 0.9790 | 0.2695 | 어려움 |
| macaroni1 | 0.7866 | 0.9656 | 0.0597 | **매우 어려움** |
| macaroni2 | 0.7078 | 0.9552 | 0.0091 | **매우 어려움** |

**병목 클래스**:
1. **macaroni2**: Image AUC 0.71, Pixel AP 0.01 (매우 어려움)
2. **macaroni1**: Image AUC 0.79, Pixel AP 0.06
3. **capsules**: Image AUC 0.71

### 10.5 MVTec에서 VisA로의 하이퍼파라미터 이전

| 하이퍼파라미터 | MVTec 최적값 | VisA 테스트 결과 | 권장 방향 |
|----------------|--------------|------------------|-----------|
| **lora_rank** | 64 | 128이 약간 우수 | **128 권장** |
| **dia_n_blocks** | 4-5 | 6이 Image AUC 향상 | **6 권장** |
| **lr** | 3e-4 | 2e-4가 더 안정적 | **2e-4 권장** |
| **num_epochs** | 60-80 | 80이 약간 우수 | **80 권장** |
| **backbone** | WRN50 | WRN50 (Pixel) vs ViT (Image) | **WRN50** (Pixel 중시) |

### 10.6 VisA 최적 설정 권장안

#### 최적 Configuration (Pixel AP 중시)
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --experiment_name "VISA-Optimal-WRN50" \
    --backbone_name wide_resnet50_2 \
    --lora_rank 128 \
    --dia_n_blocks 6 \
    --lr 2e-4 \
    --num_epochs 80 \
    --use_tail_aware_loss \
    --tail_weight 0.5 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 5 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ./logs/Final
```
**예상 성능**: Image AUC ~0.86, Pixel AP ~0.30

#### Image AUC 중시 Configuration
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --experiment_name "VISA-ImageFocus-ViT" \
    --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --lora_rank 64 \
    --dia_n_blocks 4 \
    --lr 1e-4 \
    --num_epochs 60 \
    --log_dir ./logs/Final
```
**예상 성능**: Image AUC ~0.88, Pixel AP ~0.20

### 10.7 VisA 성능 개선을 위한 추가 실험 권장

#### 실험 1: Tail-Aware Loss 적용 (미시도)
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --experiment_name "VISA-WRN50-TailW0.7-TopK5-DIA6" \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 5 \
    --lora_rank 128 \
    --dia_n_blocks 6 \
    --lr 2e-4 \
    --num_epochs 80 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ./logs/Final
```

#### 실험 2: LogdetReg 증가
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --experiment_name "VISA-WRN50-LogdetReg2e-4-DIA6" \
    --lora_rank 128 \
    --dia_n_blocks 6 \
    --lr 2e-4 \
    --num_epochs 80 \
    --lambda_logdet 2e-4 \
    --scale_context_kernel 5 \
    --log_dir ./logs/Final
```

#### 실험 3: ViT + Tail-Aware
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --experiment_name "VISA-ViT-TailW0.6-DIA4" \
    --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --use_tail_aware_loss \
    --tail_weight 0.6 \
    --tail_top_k_ratio 0.02 \
    --lora_rank 64 \
    --dia_n_blocks 4 \
    --lr 1e-4 \
    --num_epochs 80 \
    --log_dir ./logs/Final
```

### 10.8 VisA 데이터셋 특성에 따른 인사이트

1. **PCB 클래스 (pcb1-4)**:
   - 복잡한 회로 패턴으로 인해 위치 정보가 중요
   - DIA 증가가 효과적
   - Pixel AP 0.09-0.55로 큰 편차

2. **식품 클래스 (macaroni, fryum 등)**:
   - 불규칙한 형태로 인해 position encoding 영향 적음
   - macaroni 계열이 특히 어려움 (texture variation)
   - Tail-Aware Loss가 도움될 가능성 높음

3. **Backbone 선택**:
   - **Pixel-level 중시**: WideResNet50 (multi-scale feature)
   - **Image-level 중시**: ViT-Base (global attention)

4. **MVTec에서 전이 가능한 인사이트**:
   - tail_weight 0.5-0.7이 유효할 것으로 예상
   - scale_context_kernel 5 유지
   - logdet_reg 1e-4 ~ 2e-4

---

## 11. Pixel AP 0.6+ 달성을 위한 상세 하이퍼파라미터 최적화 (2026-01-03)

### 10.1 최신 실험 결과 반영

| 순위 | 실험명 | Image AUC | Pixel AP | 핵심 변경 |
|------|--------|-----------|----------|-----------|
| 1 | TailW0.75-TopK5-TailTopK2-ScaleK5 | 0.9812 | **0.5449** | 신규 최고 |
| 2 | TailW0.8-TopK5-TailTopK3-ScaleK5 | 0.9811 | 0.5447 | TailW 증가 |
| 3 | TailW0.65-TopK5-TailTopK1-ScaleK5-lr3e-4 | 0.9828 | 0.5430 | TailTopK1 |
| 4 | TailW0.7-TopK3-TailTopK2-ScaleK5-lr3e-4 | 0.9829 | 0.5420 | TopK3 |
| 5 | TailW0.7-TopK5-TailTopK3-ScaleK5-lr3e-4 | 0.9830 | 0.5404 | 균형 |
| 6 | TailW0.55-TopK5-LogdetReg2e-4-ScaleK5-lr3e-4 | 0.9815 | 0.5399 | LogdetReg2e-4 |
| 7 | TailW0.65-TopK3-TailTopK3-ScaleK5-lr3e-4 | 0.9824 | 0.5395 | TopK3 |
| 8 | TailW0.55-TopK5-LogdetReg1e-4-ScaleCtxK5-lr3e-4 | 0.9824 | 0.5350 | 이전 최고 |

### 10.2 핵심 발견

#### TailWeight 효과 (가장 중요)
| TailW | 최고 Pixel AP | Image AUC 범위 | 최적 TailTopK |
|-------|---------------|----------------|---------------|
| 0.55 | 0.5350 | 0.982-0.984 | 5% |
| 0.65 | 0.5430 | 0.982-0.983 | 1% |
| 0.7 | 0.5420 | 0.983 | 2-3% |
| 0.75 | **0.5449** | 0.981 | 2% |
| 0.8 | 0.5447 | 0.981 | 3% |

**결론**: TailW 0.75-0.8에서 Pixel AP 최대, Image AUC 0.981대로 유지

#### TailTopK Ratio 효과
| TailTopK | 효과 |
|----------|------|
| 1% | 가장 집중된 학습, TailW 0.65와 조합시 우수 |
| 2% | 최적 범위, 안정적 |
| 3% | 기본값, 안정적 |
| 7% | 과도, 성능 감소 |

### 10.3 Pixel AP 0.6+ 달성을 위한 권장 실험 조합

#### 1순위: TailW0.85 + TailTopK2% (가장 유망)
```bash
python run_moleflow.py \
    --experiment_name "MVTec-WRN50-TailW0.85-TopK5-TailTopK2-ScaleK5-LogdetReg2e-4-lr3e-4" \
    --use_tail_aware_loss \
    --tail_weight 0.85 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k_percent \
    --score_aggregation_top_k_percent 0.05 \
    --lambda_logdet 2e-4 \
    --scale_context_kernel 5 \
    --lr 3e-4 \
    --num_epochs 60 \
    --dia_n_blocks 4 \
    --log_dir ./logs/Final
```
**예상**: Pixel AP 0.555-0.57, Image AUC ~0.981

#### 2순위: TailW0.9 + 80ep (공격적)
```bash
python run_moleflow.py \
    --experiment_name "MVTec-WRN50-TailW0.9-TopK5-TailTopK2-ScaleK5-LogdetReg2e-4-80ep" \
    --use_tail_aware_loss \
    --tail_weight 0.9 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k_percent \
    --score_aggregation_top_k_percent 0.05 \
    --lambda_logdet 2e-4 \
    --scale_context_kernel 5 \
    --lr 3e-4 \
    --num_epochs 80 \
    --dia_n_blocks 4 \
    --log_dir ./logs/Final
```
**예상**: Pixel AP 0.56-0.58, Image AUC ~0.978

#### 3순위: TailW0.8 + lr3e-4 (균형)
```bash
python run_moleflow.py \
    --experiment_name "MVTec-WRN50-TailW0.8-TopK5-TailTopK2-ScaleK5-lr3e-4" \
    --use_tail_aware_loss \
    --tail_weight 0.8 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k_percent \
    --score_aggregation_top_k_percent 0.05 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --lr 3e-4 \
    --num_epochs 60 \
    --dia_n_blocks 4 \
    --log_dir ./logs/Final
```
**예상**: Pixel AP 0.55-0.56, Image AUC ~0.982

#### 4순위: TopK3 + TailW0.8 (대안)
```bash
python run_moleflow.py \
    --experiment_name "MVTec-WRN50-TailW0.8-TopK3-TailTopK1-ScaleK5-lr3e-4" \
    --use_tail_aware_loss \
    --tail_weight 0.8 \
    --tail_top_k_ratio 0.01 \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 6 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --lr 3e-4 \
    --num_epochs 60 \
    --dia_n_blocks 4 \
    --log_dir ./logs/Final
```
**예상**: Pixel AP 0.54-0.56, Image AUC ~0.982

#### 5순위: Coupling12 + 최적 설정
```bash
python run_moleflow.py \
    --experiment_name "MVTec-WRN50-TailW0.8-TopK5-TailTopK2-ScaleK5-C12-lr3e-4" \
    --use_tail_aware_loss \
    --tail_weight 0.8 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k_percent \
    --score_aggregation_top_k_percent 0.05 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --num_coupling_layers 12 \
    --lr 3e-4 \
    --num_epochs 60 \
    --dia_n_blocks 4 \
    --log_dir ./logs/Final
```
**예상**: Pixel AP 0.54-0.56, Image AUC ~0.982

### 10.4 최적 하이퍼파라미터 범위 요약

| 파라미터 | 권장 범위 | 최적값 | 근거 |
|----------|-----------|--------|------|
| tail_weight | 0.75-0.9 | 0.85 | 0.75-0.8에서 0.5449 달성 |
| tail_top_k_ratio | 0.01-0.02 | 0.02 | 집중된 학습 |
| logdet_reg | 1e-4 ~ 2e-4 | 1e-4 | 5e-4는 성능 저하 |
| scale_context_kernel | 5 | 5 | K=7은 과도 |
| learning_rate | 2e-4 ~ 3e-4 | 3e-4 | Image AUC 유지 |
| num_epochs | 60-80 | 60 | 80ep는 marginal gain |
| dia_n_blocks | 4-6 | 4 | 안정성 |
| num_coupling_layers | 8-12 | 10 | 16은 불안정 |

### 10.5 0.6 목표에 대한 현실적 평가

| 현황 | 값 |
|------|-----|
| 현재 최고 | 0.5449 (TailW0.75) |
| 예상 최대 (공격적) | 0.56-0.58 |
| 목표 | 0.6 |
| 갭 | 0.04-0.06 |

**0.6 달성을 위한 추가 방안**:
1. **Image size 448**: 해상도 증가로 세밀한 anomaly 탐지
2. **ViT backbone**: DINOv2 ViT-L 등 강력한 특징 추출기
3. **Multi-scale 평가**: 여러 해상도에서 앙상블
4. **Class-specific 튜닝**: 병목 클래스별 최적 설정

---

## 12. VisA 데이터셋 하이퍼파라미터 최적화 분석 (2026-01-03)

### 12.1 목표 및 현재 상태

**목표**: Image AUC >= 0.95, Pixel AP >= 0.4
**현재 최고**: Image AUC = 0.8566, Pixel AP = 0.2878
**필요 개선**: Image AUC +0.09, Pixel AP +0.11 이상

### 12.2 VisA 실험 결과 요약

| 실험명 | Image AUC | Pixel AP | 주요 설정 |
|--------|-----------|----------|-----------|
| **VISA-WRN50-LoRA128-DIA6-Combined** | **0.8566** | 0.2761 | LoRA128, DIA6, lr=2e-4, 80ep |
| VISA-WRN50-60ep-lr2e4-dia4 | 0.8378 | **0.2878** | LoRA64, DIA4, lr=2e-4, 60ep |
| VISA-WRN50-DIA6-80ep | 0.8376 | 0.2750 | LoRA64, DIA6, lr=2e-4, 80ep |
| VISA-WRN50-80ep-lr3e4 | 0.8272 | 0.2698 | LoRA64, DIA4, lr=3e-4, 80ep |
| VISA-WRN50-LoRA128-80ep | 0.8202 | 0.2571 | LoRA128, DIA4, 80ep |
| VISA-ViT-60ep | 0.8801 | 0.1982 | ViT backbone, DIA2, lr=1e-4 |

### 12.3 VisA 병목 클래스 분석

**Pixel AP 낮은 클래스 (개선 필요)**:
- macaroni2: 0.0078 (극히 낮음)
- macaroni1: 0.0552
- pcb2: 0.0916
- candle: 0.1621

**Pixel AP 높은 클래스 (참조)**:
- pcb1: 0.5551-0.6797
- pipe_fryum: 0.5055-0.5229
- cashew: 0.4405-0.4962

### 12.4 MVTec 인사이트 전이

**MVTec에서 미적용된 핵심 요소**:
1. Tail-Aware Loss (use_tail_aware_loss) - 미적용
2. lambda_logdet 1e-4 (현재 1e-5) - 10배 증가 필요
3. scale_context_kernel 5 (현재 3) - 확장 필요

**MVTec 최적 설정에서 발견한 핵심 효과**:
| 파라미터 | 효과 (Pixel AP 개선) |
|----------|----------------------|
| lambda_logdet 1e-4 | **+4.15%** (가장 큰 효과) |
| tail_weight 0.75-0.8 | +3-5% |
| scale_context_kernel 5 | +2-3% |
| tail_top_k_ratio 0.02 | +1-2% |

### 12.5 VisA 최적 설정 제안 (우선순위)

#### 1순위: MVTec 최적 설정 전이 + VisA 적응
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --task_classes candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum \
    --experiment_name "VISA-Optimized-TailW0.8-TopK5-TailTopK2-ScaleK5-LogdetReg1e-4-lr3e-4" \
    --backbone_name wide_resnet50_2 \
    --num_epochs 80 \
    --lr 3e-4 \
    --lora_rank 128 \
    --num_coupling_layers 10 \
    --dia_n_blocks 5 \
    --use_tail_aware_loss \
    --tail_weight 0.8 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k_percent \
    --score_aggregation_top_k_percent 0.05 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ./logs/Final
```
**예상 성능**: Image AUC 0.88-0.91, Pixel AP 0.32-0.38

#### 2순위: DIA 강화 + 안정적 lr
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --task_classes candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum \
    --experiment_name "VISA-DIA7-TailW0.75-TopK5-LogdetReg1e-4-C10-lr2e-4" \
    --backbone_name wide_resnet50_2 \
    --num_epochs 80 \
    --lr 2e-4 \
    --lora_rank 128 \
    --num_coupling_layers 10 \
    --dia_n_blocks 7 \
    --use_tail_aware_loss \
    --tail_weight 0.75 \
    --tail_top_k_ratio 0.02 \
    --score_aggregation_mode top_k_percent \
    --score_aggregation_top_k_percent 0.05 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ./logs/Final
```
**예상 성능**: Image AUC 0.87-0.90, Pixel AP 0.30-0.35

#### 3순위: ViT Backbone + Tail-Aware Loss
```bash
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --task_classes candle capsules cashew chewinggum fryum macaroni1 macaroni2 pcb1 pcb2 pcb3 pcb4 pipe_fryum \
    --experiment_name "VISA-ViT-TailW0.7-TopK5-LogdetReg1e-4-ScaleK5-DIA4-80ep" \
    --backbone_name vit_base_patch16_224.augreg2_in21k_ft_in1k \
    --num_epochs 80 \
    --lr 1e-4 \
    --lora_rank 64 \
    --num_coupling_layers 8 \
    --dia_n_blocks 4 \
    --use_tail_aware_loss \
    --tail_weight 0.7 \
    --tail_top_k_ratio 0.03 \
    --score_aggregation_mode top_k_percent \
    --score_aggregation_top_k_percent 0.05 \
    --lambda_logdet 1e-4 \
    --scale_context_kernel 5 \
    --log_dir ./logs/Final
```
**예상 성능**: Image AUC 0.89-0.92, Pixel AP 0.28-0.35

### 12.6 목표 달성 가능성 평가

| 목표 | 현재 최고 | 예상 최대 | 달성 가능성 |
|------|-----------|-----------|-------------|
| Image AUC >= 0.95 | 0.8566 | 0.90-0.92 | **낮음** |
| Pixel AP >= 0.4 | 0.2878 | 0.35-0.40 | **중간** |

### 12.7 목표 달성을 위한 추가 방안

**Image AUC 0.95+ 달성 방안**:
1. img_size 448 (해상도 2배)
2. 더 강력한 backbone (DINOv2 ViT-L/H)
3. 모델 앙상블
4. 병목 클래스 특화 전략 (macaroni1/2, capsules)

**Pixel AP 0.4+ 달성 방안**:
1. lambda_logdet 2e-4 또는 3e-4
2. tail_weight 0.9+ (공격적 tail 학습)
3. Multi-scale 평가 앙상블
4. 병목 클래스별 특화 설정

### 12.8 권장 실험 순서

1. **1순위 설정** 먼저 실행 (MVTec 최적 설정 전이)
2. 결과에 따라:
   - Image AUC < 0.87 → 2순위(DIA 강화) 시도
   - Pixel AP < 0.30 → tail_weight 0.85-0.9로 증가
   - 둘 다 낮음 → 3순위(ViT backbone) 시도
3. 병목 클래스(macaroni1/2) 분석 후 클래스별 전략 수립

---
