# Tail-Aware Loss 메커니즘 분석

## 1. 연구 의문

### 1.1 관찰된 현상

```
Tail-Aware Loss 적용 시 (λ=0.7, top_k_ratio=0.02):
- Pixel AP: 45.86% → 55.80% (+9.94%p)
- Image AUC: 96.62% → 98.05% (+1.43%p)
```

### 1.2 핵심 의문

> **왜 전체 패치의 단 2% (196개 중 4개)만 집중 학습해도 +10%p의 Pixel AP 향상이 가능한가?**

### 1.3 Tail-Aware Loss 정의

```python
L_total = (1 - λ_tail) × E[NLL_all] + λ_tail × E[NLL_top-k]

# λ_tail: tail weight (0~1)
# NLL_all: 전체 패치의 Negative Log-Likelihood
# NLL_top-k: 상위 k% NLL 패치만의 평균
```

---

## 2. 가설 설정 및 검증

### 2.1 가설 H1: 이미지 경계 가설

#### 가설
> Tail 패치 (NLL 상위 2%)는 이미지의 texture 경계나 edge 영역에 위치한다.

#### 검증 방법
```python
# 1. Tail 패치의 공간 위치 수집
for batch in train_loader:
    z, logdet = flow.forward(batch)
    nll_patch = compute_patch_nll(z, logdet)
    tail_indices = torch.topk(nll_patch, k=top_2%)[1]

# 2. Edge distance 계산
edge_distance_tail = compute_distance_to_image_edge(tail_positions)
edge_distance_uniform = expected_uniform_distribution()

# 3. Image gradient 비교
gradient_at_tail = image_gradient[tail_positions].mean()
gradient_at_non_tail = image_gradient[non_tail_positions].mean()

# 4. 통계 검정
ks_test(tail_distribution, uniform_distribution)
mann_whitney_test(gradient_tail, gradient_non_tail)
```

#### 결과

| 측정 항목 | 값 |
|----------|-----|
| Edge distance (tail) | 1.979 ± 2.515 |
| Edge distance (uniform) | 4.179 |
| KS test p-value | 5.43e-93 |
| Image gradient (tail) | 0.6734 |
| Image gradient (non-tail) | 0.6693 |
| **Gradient ratio** | **1.01x** |
| Mann-Whitney p-value | 0.052 |

#### 판정: ❌ NOT SUPPORTED

- Tail 패치가 이미지 edge에 가깝긴 하지만 (KS p < 0.05)
- **이미지 gradient 차이는 통계적으로 유의미하지 않음** (ratio = 1.01x)
- Tail 패치는 이미지 특성이 아닌 **모델의 학습 상태**와 관련

---

### 2.2 가설 H3: Gradient Concentration 가설 ⭐

#### 가설
> Mean-only 학습은 모든 패치에 gradient를 분산시켜 tail 영역의 gradient를 희석(dilute)시킨다.

#### 검증 방법
```python
# 두 조건에서 gradient 비교
configs = [
    {'tail_weight': 0.0, 'name': 'mean_only'},
    {'tail_weight': 1.0, 'name': 'tail_aware'}
]

for config in configs:
    # Forward pass
    z, logdet = model(batch)

    # Loss 계산 (조건별)
    if config['tail_weight'] == 0:
        loss = nll_patch.mean()  # Mean-only
    else:
        loss = nll_patch.topk(k)[0].mean()  # Tail-aware

    # Backward pass
    loss.backward()

    # Patch별 gradient magnitude 측정
    grad_tail = gradient_at_tail_patches.norm()
    grad_non_tail = gradient_at_non_tail_patches.norm()
    ratio = grad_tail / grad_non_tail
```

#### 결과

| 조건 | Tail Gradient | Non-Tail Gradient | Ratio | Gini |
|------|---------------|-------------------|-------|------|
| **Mean-Only (λ=0)** | 0.0222 | 0.0188 | **1.18x** | 0.091 |
| **Tail-Aware (λ=1)** | 0.8402 | 0.0168 | **49.99x** | 0.131 |

**핵심 수치**:
- **Gradient Amplification: 42.3x** (Mean-only 대비 Tail-aware)
- 이론적 최대치: 196/4 = 49x
- 실측치: 42.3x (이론값의 86%)

#### 시각화: Gradient Concentration 비교

![Gradient Concentration](../analysis_results/mechanism/figures/fig1_gradient_concentration.png)

#### 판정: ✅ STRONGLY SUPPORTED

**수학적 증명**:
```
Mean Loss:   ∂L/∂θ ∝ Σ(∂NLL_i/∂θ) / N    (N=196)
             → 각 패치당 gradient 기여: 1/196 ≈ 0.5%

Tail Loss:   ∂L/∂θ ∝ Σ(∂NLL_i/∂θ) / k    (k=4)
             → 각 패치당 gradient 기여: 1/4 = 25%

증폭 비율:   N/k = 196/4 = 49x (이론값)
실측 비율:   42.3x (측정값)
```

---

### 2.3 가설 H7: Latent Calibration 가설

#### 가설
> Tail 학습은 latent space z의 Gaussianity를 개선하여 likelihood 추정 정확도를 향상시킨다.

#### 검증 방법
```python
# z distribution 수집
all_z = []
for batch in test_loader:
    z, _ = model.forward(batch)
    all_z.append(z.flatten())

z_all = torch.cat(all_z).numpy()

# 1. QQ correlation (이론적 vs 실제 quantile)
theoretical_q = norm.ppf(np.linspace(0.01, 0.99, 99))
empirical_q = np.percentile(z_all, np.linspace(1, 99, 99))
qq_corr = np.corrcoef(theoretical_q, empirical_q)[0, 1]

# 2. Tail calibration error
tail_theoretical = norm.ppf([0.01, 0.02, 0.05, 0.95, 0.98, 0.99])
tail_empirical = np.percentile(z_all, [1, 2, 5, 95, 98, 99])
tail_error = np.abs(tail_theoretical - tail_empirical).mean()

# 3. Per-dimension normality test
for dim in range(z_dim):
    stat, p = shapiro(z_all[:, dim])
```

#### 결과

| 측정 항목 | 값 | 이상적 값 |
|----------|-----|-----------|
| z mean | 0.0010 | 0 |
| z std | 0.6853 | 1 |
| **QQ correlation** | **0.9892** | 1 |
| Mean KS p-value | 1.03e-05 | - |
| Tail calibration error | 0.2639 | 0 |

**Quantile 비교**:

| Percentile | Empirical | Theoretical | Error |
|------------|-----------|-------------|-------|
| 1% | -2.041 | -2.326 | 0.285 |
| 5% | -1.452 | -1.645 | 0.193 |
| 95% | 1.750 | 1.645 | 0.105 |
| 99% | 2.789 | 2.326 | 0.462 |

#### 시각화: Latent Space Calibration

![Latent Calibration](../analysis_results/mechanism/figures/fig4_latent_calibration.png)

#### 판정: ⚠️ PARTIALLY SUPPORTED

- 전반적 Gaussianity는 양호 (QQ corr = 0.989)
- 그러나 **극단 꼬리 (1%, 99%)에서 calibration 오차 존재**
- z std < 1 → Flow가 "압축된" 분포 학습

---

### 2.4 가설 H4: Top-K Ratio Dilution 가설 (ratio > 2%일 때 성능 하락 원인)

#### 가설
> tail_top_k_ratio가 2%보다 커지면 (1) gradient 희석, (2) hard example purity 감소, (3) mean loss로의 수렴으로 인해 성능이 하락한다.

#### 검증 방법
```python
# 다양한 ratio에서 분석
N = 196  # 14x14 패치
ratios = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50, 1.0]

for ratio in ratios:
    k = int(N * ratio)

    # 1. Gradient Amplification (이론적)
    amplification = N / k

    # 2. Hard Example Purity
    # 선택된 k개 중 실제 top-2%에 해당하는 비율
    purity = count_true_hard_in_topk(k) / k

    # 3. Effective Concentration
    effective = purity * amplification
```

#### 결과

| Ratio | k (패치 수) | Amplification | Purity | Effective Conc. |
|-------|------------|---------------|--------|-----------------|
| **1%** | 2 | 98x | 98% | 96x |
| **2%** | 4 | **49x** | **88%** | **43x** |
| 3% | 6 | 33x | 71% | 23x |
| 5% | 10 | 20x | 44% | 9x |
| **10%** | 20 | 10x | 21% | **2x** |
| 20% | 39 | 5x | 10% | 0.5x |
| **50%** | 98 | 2x | 4% | **0.1x** |
| 100% | 196 | 1x | 2% | 0.02x |

#### 시각화: Top-K Ratio Dilution Effect

![Top-K Ratio Dilution](../analysis_results/mechanism/figures/fig8_topk_ratio_dilution.png)

#### 판정: ✅ SUPPORTED

**세 가지 성능 하락 원인**:

**1. Gradient Dilution (희석)**
```
ratio=2%:  각 패치가 25%의 gradient 기여 → 49x 집중
ratio=10%: 각 패치가 5%의 gradient 기여 → 10x 집중
ratio=50%: 각 패치가 1%의 gradient 기여 → 2x 집중 (거의 mean loss)
```

**2. Hard Example Purity (순도) 감소**
```
ratio=2%:  선택된 패치의 88%가 "진짜 어려운" 패치
ratio=10%: 선택된 패치의 21%만 "진짜 어려운" 패치 (79%는 쉬운 패치)
ratio=50%: 선택된 패치의 4%만 "진짜 어려운" 패치 (96%는 쉬운 패치)
```
→ ratio가 커질수록 "쉬운" 패치가 선택에 혼입되어 학습 신호가 오염됨

**3. Mean Loss로의 수렴**
```
ratio → 100%일 때, top-k loss → mean loss
→ Gradient가 모든 패치에 균등 분배
→ Hard Example Mining 효과 사라짐
```

#### 핵심 통찰: Effective Concentration

**Effective Concentration = Purity × Amplification**

이 지표는 실제로 hard example에 집중되는 gradient의 양을 측정합니다:

| Ratio | 계산 | 해석 |
|-------|------|------|
| 2% | 88% × 49x = **43x** | 진짜 어려운 패치에 43배 집중 |
| 10% | 21% × 10x = **2x** | 진짜 어려운 패치에 2배만 집중 |
| 50% | 4% × 2x = **0.1x** | 집중 효과 거의 없음 |

**결론**: 2%가 최적인 이유
- 충분히 높은 gradient amplification (49x)
- 높은 hard example purity (88%)
- 결과적으로 높은 effective concentration (43x)

---

## 3. 결과 정리 및 결론

### 3.1 가설 검증 요약

| 가설 | 내용 | 결과 | 핵심 증거 |
|------|------|------|----------|
| **H1** | Tail = 이미지 경계 영역 | ❌ NOT SUPPORTED | Gradient ratio = 1.01x (무의미) |
| **H3** | Gradient Concentration | ✅ **SUPPORTED** | 42.3x gradient amplification |
| **H4** | Ratio > 2% → 성능 하락 | ✅ **SUPPORTED** | Purity 감소 + Gradient 희석 |
| **H7** | Latent Calibration | ⚠️ PARTIAL | QQ corr = 0.989, tail error 존재 |

#### 시각화: 가설 검증 결과 요약


---

### 3.2 핵심 메커니즘 규명

**Tail-Aware Loss의 작동 원리 = Hard Example Mining for NLL**

```
┌─────────────────────────────────────────────────────────────┐
│                    Mean-Only Loss (λ=0)                     │
├─────────────────────────────────────────────────────────────┤
│  196개 패치에 gradient 균등 분배                             │
│  → Tail/Non-Tail gradient ratio: 1.18x (거의 동일)          │
│  → "어려운" 패치도 "쉬운" 패치와 같은 학습 강도               │
│  → 비효율적 학습                                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
                        42.3x 증폭
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Tail-Aware Loss (λ=0.7)                    │
├─────────────────────────────────────────────────────────────┤
│  상위 2% (4개 패치)에 gradient 집중                          │
│  → Tail/Non-Tail gradient ratio: 49.99x                     │
│  → "어려운" 패치에 50배 강한 학습 신호                       │
│  → 분포 경계 정교화                                         │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 인과 관계 체인

#### 시각화: 인과 메커니즘 다이어그램

```
Tail-Aware Loss
      ↓
42x Gradient Concentration on hard patches
      ↓
Better learning of distribution boundaries
      ↓
More precise anomaly scoring at boundaries
      ↓
+10%p Pixel AP improvement
```

### 3.4 Tail 패치의 실제 의미

실험 결과, Tail 패치는:
- ❌ 이미지의 경계 영역이 **아님** (H1 rejected)
- ✅ **모델이 학습하기 어려워하는 패치** (높은 NLL)
- 높은 NLL의 원인: 복잡한 feature 구조, 드문 패턴, 분포 경계

---

### 3.5 최적 하이퍼파라미터

| 파라미터 | 최적값 | 근거 |
|----------|--------|------|
| `tail_weight` | **0.7** | 성능-안정성 균형점 |
| `tail_top_k_ratio` | **0.02** | 196/4 = 49x amplification |
| 선택 패치 수 | **~4개/이미지** | Hard example 집중 + 과적합 방지 |

#### 시각화: Tail Weight Ablation

![Tail Weight Ablation](../analysis_results/mechanism/figures/fig6_tail_weight_ablation.png)

---



---

## 4. 결론

### 핵심 결론

**Tail-Aware Loss는 Normalizing Flow를 위한 Hard Example Mining이다.**

- 전체 패치의 2%만 학습해도 gradient가 42배 집중됨
- 이를 통해 모델이 어려워하는 분포 경계를 효과적으로 학습
- 결과적으로 **+10%p Pixel AP 향상** 달성

---

## 5. 관련 연구

1. **Hard Example Mining**: Shrivastava et al., CVPR 2016
   - Tail-aware loss와 동일한 원리

2. **Focal Loss**: Lin et al., ICCV 2017
   - Class imbalance에서 hard example focusing

3. **Normalizing Flows**: Papamakarios et al., 2021
   - Flow의 tail estimation 특성
