# Tail-Aware Loss 메커니즘 심층 분석 설계

## 1. 현상 요약

### 관찰된 결과
```
Tail-Aware Loss (tail_weight=1.0, tail_top_k_ratio=0.02):
- Pix AP: 48.61% -> 56.18% (+7.57%p)
- Img AUC: 94.97% -> 97.92% (+2.95%p)
```

### 핵심 질문
**왜 전체 패치의 2%만 집중 학습해도 +7.57%p의 Pixel AP 향상이 가능한가?**

---

## 2. 제안된 가설 비판적 검토

### H1: Tail 패치는 경계/전이 영역에 해당한다
**가설**: Tail (NLL 상위 2%) 패치들은 normal-abnormal 경계나 texture 전이 영역에 위치한다.

**비판적 검토**:
- **강점**: 경계 영역은 모델이 학습하기 어려운 영역이므로 높은 NLL을 보일 가능성 높음
- **약점**: Train set에서의 tail이 반드시 의미 있는 경계를 의미하지 않을 수 있음 (noise일 수도)
- **검증 필요**: Train set의 tail이 test set의 decision boundary와 어떤 관계가 있는지 확인 필요
- **대안 설명**: Tail이 단순히 feature extraction의 artifacts (예: 이미지 가장자리)일 수도 있음

### H2: Tail 패치가 normal-anomaly 결정 경계를 형성한다
**가설**: Tail 영역에서 gradient를 강화하면 normal/anomaly 분리가 명확해진다.

**비판적 검토**:
- **강점**: Anomaly detection에서 어려운 샘플은 대부분 decision boundary 근처에 위치
- **약점**: Train set은 normal만 포함하므로, "decision boundary"라는 표현이 부적절
- **수정된 해석**: Train에서의 tail은 "normal 분포의 경계"이며, 이것이 test에서의 anomaly detection과 연결
- **검증 필요**: Train tail과 test anomaly score의 상관관계 분석

### H3: Mean-only 학습은 분포를 과도하게 smooth하게 만든다
**가설**: Mean loss만 사용하면 tail 영역의 likelihood가 부정확해진다.

**비판적 검토**:
- **강점**: NLL mean 최소화는 분포 전체를 골고루 fitting하므로 tail precision 손실 가능
- **약점**: Flow 모델의 특성상 likelihood는 정확히 계산되므로 "부정확"이라는 표현이 부적절
- **수정된 해석**: Mean 학습은 "tail 영역에서의 likelihood gradient magnitude"를 감소시킴
- **메커니즘**: Mean에서는 tail gradient가 희석(diluted)되어 tail 영역 최적화 부족

### H4: Training-Evaluation Alignment가 핵심이다
**가설**: Train에서 top-k 학습 -> Eval에서 top-k 평가 = 목표 일치

**비판적 검토**:
- **강점**: 명확한 인과관계 - 같은 metric으로 학습하면 성능 향상
- **약점**: 너무 단순한 설명 - 왜 이것이 효과적인지 메커니즘 설명 부족
- **심층 질문**:
  1. Top-k evaluation이 왜 효과적인가?
  2. Alignment가 정확히 무엇을 의미하는가?
- **검증 필요**: Train tail과 Eval tail의 실제 overlap 정도 측정

---

## 3. 추가 가설 제안

### H5: Tail 패치는 feature 공간에서 특정 구조를 형성한다
**가설**: Tail 패치들은 random하게 분포하지 않고, feature space에서 coherent한 cluster를 형성한다.

**근거**:
- 이미지에서 어려운 패치들은 비슷한 특성을 공유할 가능성 (예: texture boundary, edge)
- 이 cluster를 잘 학습하면 test에서도 비슷한 영역을 잘 처리

### H6: Tail 학습은 Flow의 Jacobian 정밀도를 향상시킨다
**가설**: Tail 패치에서 gradient를 강화하면 coupling layer의 s, t network가 더 정교해진다.

**근거**:
- log p(x) = log p(z) + log|det J|
- Tail은 log|det J|가 특히 중요한 영역일 수 있음
- Mean 학습에서는 Jacobian precision이 전체적으로 낮게 유지될 수 있음

### H7: Tail 학습은 latent space calibration을 개선한다
**가설**: Tail 집중 학습으로 z의 Gaussianity가 향상되고, 특히 tail 영역에서 정확한 likelihood 추정이 가능해진다.

**근거**:
- Flow는 z ~ N(0,I)를 가정하지만, 실제로는 deviation이 있음
- Tail 학습이 z distribution의 tail calibration을 개선

---

## 4. 엄밀한 실험 설계

### Experiment 1: Tail 패치의 공간적 특성 분석

**목적**: Tail 패치가 이미지의 어디에 위치하는지 분석

**실험 설계**:
```python
def analyze_tail_spatial_distribution():
    """
    실험 1.1: Tail 패치의 공간적 분포 분석

    측정 지표:
    1. Spatial Position Histogram: Tail이 edge/center/random 중 어디에 집중되는지
    2. Texture Gradient Correlation: Tail과 image gradient의 상관관계
    3. Semantic Region Analysis: Tail이 특정 semantic 영역에 집중되는지
    """
    for batch in train_loader:
        z, logdet = flow.forward(batch)
        nll_patch = compute_patch_nll(z, logdet)

        # Top-k% tail 추출
        flat_nll = nll_patch.reshape(B, -1)
        k = int(H * W * 0.02)
        tail_indices = torch.topk(flat_nll, k, dim=1)[1]

        # 공간 위치 분석
        tail_positions = indices_to_2d(tail_indices, H, W)

        # 1. Edge proximity 계산
        edge_distance = compute_edge_distance(tail_positions, H, W)

        # 2. Gradient magnitude at tail locations
        image_grad = compute_image_gradient(batch)
        tail_gradient_strength = image_grad[tail_positions]

        # 3. Semantic segmentation overlap (if available)
        # ...

    return {
        'edge_distance_histogram': ...,
        'gradient_correlation': ...,
        'spatial_heatmap': ...
    }
```

**기대 결과**:
- Tail이 edge 근처에 집중되면 H1 지지
- Random 분포면 H1 기각

**통계 검정**:
- Kolmogorov-Smirnov test: Tail 위치 분포 vs Uniform 분포
- Spearman correlation: Tail NLL vs Image gradient magnitude

---

### Experiment 2: Train Tail과 Test Anomaly의 관계 분석

**목적**: Training에서의 tail이 test-time anomaly detection과 어떤 관계가 있는지 분석

**실험 설계**:
```python
def analyze_train_tail_test_anomaly_relationship():
    """
    실험 2.1: Train tail feature와 Test anomaly feature의 유사도

    가설: Train에서 어려웠던 패치의 feature가 Test anomaly와 유사
    """
    # Phase 1: Train에서 tail 패치들의 feature 수집
    train_tail_features = []
    for batch in train_loader:
        features = extractor(batch)
        z, logdet = flow.forward(features)
        nll_patch = compute_patch_nll(z, logdet)

        # Tail 패치 feature 추출
        k = int(H * W * 0.02)
        tail_indices = torch.topk(nll_patch.flatten(), k)[1]
        tail_features.append(features.flatten()[tail_indices])

    train_tail_features = torch.cat(train_tail_features)
    train_tail_prototype = train_tail_features.mean(dim=0)

    # Phase 2: Test에서 anomaly/normal 패치들의 feature 수집
    test_anomaly_features = []
    test_normal_features = []
    for batch, mask in test_loader:
        features = extractor(batch)
        z, logdet = flow.forward(features)
        nll_patch = compute_patch_nll(z, logdet)

        # Ground truth mask로 분리
        anomaly_mask = (mask > 0.5).flatten()
        test_anomaly_features.append(features.flatten()[anomaly_mask])
        test_normal_features.append(features.flatten()[~anomaly_mask])

    # Phase 3: 유사도 계산
    sim_train_tail_to_test_anomaly = cosine_similarity(
        train_tail_prototype,
        torch.cat(test_anomaly_features).mean(dim=0)
    )
    sim_train_tail_to_test_normal = cosine_similarity(
        train_tail_prototype,
        torch.cat(test_normal_features).mean(dim=0)
    )

    return {
        'sim_tail_anomaly': sim_train_tail_to_test_anomaly,
        'sim_tail_normal': sim_train_tail_to_test_normal,
        'ratio': sim_train_tail_to_test_anomaly / sim_train_tail_to_test_normal
    }
```

**기대 결과**:
- ratio > 1이면 H2 부분 지지 (train tail이 anomaly와 더 유사)
- ratio ~ 1이면 H2 기각

---

### Experiment 3: Gradient Dynamics 분석

**목적**: Tail-aware loss가 gradient 분포를 어떻게 변화시키는지 분석

**실험 설계**:
```python
def analyze_gradient_dynamics():
    """
    실험 3.1: Mean vs Tail-Aware loss의 gradient magnitude 비교

    측정:
    1. Tail 패치 gradient magnitude
    2. Non-tail 패치 gradient magnitude
    3. Coupling layer별 gradient 분포
    """
    # Two training runs: mean only vs tail-aware
    configs = [
        {'tail_weight': 0.0, 'name': 'mean_only'},
        {'tail_weight': 1.0, 'name': 'tail_aware'}
    ]

    results = {}
    for config in configs:
        model = create_model()

        # Gradient 수집 (첫 N epochs)
        gradient_history = []
        for epoch in range(N):
            for batch in train_loader:
                z, logdet = model.forward(batch)
                loss = compute_loss(z, logdet, **config)
                loss.backward()

                # Per-patch gradient magnitude 계산
                # Flow의 coupling layer에서 gradient 추출
                grad_magnitudes = []
                for layer in model.coupling_layers:
                    for param in layer.parameters():
                        if param.grad is not None:
                            grad_magnitudes.append(param.grad.norm().item())

                gradient_history.append({
                    'epoch': epoch,
                    'grad_mean': np.mean(grad_magnitudes),
                    'grad_std': np.std(grad_magnitudes),
                    'grad_max': np.max(grad_magnitudes)
                })

        results[config['name']] = gradient_history

    return results
```

**분석 포인트**:
1. Tail-aware에서 gradient variance가 더 큰가?
2. 학습 후반에 gradient magnitude 차이가 어떻게 변하는가?
3. 특정 layer에서 gradient 차이가 더 큰가?

---

### Experiment 4: Latent Space Calibration 분석

**목적**: Tail 학습이 z distribution의 Gaussianity에 미치는 영향 분석

**실험 설계**:
```python
def analyze_latent_calibration():
    """
    실험 4.1: z distribution의 Gaussianity 분석

    측정:
    1. z의 각 dimension별 normality test
    2. z의 전체적인 multivariate normality
    3. z의 tail 영역 calibration (QQ-plot)
    """
    from scipy.stats import shapiro, anderson, kstest

    configs = ['mean_only', 'tail_aware']
    results = {}

    for config in configs:
        model = load_trained_model(config)

        all_z = []
        for batch in test_loader:
            z, _ = model.forward(batch)
            all_z.append(z.flatten(0, 2))  # (B*H*W, D)

        all_z = torch.cat(all_z, dim=0).cpu().numpy()

        # 1. Per-dimension normality (Shapiro-Wilk)
        normality_p_values = []
        for d in range(all_z.shape[1]):
            stat, p = shapiro(all_z[:1000, d])  # Subsample for efficiency
            normality_p_values.append(p)

        # 2. Tail calibration (empirical vs theoretical quantiles)
        theoretical_quantiles = norm.ppf(np.linspace(0.01, 0.99, 99))
        empirical_quantiles = np.percentile(all_z.flatten(),
                                            np.linspace(1, 99, 99))
        qq_correlation = np.corrcoef(theoretical_quantiles,
                                     empirical_quantiles)[0, 1]

        # 3. Tail-specific calibration (focus on extreme percentiles)
        tail_theoretical = norm.ppf([0.01, 0.02, 0.05, 0.95, 0.98, 0.99])
        tail_empirical = np.percentile(all_z.flatten(),
                                       [1, 2, 5, 95, 98, 99])
        tail_error = np.abs(tail_theoretical - tail_empirical).mean()

        results[config] = {
            'mean_normality_p': np.mean(normality_p_values),
            'qq_correlation': qq_correlation,
            'tail_calibration_error': tail_error
        }

    return results
```

**기대 결과**:
- Tail-aware가 더 높은 QQ correlation을 가지면 H7 지지
- 특히 tail_calibration_error가 더 작으면 tail 영역 calibration 개선 증거

---

### Experiment 5: Score Distribution 분석 (Normal vs Anomaly)

**목적**: Tail 학습이 anomaly score의 separation을 어떻게 개선하는지 분석

**실험 설계**:
```python
def analyze_score_separation():
    """
    실험 5.1: Normal vs Anomaly score distribution 분석

    측정:
    1. Score distribution의 mean/std
    2. Fisher's Discriminant Ratio
    3. Threshold-independent separation (AUC decomposition)
    """
    configs = ['mean_only', 'tail_aware']
    results = {}

    for config in configs:
        model = load_trained_model(config)

        normal_scores = []
        anomaly_scores = []

        for batch, mask in test_loader:
            patch_scores = model.inference(batch)

            # Ground truth로 분리
            flat_scores = patch_scores.flatten()
            flat_mask = mask.flatten() > 0.5

            normal_scores.extend(flat_scores[~flat_mask].tolist())
            anomaly_scores.extend(flat_scores[flat_mask].tolist())

        normal_scores = np.array(normal_scores)
        anomaly_scores = np.array(anomaly_scores)

        # Fisher's Discriminant Ratio
        mean_diff = anomaly_scores.mean() - normal_scores.mean()
        pooled_var = (normal_scores.var() + anomaly_scores.var()) / 2
        fdr = mean_diff ** 2 / pooled_var

        # Overlap coefficient
        overlap = compute_distribution_overlap(normal_scores, anomaly_scores)

        results[config] = {
            'normal_mean': normal_scores.mean(),
            'normal_std': normal_scores.std(),
            'anomaly_mean': anomaly_scores.mean(),
            'anomaly_std': anomaly_scores.std(),
            'fisher_ratio': fdr,
            'overlap': overlap
        }

    return results
```

**기대 결과**:
- Tail-aware가 더 높은 Fisher ratio를 가지면 separation 개선 증거
- Overlap 감소는 detection threshold robustness 증가 의미

---

### Experiment 6: Coupling Layer별 기여도 분석

**목적**: Tail 학습이 어떤 coupling layer에 가장 큰 영향을 미치는지 분석

**실험 설계**:
```python
def analyze_per_layer_contribution():
    """
    실험 6.1: Layer별 log-det 기여도 분석

    Normalizing Flow: log p(x) = log p(z) + sum_{l} log|det J_l|

    각 layer의 log-det가 tail 학습으로 어떻게 변하는지 분석
    """
    configs = ['mean_only', 'tail_aware']
    results = {}

    for config in configs:
        model = load_trained_model(config)

        # Hook으로 layer별 log-det 수집
        layer_logdets = {i: [] for i in range(len(model.coupling_layers))}

        def hook_fn(layer_idx):
            def hook(module, input, output):
                _, logdet = output
                layer_logdets[layer_idx].append(logdet.detach())
            return hook

        for i, layer in enumerate(model.coupling_layers):
            layer.register_forward_hook(hook_fn(i))

        for batch in test_loader:
            _ = model.forward(batch)

        # 분석
        for layer_idx, logdets in layer_logdets.items():
            logdets = torch.cat(logdets, dim=0)
            results[f'{config}_layer{layer_idx}'] = {
                'mean': logdets.mean().item(),
                'std': logdets.std().item(),
                'min': logdets.min().item(),
                'max': logdets.max().item()
            }

    return results
```

**기대 결과**:
- 특정 layer에서 mean-only vs tail-aware의 차이가 크면, 해당 layer가 tail adaptation에 핵심

---

### Experiment 7: Ablation - Top-K Ratio 변화에 따른 효과 분석

**목적**: tail_top_k_ratio가 성능에 미치는 영향의 메커니즘 이해

**실험 설계**:
```python
def analyze_topk_ratio_effect():
    """
    실험 7.1: Top-K ratio 변화에 따른 다양한 메트릭 분석

    ratio: 0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50

    측정:
    1. Pixel AP (performance)
    2. Gradient concentration (how focused is learning)
    3. Tail coverage (what fraction of "hard" patches are covered)
    """
    ratios = [0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.50]
    results = []

    for ratio in ratios:
        model = train_model(tail_top_k_ratio=ratio, tail_weight=1.0)

        # Performance
        pixel_ap = evaluate(model)['pixel_ap']

        # Gradient concentration: Gini coefficient of gradient magnitudes
        grad_magnitudes = get_gradient_distribution(model)
        gini = compute_gini(grad_magnitudes)

        # Effective rank of gradient distribution
        effective_rank = compute_effective_rank(grad_magnitudes)

        results.append({
            'ratio': ratio,
            'pixel_ap': pixel_ap,
            'gradient_gini': gini,
            'effective_rank': effective_rank
        })

    return results
```

**기대 결과**:
- 최적 ratio에서 gradient concentration과 coverage의 균형점 존재
- ratio가 너무 작으면: coverage 부족, 너무 크면: mean loss와 유사해짐

---

## 5. 종합 분석 프레임워크

### 메커니즘 인과 그래프

```
Tail-Aware Loss
      |
      v
+---------------------+
| Gradient Focusing   |  (Exp 3)
| on High-NLL Patches |
+---------------------+
      |
      +-----------------+------------------+
      |                 |                  |
      v                 v                  v
+----------+    +---------------+    +--------------+
| Coupling |    | Latent Space  |    | Log-det      |
| Layer    |    | Calibration   |    | Precision    |
| Adaptation|   | Improvement   |    | Improvement  |
+----------+    +---------------+    +--------------+
      |                 |                  |
      v                 v                  v
+---------------------------------------------------+
|      Improved Score Separation (Normal vs Anomaly) |
+---------------------------------------------------+
      |
      v
+---------------------------------------------------+
|             Higher Pixel AP                        |
+---------------------------------------------------+
```

### 검증 순서

1. **Exp 1**: Tail 패치의 특성 파악 (what)
2. **Exp 2**: Train-Test 관계 파악 (why relevant)
3. **Exp 3**: Gradient dynamics (how)
4. **Exp 4-6**: 구체적 메커니즘 검증 (detailed how)
5. **Exp 7**: Hyperparameter sensitivity (robustness)

---

## 6. 구현 계획

### Phase 1: 데이터 수집 (분석용 Hook 추가)
- [ ] Tail 패치 위치 수집 hook
- [ ] Per-layer log-det 수집 hook
- [ ] Gradient magnitude 수집 hook

### Phase 2: 기본 분석 실행
- [ ] Experiment 1: Spatial distribution
- [ ] Experiment 3: Gradient dynamics
- [ ] Experiment 5: Score separation

### Phase 3: 심층 분석
- [ ] Experiment 2: Train-Test relationship
- [ ] Experiment 4: Latent calibration
- [ ] Experiment 6: Layer contribution

### Phase 4: 종합 및 검증
- [ ] Experiment 7: Ablation study
- [ ] 가설 검증 결과 종합
- [ ] 메커니즘 인과 관계 확립

---

## 7. 예상 결론 시나리오

### 시나리오 A: Gradient Focusing이 핵심
- Exp 3에서 명확한 gradient concentration 차이
- Exp 6에서 특정 layer의 기여도 변화
- **해석**: Tail 학습은 어려운 패치에 gradient를 집중시켜 해당 영역의 transformation 정밀도 향상

### 시나리오 B: Latent Calibration이 핵심
- Exp 4에서 명확한 Gaussianity 차이
- Exp 5에서 score distribution의 tail 영역 개선
- **해석**: Tail 학습은 z distribution의 tail calibration을 개선하여 likelihood 추정 정확도 향상

### 시나리오 C: 복합 효과
- 여러 실험에서 유의미한 차이
- **해석**: Tail 학습은 gradient focusing + calibration의 시너지 효과

---

## 8. 코드 구현 위치

```
moleflow/
  analysis/
    tail_aware_analysis.py      # 메인 분석 코드
    gradient_analyzer.py        # Gradient dynamics 분석
    latent_analyzer.py          # Latent space 분석
    score_analyzer.py           # Score distribution 분석
    visualization.py            # 시각화 utilities
```

---

## 9. 참고문헌 및 관련 연구

1. **Normalizing Flow Tail Behavior**:
   - Papamakarios et al., "Normalizing Flows for Probabilistic Modeling and Inference" (2021)
   - Flow의 tail estimation 한계에 대한 논의

2. **Hard Example Mining**:
   - Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining" (CVPR 2016)
   - Tail-aware loss와 유사한 접근

3. **Focal Loss**:
   - Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
   - Class imbalance 문제에서 hard example focusing

4. **Anomaly Detection Score Calibration**:
   - Ruff et al., "A Unifying Review of Deep and Shallow Anomaly Detection" (2021)
   - Score calibration의 중요성
