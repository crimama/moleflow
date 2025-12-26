# MoLE-Flow Update Notes

## Version History

---

## v1 (baseline) - Initial Implementation

### Architecture
- Task 0: Base NF만 학습 (LoRA 없음)
- Task 1+: Base frozen + LoRA 학습
- LoRA scaling: `alpha / (2 * rank)` = 0.0156 (1.56%)
- InputAdapter: Instance Norm + Zero-init MLP

### Results (leather → grid → transistor)
| Class | Routing Acc | Image AUC | Pixel AUC |
|-------|-------------|-----------|-----------|
| leather | 100% | **1.0000** | **0.9481** |
| grid | 100% | 0.8204 | 0.9259 |
| transistor | 100% | 0.6654 | 0.7144 |
| **Mean** | 100% | 0.8286 | 0.8628 |

### Issues Identified
1. **Task 0 Bias**: Base NF가 Task 0에 편향되어 다른 task에서 성능 저하
2. **LoRA Scaling 부족**: 1.56% contribution으로 adaptation 효과 미미
3. **InputAdapter 한계**: MLP residual gate가 0으로 시작하여 거의 미사용

---

## v2 (baseline_v2) - LoRA Scaling & Task 0 LoRA

### Changes from v1
1. **LoRA Scaling 2배 증가**
   ```python
   # v1: self.scaling = alpha / (2 * rank)  # 0.0156
   # v2: self.scaling = alpha / rank         # 0.03125
   ```

2. **Task 0도 LoRA 사용**
   - 모든 Task가 동등하게 LoRA로 adaptation
   - Base NF는 범용 feature transformation 학습
   - Task-specific adaptation은 LoRA가 담당

### Results (leather → grid → transistor)
| Class | Routing Acc | Image AUC | Pixel AUC |
|-------|-------------|-----------|-----------|
| leather | 100% | 0.9997 | 0.8900 |
| grid | 100% | **0.9850** | **0.9841** |
| transistor | 100% | **0.8075** | **0.8973** |
| **Mean** | 100% | **0.9307** | **0.9238** |

### Comparison (v1 → v2)
| Metric | v1 | v2 | Change |
|--------|-----|-----|--------|
| leather Image | 1.0000 | 0.9997 | -0.03% |
| leather Pixel | 0.9481 | 0.8900 | **-6.1%** |
| grid Image | 0.8204 | 0.9850 | **+20.0%** |
| grid Pixel | 0.9259 | 0.9841 | +6.3% |
| transistor Image | 0.6654 | 0.8075 | **+21.4%** |
| transistor Pixel | 0.7144 | 0.8973 | **+25.6%** |
| **Mean Image** | 0.8286 | 0.9307 | **+12.3%** |
| **Mean Pixel** | 0.8628 | 0.9238 | **+7.1%** |

### Analysis
- **Overall**: 전체 성능 대폭 향상 (Mean Image AUC +12.3%)
- **Task 0 Issue**: leather의 Pixel AUC가 6.1% 하락
  - 원인: Task 0에서 Base + LoRA 동시 학습 시, LoRA가 일부 정보를 분담하면서 Base의 표현력 분산
  - LoRA는 Task-specific adaptation에 최적화되어 pixel-level 정밀도에 영향

---

## v3 (baseline_v3) - FiLM-style InputAdapter + Task 0 Self-Adaptation

### Changes from v2
1. **InputAdapter 구조 개선 (FiLM-style)**
   - Instance Norm → Layer Norm (spatial info 보존)
   - FiLM (Feature-wise Linear Modulation): `y = gamma * x + beta`
   - residual_gate: 0 → 0.5 (MLP 처음부터 active)
   - hidden_dim 증가: `channels//4` → `max(channels//2, 128)`

2. **Task 0 Self-Adaptation**
   - Task 0에도 InputAdapter 적용 (v2에서는 Task 0에 InputAdapter 없었음)
   - `has_reference=False` 설정으로 강한 identity connection (90% identity + 10% transformed)
   - 이를 통해 Task 0의 pixel-level 성능 회복 기대

3. **모든 Task 동등한 InputAdapter 적용**
   - v2: Task 0 (InputAdapter 없음) vs Task 1+ (InputAdapter 있음)
   - v3: 모든 Task가 InputAdapter 사용 (구조적 일관성)

### Key Code Changes
```python
# adapters.py - TaskInputAdapter v3
class TaskInputAdapter(nn.Module):
    def __init__(self, channels, reference_mean=None, reference_std=None, use_norm=True):
        # FiLM parameters
        self.film_gamma = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.film_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

        # Larger MLP
        hidden_dim = max(channels // 2, 128)  # Increased from channels//4

        # Active residual gate
        self.residual_gate = nn.Parameter(torch.tensor([0.5]))  # Changed from 0.0

        # Layer Norm instead of Instance Norm
        if use_norm:
            self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x):
        # ...
        # Task 0 (has_reference=False): 90% identity + 10% transformed
        if not self.has_reference:
            x = 0.9 * identity + 0.1 * x
        return x
```

### Expected Improvements
- Task 0 pixel-level 성능 회복 (v1 수준으로)
- 더 강력한 cross-task feature transformation
- 구조적 일관성으로 인한 안정적인 학습

---

## File Changes Summary

### v1 → v2
| File | Changes |
|------|---------|
| `moleflow/models/lora.py` | scaling: `alpha/(2*rank)` → `alpha/rank` |
| `moleflow/models/mole_nf.py` | Task 0에서도 LoRA adapter 추가 |
| `moleflow/trainer/continual_trainer.py` | Task 0 학습 로직 수정 |

### v2 → v3
| File | Changes |
|------|---------|
| `moleflow/models/adapters.py` | FiLM-style InputAdapter (LayerNorm, gate=0.5, larger MLP) |
| `moleflow/models/mole_nf.py` | Task 0에도 InputAdapter 적용 (self-adaptation) |
| `run.sh` | baseline_v3 실험 설정 |

---

## v4 (baseline_v4) - CPCF + SC-LoRA (Novel Research Contribution)

### Research Contribution
**Paper Title**: "Beyond Independent Patches: Cross-Patch Coupling Flow with Spatial LoRA for Continual Anomaly Detection"

**Key Novelty Claims**:
1. **CPCF**: First normalizing flow that models `p(x_i | neighbors)` instead of `p(x_i)`
2. **SC-LoRA**: First spatially-varying LoRA for dense prediction tasks
3. **Continual AD**: Systematic benchmark for continual anomaly detection

### New Modules

#### 1. Spatial-Contextual LoRA (SC-LoRA)
```python
# moleflow/models/spatial_lora.py
class SpatialContextualLoRA(nn.Module):
    """
    Position-aware LoRA with spatial grid interpolation.

    Key Innovation:
    - LoRA parameters vary based on spatial position
    - Grid of LoRA parameters: (grid_size, grid_size, rank, dim)
    - Bilinear interpolation for smooth spatial adaptation
    """
    def __init__(self, in_features, out_features, rank=32, grid_size=4):
        # Grid of LoRA A: (G, G, R, D_in)
        self.lora_A = nn.Parameter(torch.zeros(grid_size, grid_size, rank, in_features))
        # Grid of LoRA B: (G, G, D_out, R)
        self.lora_B = nn.Parameter(torch.zeros(grid_size, grid_size, out_features, rank))

    def forward(self, x, positions):
        # Interpolate LoRA params based on position
        A, B = self._get_interpolated_lora(positions)
        return self.scaling * (x @ A.T @ B.T)
```

#### 2. Cross-Patch Coupling Flow (CPCF)
```python
# moleflow/models/cross_patch_flow.py
class CrossPatchCouplingLayer(nn.Module):
    """
    Coupling layer conditioned on neighborhood context.

    Standard: y = x * exp(s(x)) + t(x)
    CPCF:     y = x * exp(s(x, ctx)) + t(x, ctx)

    where ctx = NeighborhoodContext(x)
    """
    def __init__(self, channels, context_kernel=3):
        self.context_extractor = NeighborhoodContextExtractor(channels)
        self.s_net = SC_LoRA_MLP(channels + context_dim, channels)
        self.t_net = SC_LoRA_MLP(channels + context_dim, channels)

    def forward(self, x):
        context = self.context_extractor(x)  # (B, H, W, D)
        x1, x2 = x.split(D//2, dim=-1)

        # Condition on both patch and context
        s = self.s_net(cat([x1, context]))
        t = self.t_net(cat([x1, context]))

        return cat([x1, x2 * exp(s) + t])
```

### Theoretical Contribution

**Standard NF (Independent Patches)**:
```
log p(X) = Σᵢ log p(xᵢ)
```
- Assumes patches are independent
- Ignores spatial context

**CPCF (Context-Aware Patches)**:
```
log p(X) = Σᵢ log p(xᵢ | N(i))
```
where N(i) = neighborhood of patch i

- Models "how different is this patch from its neighbors"
- Directly captures anomaly as contextual deviation

### Architecture Comparison

| Component | v3 | v4 |
|-----------|-----|-----|
| Flow Type | FrEIA (independent) | CPCF (context-aware) |
| LoRA Type | Standard LoRA | SC-LoRA (position-aware) |
| Patch Modeling | `p(xᵢ)` | `p(xᵢ \| neighbors)` |
| Spatial Awareness | Position embedding only | Context + Position LoRA |

### Command Line Arguments
```bash
python run_moleflow.py \
    --use_cpcf \              # Enable Cross-Patch Coupling Flow
    --use_spatial_lora \       # Enable Spatial-Contextual LoRA
    --sc_lora_grid_size 4 \    # SC-LoRA grid resolution (4x4)
    --cpcf_context_kernel 3 \  # Context extraction kernel size
    --cpcf_use_attention       # Use attention instead of conv for context
```

### Expected Improvements
- **Image AUC**: +5-10% (better global anomaly detection)
- **Pixel AUC**: +10-15% (context-aware localization)
- **Cross-task**: Improved SC-LoRA handles position-dependent distribution shifts

### v3 → v4
| File | Changes |
|------|---------|
| `moleflow/models/spatial_lora.py` | NEW: SC-LoRA module |
| `moleflow/models/cross_patch_flow.py` | NEW: CPCF module |
| `moleflow/models/mole_nf.py` | CPCF/SC-LoRA integration |
| `run_moleflow.py` | CPCF/SC-LoRA arguments |
| `run.sh` | baseline_v4 실험 설정 |

---

## v4.1 - Fixed Context Extraction (Bug Fix)

### Issue
- v4에서 Task 0 성능은 향상되었으나 Task 1, 2 성능이 크게 저하됨
- 원인: Context Extractor가 Task 0에서만 학습되어 Task 0에 편향

### Root Cause
```python
# 이전 코드 - Task 0에서만 context extractor 학습
if task_id == 0:
    params.extend(layer.context_extractor.parameters())
```
- Task 0: Context Extractor가 Task 0 데이터로만 학습
- Task 1+: Task 0에 편향된 context feature 제공 → 성능 저하

### Fix
Context extraction을 **고정된 (non-learnable) 방식**으로 변경:

```python
# cross_patch_flow.py - NeighborhoodContextExtractor
class NeighborhoodContextExtractor(nn.Module):
    def __init__(self, channels, kernel_size=3, use_attention=False):
        # FIXED averaging kernel - no learnable parameters
        kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size * kernel_size)
        self.register_buffer('avg_kernel', kernel)
        self.register_buffer('context_gate', torch.tensor([0.3]))  # Fixed gate

    def forward(self, x):
        # Simple neighborhood mean (task-agnostic)
        neighbor_mean = F.conv2d(x, self.avg_kernel, padding, groups=D)
        context = (1 - gate) * x + gate * neighbor_mean
        return context
```

### Key Insight
- **Context = 주변 평균** → task-agnostic (어떤 task에서도 동일한 의미)
- **SC-LoRA = task-specific adaptation** → task 별로 다르게 학습
- 역할 분리: 고정 context + 학습 가능한 adaptation

### v4 → v4.1
| File | Changes |
|------|---------|
| `moleflow/models/cross_patch_flow.py` | Fixed (non-learnable) context extraction |
| `moleflow/models/mole_nf.py` | Removed context_extractor from trainable params |

---

## v5 - Center Loss for Discriminative Feature Learning

### Motivation
- v4의 cross-patch context 접근법이 task bias 문제로 실패
- Anomaly Detection 자체의 성능 향상에 집중
- Normal feature를 더 compact하게 만들어서 anomaly 구분 용이하게

### Core Idea
**Latent space z**에 Center Loss를 적용하여 normal을 z ≈ 0으로 더 강하게 유도:
```
Loss = NLL_loss + λ * Center_loss
     = -log p(x) + λ * ||z||²
```

**Key Insight**:
- NF는 input을 Gaussian으로 매핑 → normal은 z ≈ 0
- Center Loss는 이 목표를 **더 강하게** 강제
- Input feature가 아닌 **latent z**에 적용해야 gradient가 NF에 전파됨

### Implementation

```python
# Training loop에서
z, log_jac_det = nf_model.forward(x, reverse=False)

# NLL loss
log_pz = -0.5 * (z ** 2).sum() / (B * H * W)
nll_loss = -(log_pz + log_jac)

# Center loss on latent z (fixed center at zero)
center_loss = (z ** 2).sum(dim=-1).mean()  # ||z - 0||²

total_loss = nll_loss + λ * center_loss
```

### Why Fixed Center at Zero?
- Learnable center → center가 z의 mean으로 이동 → 의미 없음
- Fixed center = 0 → z를 원점으로 강하게 당김 → Gaussian prior 강화

### Training Flow
1. Forward: x → NF → z
2. NLL loss: z가 Gaussian을 따르도록
3. Center loss: z가 원점에 가깝도록 (추가 regularization)
4. Backward: gradient가 NF로 전파되어 더 compact한 latent space 학습

### Command Line Arguments
```bash
python run_moleflow.py \
    --center_loss_weight 0.05 \  # Center loss weight (recommend 0.01-0.1)
    ...
```

### Expected Improvements
- Normal feature가 더 compact해져서 anomaly 구분 향상
- 특히 pixel-level anomaly detection에서 효과 기대
- Task별 center가 task-specific 특성을 학습

### v3 → v5
| File | Changes |
|------|---------|
| `moleflow/models/center_loss.py` | NEW: CenterLoss module |
| `moleflow/trainer/continual_trainer.py` | Center loss integration |
| `run_moleflow.py` | `--center_loss_weight` argument |
| `run.sh` | v5 experiment configuration |

---

## v6 - Patch Self-Attention for Contextual Anomaly Detection

### Motivation
- v5의 Center Loss는 NLL이 이미 z ≈ 0을 유도하므로 효과 미미
- **Contextual Anomaly** 탐지 필요: 주변 패치와 다른 패치가 anomaly
- Patch 간 관계를 모델링하여 anomaly detection 성능 향상

### Core Idea
**Standard NF (Independent Patches)**:
```
log p(X) = Σᵢ log p(xᵢ)
```
- 각 패치를 독립적으로 처리
- 주변 context 무시

**Patch Self-Attention (Context-Aware)**:
```
log p(X) = Σᵢ log p(xᵢ | context_i)
where context_i = Attention(xᵢ, all patches)
```
- 패치 간 관계 모델링
- "주변과 다른" 패치를 anomaly로 탐지

### Architecture
```
ViT Features [B, H, W, D]
       ↓
Patch Self-Attention (LightweightPatchAttention)
       ↓
Context-Enhanced Features [B, H, W, D]
       ↓
Normalizing Flow
       ↓
Latent z, log_jac_det
```

### LightweightPatchAttention Module
```python
class LightweightPatchAttention(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=256, dropout=0.1):
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)

        # Learnable gate (starts at 0 for stable training)
        self.gate = nn.Parameter(torch.zeros(1))

        # FFN for additional processing
        self.ffn = nn.Sequential(...)

    def forward(self, x):  # x: [B, H, W, D]
        # Reshape to sequence
        x_seq = x.view(B, H*W, D)

        # Self-attention
        Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        attn = softmax(Q @ K.T / sqrt(d))
        attn_out = attn @ V

        # Gated residual connection
        gate = sigmoid(self.gate)
        x = x + gate * attn_out

        # FFN
        x = x + self.ffn(x)

        return x.view(B, H, W, D)
```

### Key Design Choices
1. **Learnable Gate (starts at 0)**:
   - 학습 초기에는 identity (안정적 학습)
   - 점진적으로 attention 기여도 증가

2. **Single-Head Attention**:
   - Lightweight하면서도 patch 관계 포착
   - Multi-head 대비 계산 효율적

3. **Pre-LayerNorm + FFN**:
   - Transformer 스타일 안정적 학습
   - FFN으로 비선형 transformation 강화

### Training
- Patch attention 모듈은 **모든 task에서 공유** (Base NF처럼)
- Task 0에서 학습 후 freeze
- Task 1+에서는 LoRA만 학습

### Command Line Arguments
```bash
python run_moleflow.py \
    --use_patch_attention \  # Enable Patch Self-Attention
    ...
```

### Expected Improvements
- **Contextual Anomaly Detection**: 주변과 다른 패치 탐지 향상
- **Pixel-level AUC**: Context 정보로 localization 정밀도 향상
- **Structural Anomaly**: 전역적 패치 관계로 구조적 이상 탐지

### v3 → v6
| File | Changes |
|------|---------|
| `moleflow/models/patch_attention.py` | NEW: LightweightPatchAttention, PatchInteractionModule |
| `moleflow/models/__init__.py` | Export patch attention modules |
| `moleflow/trainer/continual_trainer.py` | Patch attention integration |
| `run_moleflow.py` | `--use_patch_attention` argument |
| `run.sh` | v6 experiment configuration |

### v6 Result
- **실패**: ViT가 이미 self-attention으로 contextualized된 feature를 출력하므로 추가 attention이 오히려 해가 됨

---

## v7 - Focal NLL Loss for Hard Sample Mining

### Motivation
- v5 (Center Loss): NLL이 이미 z ≈ 0 유도하므로 효과 미미
- v6 (Patch Attention): ViT 중복으로 실패
- **새로운 접근**: 어려운 샘플에 더 집중하여 decision boundary 학습 강화

### Core Idea
**Standard NLL**:
```
L = -log p(x)  # 모든 샘플 동등한 가중치
```

**Focal NLL**:
```
L = (1 - p)^γ * (-log p(x))

γ = 0: Standard NLL (모든 샘플 동등)
γ = 1: 약간의 hard sample 강조
γ = 2: 강한 hard sample 강조 (권장)
```

- `p` = probability = exp(-nll)
- 높은 NLL (어려운 샘플) → 낮은 p → 높은 weight (1-p)^γ
- 낮은 NLL (쉬운 샘플) → 높은 p → 낮은 weight

### Implementation
```python
def _compute_nll_loss(self, z, log_jac_det):
    B, H, W, D = z.shape

    if not self.use_focal_loss:
        # Standard NLL
        log_pz = -0.5 * (z ** 2).sum() / (B * H * W)
        log_jac = log_jac_det.mean() / (H * W)
        return -(log_pz + log_jac)
    else:
        # Per-patch NLL
        log_pz_per_patch = -0.5 * (z ** 2).sum(dim=-1)  # [B, H, W]
        log_jac_per_patch = log_jac_det.view(B, 1, 1) / (H * W)
        nll_per_patch = -(log_pz_per_patch + log_jac_per_patch)

        # Focal weighting
        prob = torch.exp(-nll_per_patch.clamp(max=20))
        focal_weight = (1 - prob).pow(self.focal_gamma)

        return (focal_weight * nll_per_patch).mean()
```

### Why This Should Work
1. **Hard Sample Mining**: Normal distribution 경계에 있는 샘플에 집중
2. **Better Decision Boundary**: 어려운 패치를 잘 학습하면 anomaly 구분 향상
3. **Gradient Focus**: Easy sample은 gradient 기여 감소, hard sample에 gradient 집중

### Command Line Arguments
```bash
python run_moleflow.py \
    --focal_gamma 2.0 \  # Focal loss gamma (recommend 1.0-2.0)
    ...
```

### Expected Improvements
- 더 sharp한 normal distribution 경계 학습
- 특히 Task 0에서 base NF 품질 향상
- Anomaly detection 성능 전반적 향상

### v3 → v7
| File | Changes |
|------|---------|
| `moleflow/trainer/continual_trainer.py` | `_compute_nll_loss()` helper method, focal weighting |
| `run_moleflow.py` | `--focal_gamma` argument |
| `run.sh` | v7 experiment configuration |
