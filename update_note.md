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

---

## Baseline 1.5 → 2.0: Patch-wise Context Gate

### Motivation
**Baseline 1.5 (Global Alpha)의 문제점**:
- `alpha`는 global scalar → 모든 패치에 동일한 context 강도 적용
- 학습 중 **정상 데이터만** 봄 → anomaly-aware 학습 불가능
- 결과: `alpha ≈ 초기값`으로 고정, sigmoid bound가 의미 없음

**핵심 통찰**:
> Global alpha는 "knob"일 뿐이고,
> Anomaly detection에서는 "switch"가 필요하다.
> 그 switch는 **patch-wise gate**다.

### Core Idea
| 구분 | Baseline 1.5 (Global Alpha) | Baseline 2.0 (Patch-wise Gate) |
|------|----------------------------|-------------------------------|
| 수식 | `ctx = alpha * ctx` | `ctx = gate(x, ctx) * ctx` |
| 차원 | `alpha` ∈ ℝ (scalar) | `gate` ∈ ℝ^(B×H×W×1) (per-patch) |
| 학습 | 모든 패치 동일 | 패치별 독립 결정 |
| 정상 패치 | α * ctx | gate → 0 (context 무시) |
| 이상 패치 | α * ctx | gate → 1 (context 사용) |

### Code Changes

#### 1. MoLEContextSubnet (lora.py)

**Before (Baseline 1.5 - Global Alpha)**:
```python
class MoLEContextSubnet(nn.Module):
    def __init__(self, dims_in, dims_out, ...,
                 context_init_scale=0.1, context_max_alpha=0.2):
        # ...

        # Global alpha with sigmoid upper bound
        # alpha = alpha_max * sigmoid(alpha_param)
        p = min(max(context_init_scale / context_max_alpha, 0.01), 0.99)
        init_param = torch.log(torch.tensor([p / (1 - p)]))  # Inverse sigmoid
        self.context_scale_param = nn.Parameter(init_param)

    def forward(self, x):
        # ...
        ctx = self.context_conv(x_spatial)

        # Global alpha scaling (same for ALL patches)
        alpha = self.context_max_alpha * torch.sigmoid(self.context_scale_param)
        ctx = alpha * ctx  # (BHW, D) * scalar

        s_input = torch.cat([x, ctx], dim=-1)
        # ...
```

**After (Baseline 2.0 - Patch-wise Gate)**:
```python
class MoLEContextSubnet(nn.Module):
    def __init__(self, dims_in, dims_out, ...,
                 context_init_scale=0.1, context_max_alpha=0.2,
                 use_context_gate=False, context_gate_hidden=64):  # NEW
        # ...
        self.use_context_gate = use_context_gate

        if use_context_gate:
            # NEW: Patch-wise gate network
            # gate = sigmoid(MLP([x, ctx])) → (BHW, 1)
            self.context_gate_net = nn.Sequential(
                nn.Linear(dims_in * 2, context_gate_hidden),
                nn.ReLU(),
                nn.Linear(context_gate_hidden, 1)
            )
            # Initialize to output ~0 → gate starts at 0.5
            nn.init.zeros_(self.context_gate_net[0].weight)
            nn.init.zeros_(self.context_gate_net[0].bias)
            nn.init.zeros_(self.context_gate_net[2].weight)
            nn.init.zeros_(self.context_gate_net[2].bias)

            self.context_scale_param = None  # No global alpha
        else:
            # Legacy: Global alpha (Baseline 1.5)
            p = min(max(context_init_scale / context_max_alpha, 0.01), 0.99)
            init_param = torch.log(torch.tensor([p / (1 - p)]))
            self.context_scale_param = nn.Parameter(init_param)
            self.context_gate_net = None

    def forward(self, x):
        # ...
        ctx = self.context_conv(x_spatial)

        if self.use_context_gate and self.context_gate_net is not None:
            # NEW: Patch-wise gate (per-patch decision)
            gate_input = torch.cat([x, ctx], dim=-1)  # (BHW, 2D)
            gate_logit = self.context_gate_net(gate_input)  # (BHW, 1)
            gate = torch.sigmoid(gate_logit)  # (BHW, 1)

            self._last_gate = gate.detach()  # For logging
            ctx = gate * ctx  # (BHW, D) * (BHW, 1) → per-patch scaling
        else:
            # Legacy: Global alpha
            alpha = self.context_max_alpha * torch.sigmoid(self.context_scale_param)
            ctx = alpha * ctx

        s_input = torch.cat([x, ctx], dim=-1)
        # ...

    # NEW: Logging utilities
    def get_context_alpha(self) -> float:
        """Get global alpha value (legacy mode)."""
        if self.context_scale_param is not None:
            with torch.no_grad():
                return (self.context_max_alpha
                        * torch.sigmoid(self.context_scale_param)).item()
        return None

    def get_last_gate_stats(self) -> dict:
        """Get gate statistics (patch-wise mode)."""
        if hasattr(self, '_last_gate') and self._last_gate is not None:
            gate = self._last_gate
            return {
                'mean': gate.mean().item(),
                'std': gate.std().item(),
                'min': gate.min().item(),
                'max': gate.max().item()
            }
        return None
```

#### 2. AblationConfig (ablation.py)

**Added**:
```python
@dataclass
class AblationConfig:
    # ... existing fields ...

    # Scale-specific Context (Baseline 1.5)
    use_scale_context: bool = False
    scale_context_kernel: int = 3
    scale_context_init_scale: float = 0.1
    scale_context_max_alpha: float = 0.2

    # NEW: Patch-wise Context Gate (Baseline 2.0)
    use_context_gate: bool = False    # Use patch-wise gate instead of global alpha
    context_gate_hidden: int = 64     # Hidden dim for gate MLP
```

#### 3. MoLESpatialAwareNF (mole_nf.py)

**Before**:
```python
subnet = MoLEContextSubnet(
    dims_in, dims_out,
    rank=self.lora_rank,
    alpha=self.lora_alpha,
    use_lora=self.use_lora,
    use_task_bias=self.use_task_bias,
    context_kernel=self.scale_context_kernel,
    context_init_scale=self.scale_context_init_scale,
    context_max_alpha=self.scale_context_max_alpha
)
```

**After**:
```python
subnet = MoLEContextSubnet(
    dims_in, dims_out,
    rank=self.lora_rank,
    alpha=self.lora_alpha,
    use_lora=self.use_lora,
    use_task_bias=self.use_task_bias,
    context_kernel=self.scale_context_kernel,
    context_init_scale=self.scale_context_init_scale,
    context_max_alpha=self.scale_context_max_alpha,
    use_context_gate=self.use_context_gate,      # NEW
    context_gate_hidden=self.context_gate_hidden  # NEW
)
```

**Added get_context_info() method**:
```python
def get_context_info(self) -> dict:
    """Get context gate/alpha information for logging."""
    if not self.use_scale_context:
        return {}

    info = {}
    if self.use_context_gate:
        # Aggregate gate stats from all subnets
        gate_stats = []
        for subnet in self.subnets:
            if hasattr(subnet, 'get_last_gate_stats'):
                stats = subnet.get_last_gate_stats()
                if stats is not None:
                    gate_stats.append(stats)

        if gate_stats:
            info['gate_mean'] = sum(s['mean'] for s in gate_stats) / len(gate_stats)
            info['gate_std'] = sum(s['std'] for s in gate_stats) / len(gate_stats)
            info['gate_min'] = min(s['min'] for s in gate_stats)
            info['gate_max'] = max(s['max'] for s in gate_stats)
    else:
        # Collect alpha from all subnets
        alphas = [s.get_context_alpha() for s in self.subnets
                  if hasattr(s, 'get_context_alpha')]
        alphas = [a for a in alphas if a is not None]
        if alphas:
            info['alpha_mean'] = sum(alphas) / len(alphas)

    return info
```

#### 4. Trainer Logging (continual_trainer.py)

**Added context logging per epoch**:
```python
# In _train_base_task, _train_fast_stage, _train_slow_stage:
avg_epoch_loss = epoch_loss / max(num_batches, 1)
current_lr = optimizer.param_groups[0]['lr']

# NEW: Get context gate/alpha info for logging
extra_info = {"LR": current_lr}
context_info = self.nf_model.get_context_info()
if context_info:
    extra_info.update(context_info)

if self.logger:
    self.logger.log_epoch(task_id, epoch, num_epochs, avg_epoch_loss,
                          stage="FAST", extra_info=extra_info)
else:
    ctx_str = ""
    if 'gate_mean' in context_info:
        ctx_str = f" | Gate: {context_info['gate_mean']:.4f}±{context_info['gate_std']:.4f}"
    elif 'alpha_mean' in context_info:
        ctx_str = f" | Alpha: {context_info['alpha_mean']:.4f}"
    print(f"  📊 [FAST] Epoch [...] Average Loss: {avg_epoch_loss:.4f}{ctx_str}")
```

### Command Line Usage

```bash
# Baseline 1.5: Global Alpha (기존)
python run_moleflow.py \
    --use_scale_context \
    --scale_context_kernel 3 \
    --scale_context_init_scale 0.1 \
    --scale_context_max_alpha 0.2

# Baseline 2.0: Patch-wise Gate (NEW)
python run_moleflow.py \
    --use_scale_context \
    --use_context_gate \
    --context_gate_hidden 64
```

### Expected Improvements
- 패치별로 context 사용 여부 결정 → anomaly 경계에서 더 정밀한 detection
- Gate network가 normal/anomaly 패치 특성 학습
- 더 interpretable한 anomaly map 생성 가능

### Baseline 1.5 → 2.0
| File | Changes |
|------|---------|
| `moleflow/models/lora.py` | `MoLEContextSubnet` - context_gate_net 추가 |
| `moleflow/config/ablation.py` | `use_context_gate`, `context_gate_hidden` 설정 추가 |
| `moleflow/models/mole_nf.py` | context gate 파라미터 전달, `get_context_info()` 메서드 |
| `moleflow/trainer/continual_trainer.py` | Context gate/alpha 로깅 추가 |

---

## Version 3 - No-Replay Continual Learning Solutions

### Motivation
Version 2에서 continual learning 시 성능 저하 문제가 여전히 존재:
- Task 0 → Task 1 → Task 2 학습 시 이전 task 성능 감소 (Catastrophic Forgetting)
- 기존 방법: Replay buffer 사용 → 메모리 비용, 프라이버시 문제

**목표**: Replay 없이 continual learning 성능 향상

### V3 New Modules Overview

| Module | 목적 | 위치 |
|--------|------|------|
| **WhiteningAdapter** | Task-agnostic feature normalization | Feature → NF 전 |
| **LightweightMSContext** | Multi-scale receptive field 확장 | NF 입력 전 |
| **DeepInvertibleAdapter (DIA)** | Task-specific nonlinear manifold adaptation | NF 출력 후 |
| **OrthogonalGradientProjection (OGP)** | Gradient projection to null space | Training loop |
| **TwoStageHybridRouter** | Prototype + Likelihood routing | Inference |

---

## V3-1: WhiteningAdapter

### Core Idea
Task 간 feature distribution shift 문제 해결:
```
Task 0 features: mean=μ₀, cov=Σ₀
Task 1 features: mean=μ₁, cov=Σ₁  (다른 분포)
```

**Solution**: Whitening → Constrained De-whitening
```
x → Whiten(x) → z (zero mean, unit variance)
z → ConstrainedDewhiten(z) → x' (controlled distribution)
```

### Implementation
```python
# moleflow/models/adapters.py
class WhiteningAdapter(nn.Module):
    """
    Whitening-based Task Adapter (V3 Solution 3).

    Key Design:
    1. All tasks go through Whitening first (mean=0, std=1) via LayerNorm
    2. Task-specific de-whitening with constrained gamma/beta parameters
    3. Task 0 stays close to identity (anchor point)

    Parameters:
    - gamma: constrained to [gamma_min, gamma_max] via sigmoid
    - beta: constrained to [-beta_max, beta_max] via tanh
    """
    def __init__(self, channels: int, task_id: int = 0,
                 reference_mean=None, reference_std=None,
                 gamma_range: tuple = (0.5, 2.0), beta_max: float = 2.0):
        super().__init__()
        self.gamma_min, self.gamma_max = gamma_range
        self.beta_max = beta_max

        # Whitening layer (shared across all tasks, no learnable affine)
        self.whiten = nn.LayerNorm(channels, elementwise_affine=False)

        if task_id == 0:
            # Task 0: Start very close to identity
            # gamma ≈ 1.0, beta ≈ 0.0
            init_gamma_raw = -0.7 * torch.ones(1, 1, 1, channels)
            self.gamma_raw = nn.Parameter(init_gamma_raw)
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.1  # Regularize toward identity
        else:
            # Task 1+: Learnable, initialized at midpoint
            self.gamma_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.0

    @property
    def gamma(self):
        """Constrained gamma in [gamma_min, gamma_max]."""
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(self.gamma_raw)

    @property
    def beta(self):
        """Constrained beta in [-beta_max, beta_max]."""
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D = x.shape
        # 1. Whitening: normalize to N(0, 1)
        x_white = self.whiten(x.reshape(-1, D)).reshape(B, H, W, D)
        # 2. Task-specific de-whitening
        return self.gamma * x_white + self.beta

    def identity_regularization(self) -> torch.Tensor:
        """Regularization loss to keep Task 0 adapter close to identity."""
        if self.identity_reg_weight > 0:
            gamma_reg = ((self.gamma - 1.0) ** 2).mean()
            beta_reg = (self.beta ** 2).mean()
            return self.identity_reg_weight * (gamma_reg + beta_reg)
        return torch.tensor(0.0, device=self.gamma_raw.device)
```

### Key Design
1. **LayerNorm-based Whitening**: Task-agnostic normalization (no learnable params)
2. **Constrained Parameters**: sigmoid/tanh로 범위 제한 → 안정적 학습
3. **Per-Task Adapter**: 각 task마다 별도 WhiteningAdapter (create_task_adapter factory 함수 사용)
4. **Task 0 Identity Regularization**: Task 0는 identity에 가깝게 유지

### Command Line
```bash
python run_moleflow.py --use_whitening_adapter
```

---

## V3-2: LightweightMSContext (Multi-Scale Context)

### Core Idea
기존 NF는 patch 단위 독립 처리 → 주변 context 무시

**Solution**: Multi-scale dilated convolution으로 receptive field 확장
```
x → [Conv_d1, Conv_d2, Conv_d4] → concat → fusion → x + context
```

### Implementation
```python
# moleflow/models/ms_context.py
class LightweightMSContext(nn.Module):
    """
    Multi-scale context via dilated depthwise convolutions.

    Uses multiple dilation rates to capture context at different scales
    without significantly increasing parameters.
    """
    def __init__(self, channels, dilations=[1, 2, 4], kernel_size=3):
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size,
                      padding=d*(kernel_size//2), dilation=d, groups=channels)
            for d in dilations
        ])
        self.fusion = nn.Conv2d(channels * len(dilations), channels, 1)
        self.gate = nn.Parameter(torch.zeros(1))  # Starts at 0.5 after sigmoid

    def forward(self, x):
        # x: (B, H, W, D) → (B, D, H, W) for conv
        x_conv = x.permute(0, 3, 1, 2)

        # Multi-scale features
        ms_features = [conv(x_conv) for conv in self.dilated_convs]
        ms_concat = torch.cat(ms_features, dim=1)

        # Fusion and gating
        context = self.fusion(ms_concat).permute(0, 2, 3, 1)
        gate = torch.sigmoid(self.gate)

        return x + gate * context
```

### Key Design
1. **Depthwise Separable**: 파라미터 효율적
2. **Multiple Dilations**: d=1,2,4로 다양한 scale 포착
3. **Learnable Gate**: 학습 초기 안정성

### Command Line
```bash
python run_moleflow.py --use_ms_context
```

### ⚠️ Warning: WhiteningAdapter + MS-Context 충돌

두 모듈을 동시 사용 시 학습 불안정 발생:
- Loss가 음수로 발산
- Task 0 성능 급격히 저하

**원인**: 두 모듈 모두 NF 입력 전에 feature를 변환하여 distribution 충돌

**자동 해결**: `AblationConfig`에서 자동으로 MS-Context 비활성화
```python
# ablation.py __post_init__()
if self.use_whitening_adapter and self.use_ms_context:
    print("⚠️  Warning: use_whitening_adapter + use_ms_context 조합은 학습 불안정")
    self.use_ms_context = False
```

---

## V3-3: DeepInvertibleAdapter (DIA)

### Core Idea
Base NF 출력 후 task-specific nonlinear adaptation:
```
x → Base NF → z_base → DIA_task → z_final
```

**Why DIA?**
- LoRA: Linear adaptation (표현력 제한)
- DIA: Invertible nonlinear adaptation (더 강력한 manifold adaptation)

### Implementation
```python
# moleflow/models/lora.py
class DeepInvertibleAdapter(nn.Module):
    """
    Deep Invertible Adapter (DIA) - V3 Solution 1 (No Replay).

    Key Insight:
    Instead of linear LoRA (W + BA), we add a small task-specific Flow
    AFTER the base NF. This allows nonlinear manifold adaptation.

    Architecture:
    - Base NF: Frozen after Task 0 (extracts common features)
    - DIA: 1-2 lightweight coupling blocks per task (learns task-specific warping)

    Mathematical Formulation:
    - Base: z_base = f_base(x)
    - DIA:  z_final = f_DIA_t(z_base)
    - log p(x) = log p(z_final) + log|det J_base| + log|det J_DIA|
    """
    def __init__(self, channels: int, task_id: int, n_blocks: int = 2,
                 hidden_ratio: float = 0.5, clamp_alpha: float = 1.9):
        super().__init__()
        self.clamp_alpha = clamp_alpha
        hidden_dim = int(channels * hidden_ratio)

        # Build mini-flow: sequence of affine coupling blocks
        self.coupling_blocks = nn.ModuleList([
            AffineCouplingBlock(
                channels=channels,
                hidden_dim=hidden_dim,
                clamp_alpha=clamp_alpha,
                reverse=(i % 2 == 1)  # Alternate which half is transformed
            ) for i in range(n_blocks)
        ])
        self._initialize_near_identity()

    def _initialize_near_identity(self):
        """Initialize to near-identity transformation."""
        for block in self.coupling_blocks:
            nn.init.zeros_(block.s_net.layers[-1].weight)
            nn.init.zeros_(block.s_net.layers[-1].bias)
            nn.init.zeros_(block.t_net.layers[-1].weight)
            nn.init.zeros_(block.t_net.layers[-1].bias)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        B, H, W, D = x.shape
        log_det = torch.zeros(B, H, W, device=x.device)
        blocks = reversed(self.coupling_blocks) if reverse else self.coupling_blocks

        for block in blocks:
            x, block_log_det = block(x, reverse=reverse)
            log_det = log_det + block_log_det
        return x, log_det


class AffineCouplingBlock(nn.Module):
    """Affine Coupling Block for DIA with clamped scale."""
    def __init__(self, channels: int, hidden_dim: int,
                 clamp_alpha: float = 1.9, reverse: bool = False):
        super().__init__()
        self.clamp_alpha = clamp_alpha
        self.reverse_split = reverse
        self.split_dim = channels // 2

        # Scale network: x1 -> s
        self.s_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)
        # Translation network: x1 -> t
        self.t_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        B, H, W, D = x.shape
        if self.reverse_split:
            x2, x1 = x[..., :self.split_dim], x[..., self.split_dim:]
        else:
            x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        x1_flat = x1.reshape(-1, self.split_dim)
        s = self.s_net(x1_flat).reshape(B, H, W, self.split_dim)
        t = self.t_net(x1_flat).reshape(B, H, W, self.split_dim)
        s = self.clamp_alpha * torch.tanh(s / self.clamp_alpha)

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)

        if self.reverse_split:
            return torch.cat([y2, x1], dim=-1), log_det
        return torch.cat([x1, y2], dim=-1), log_det


class SimpleSubnet(nn.Module):
    """Simple MLP subnet for DIA coupling blocks."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # Initialize output layer to zero for identity start
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)
```

### Integration in mole_nf.py
```python
class MoLESpatialAwareNF(nn.Module):
    def __init__(self, ...):
        # ...
        self.dia_adapters = nn.ModuleDict()  # Per-task DIA

    def add_task(self, task_id):
        # ...
        if self.use_dia:
            self.dia_adapters[str(task_id)] = DeepInvertibleAdapter(
                channels=self.embed_dim,
                task_id=task_id,
                n_blocks=self.dia_n_blocks,
                hidden_ratio=self.dia_hidden_ratio,
                clamp_alpha=self.clamp_alpha
            ).to(self.device)

    def forward(self, x, reverse=False):
        # Base NF forward
        z, logdet = self.flow(x)

        # DIA forward (applied AFTER base NF)
        if self.use_dia and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.dia_adapters:
                z, dia_logdet = self.dia_adapters[task_key](z, reverse=reverse)
                logdet = logdet + dia_logdet

        return z, logdet
```

### Command Line
```bash
python run_moleflow.py \
    --use_dia \
    --dia_n_blocks 2 \
    --dia_hidden_ratio 0.5
```

---

## V3-4: OrthogonalGradientProjection (OGP)

### Core Idea
이전 task에서 중요한 gradient 방향을 보존:
```
∇L_new → Project to null space of previous tasks → ∇L_projected
```

**Gradient Projection**:
```
g' = g - Σᵢ (basis_i @ basis_i.T @ g)
```
where basis_i = important gradient directions from task i

### Implementation
```python
# moleflow/utils/replay.py
class OrthogonalGradientProjection:
    """
    Orthogonal Gradient Projection (OGP) - V3 No-Replay Solution.

    Key Idea:
    After learning Task t, compute the principal subspace of gradients
    (or features) that are important for that task. When learning Task t+1,
    project gradients to be orthogonal to this subspace, ensuring that
    updates don't interfere with previously learned knowledge.

    This is based on GPM (Gradient Projection Memory) from:
    "Continual Learning in Low-rank Orthogonal Subspaces", NeurIPS 2020

    Mathematical Formulation:
    1. After Task t: Compute U_t = SVD(G_t)[:, :k] where G_t is gradient matrix
    2. Store basis vectors (Vh transposed from SVD)
    3. For Task t+1: g' = g - basis @ (basis.T @ g) for each stored basis

    Advantages over Replay:
    - No data storage required
    - Memory: O(d × k) per task where k << d
    - Mathematically guarantees no interference in stored subspace
    """
    def __init__(self, threshold: float = 0.99, max_rank_per_task: int = 50,
                 device: str = 'cuda'):
        self.threshold = threshold
        self.max_rank_per_task = max_rank_per_task
        self.device = device

        # Store projection bases per parameter
        # param_name -> list of basis matrices (one per task)
        self.bases: Dict[str, List[torch.Tensor]] = {}
        self.is_initialized = False
        self.n_tasks = 0

    def compute_and_store_basis(self, model: nn.Module, data_loader,
                                 task_id: int, n_samples: int = 300):
        """
        Compute gradient subspace basis for completed task.
        Called AFTER training on a task is complete.
        """
        model.eval()
        gradient_matrices: Dict[str, List[torch.Tensor]] = {}

        n_processed = 0
        for batch in data_loader:
            if n_processed >= n_samples:
                break
            features = batch[0].to(self.device)
            batch_size = features.shape[0]

            model.zero_grad()
            log_prob = model.log_prob(features)
            loss = -log_prob.mean()
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad = param.grad.detach().flatten()
                    if name not in gradient_matrices:
                        gradient_matrices[name] = []
                    gradient_matrices[name].append(grad.clone())
            n_processed += batch_size

        # Compute SVD for each parameter's gradient matrix
        for name, grads in gradient_matrices.items():
            if len(grads) < 5:
                continue
            G = torch.stack(grads, dim=0)  # (n_samples, n_params)

            U, S, Vh = torch.linalg.svd(G, full_matrices=False)

            # Select top-k components based on variance threshold
            var_ratio = (S ** 2).cumsum(0) / (S ** 2).sum()
            k = min(
                (var_ratio < self.threshold).sum().item() + 1,
                self.max_rank_per_task,
                S.shape[0]
            )

            # Store basis vectors: Vh[:k, :].T gives (n_params, k)
            basis = Vh[:k, :].T  # (n_params, k)

            if name not in self.bases:
                self.bases[name] = []
            self.bases[name].append(basis.to(self.device))

        self.n_tasks = task_id + 1
        self.is_initialized = True

    def project_gradient(self, model: nn.Module):
        """
        Project current gradients to be orthogonal to stored subspaces.
        Call AFTER loss.backward() and BEFORE optimizer.step().
        """
        if not self.is_initialized:
            return

        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if name not in self.bases:
                continue

            grad = param.grad.flatten()

            # Project out all stored subspaces for this parameter
            for basis in self.bases[name]:
                # basis: (n_params, k)
                # proj = basis @ (basis.T @ grad)
                proj = basis @ (basis.T @ grad)
                grad = grad - proj

            param.grad = grad.reshape(param.shape)

    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics."""
        total_elements = sum(b.numel() for bl in self.bases.values() for b in bl)
        return {
            'n_params': len(self.bases),
            'n_tasks': self.n_tasks,
            'total_elements': total_elements,
            'memory_mb': total_elements * 4 / (1024 * 1024)
        }
```

### Integration in Trainer
```python
# moleflow/trainer/continual_trainer.py
class MoLEContinualTrainer:
    def __init__(self, ...):
        if self.use_ogp:
            self.ogp = OrthogonalGradientProjection(
                threshold=self.ogp_threshold,
                max_rank_per_task=self.ogp_max_rank,
                device=device
            )

    def _train_fast_stage(self, task_id, ...):
        for batch in dataloader:
            loss.backward()

            # OGP: Project gradients for Task > 0
            if self.use_ogp and self.ogp is not None and self.ogp.is_initialized:
                self.ogp.project_gradient(self.nf_model)

            optimizer.step()

    def train_task(self, task_id, ...):
        # ... training code ...

        # Compute OGP basis AFTER task training completes
        if self.use_ogp and self.ogp is not None:
            self._compute_ogp_basis(task_id, train_loader)

    def _compute_ogp_basis(self, task_id, train_loader):
        """Compute and store OGP gradient basis for completed task."""
        # Creates FeatureDataLoader wrapper to provide features
        self.ogp.compute_and_store_basis(
            model=self.nf_model,
            data_loader=feature_loader,
            task_id=task_id,
            n_samples=self.ogp_n_samples
        )
```

### Command Line
```bash
python run_moleflow.py \
    --use_ogp \
    --ogp_threshold 0.99 \
    --ogp_max_rank 50 \
    --ogp_n_samples 300
```

---

## V3-5: TwoStageHybridRouter

### Core Idea
기존 Router는 Prototype matching만 사용 → 유사한 task 구분 어려움

**Solution**: Two-stage routing
1. **Stage 1 (Fast)**: Prototype filtering → Top-K candidates
2. **Stage 2 (Accurate)**: NF likelihood comparison → Final selection

### Implementation
```python
# moleflow/models/routing.py
class TwoStageHybridRouter(nn.Module):
    """
    Two-stage routing: Prototype filtering + Likelihood refinement.

    Stage 1: Mahalanobis distance to prototypes → Top-K candidates
    Stage 2: NF log-likelihood for final selection
    """
    def __init__(self, nf_model, top_k=2):
        self.prototype_router = PrototypeRouter()
        self.nf_model = nf_model
        self.top_k = top_k

    def forward(self, features):
        # Stage 1: Prototype distances
        distances = self.prototype_router.compute_distances(features)
        top_k_tasks = distances.argsort()[:self.top_k]

        # Stage 2: NF likelihood
        likelihoods = []
        for task_id in top_k_tasks:
            self.nf_model.set_task(task_id)
            z, logdet = self.nf_model(features)
            log_prob = -0.5 * (z**2).sum() + logdet
            likelihoods.append(log_prob)

        # Select task with highest likelihood
        best_idx = torch.stack(likelihoods).argmax()
        return top_k_tasks[best_idx]
```

### Command Line
```bash
python run_moleflow.py \
    --use_hybrid_router \
    --router_top_k 2
```

---

## V3 Experiment Results

### Ablation Study (leather → grid → transistor)

| Configuration | Image AUC | Pixel AUC | Notes |
|---------------|-----------|-----------|-------|
| **Baseline (V2)** | 0.8168 | 0.9166 | - |
| DIA only | 0.8217 | 0.9277 | +0.5% Image, +1.1% Pixel |
| OGP only | 0.8180 | 0.9161 | Minimal change |
| **DIA + OGP** | **0.8226** | **0.9231** | **Best combination** |
| WhiteningAdapter only | (실험 중) | (실험 중) | - |
| MS-Context only | (실험 중) | (실험 중) | - |
| All V3 (conflict) | 0.4471 | 0.5527 | ❌ 실패 (충돌) |

### Key Findings
1. **DIA + OGP**: Best performance without replay
2. **WhiteningAdapter + MS-Context**: 조합 시 충돌 → 자동 비활성화 처리
3. **DIA > OGP**: DIA가 더 큰 성능 향상 기여

---

## V3 File Changes Summary

| File | Changes |
|------|---------|
| `moleflow/models/adapters.py` | **WhiteningAdapter** 추가 (line 417-531), `create_task_adapter` factory에 "whitening" 모드 추가 |
| `moleflow/models/lora.py` | **LightweightMSContext** (line 223-366), **TaskConditionedMSContext** (line 373-701), **DeepInvertibleAdapter** + AffineCouplingBlock + SimpleSubnet (line 708-908) 추가 |
| `moleflow/utils/replay.py` | **OrthogonalGradientProjection** (line 512-687), GradientProjectionHook, FeatureBank, DistillationLoss, EWC 추가 |
| `moleflow/models/mole_nf.py` | DIA integration (`dia_adapters`), WhiteningAdapter/MSContext/TaskConditionedMSContext 통합, V3 config options 처리 |
| `moleflow/config/ablation.py` | V3 options: `use_dia`, `use_ogp`, `use_whitening_adapter`, `use_ms_context`, `use_task_conditioned_ms_context`, `ogp_*` params, `dia_*` params |
| `moleflow/trainer/continual_trainer.py` | OGP integration: `_compute_ogp_basis()`, `ogp.project_gradient()` in training loop |
| `run_moleflow.py` | V3 CLI arguments, config saving |

---

## V3 Command Line Reference

```bash
# Full V3 with recommended settings (DIA + OGP)
python run_moleflow.py \
    --use_dia \
    --dia_n_blocks 2 \
    --use_ogp \
    --ogp_threshold 0.99 \
    --ogp_max_rank 50 \
    --experiment_name Version3-DIA_OGP

# WhiteningAdapter only
python run_moleflow.py \
    --use_whitening_adapter \
    --experiment_name Version3-WhiteningAdapter

# MS-Context only (automatically disabled if WhiteningAdapter is on)
python run_moleflow.py \
    --use_ms_context \
    --experiment_name Version3-MSContext

# All options with diagnostics
python run_moleflow.py \
    --run_diagnostics \
    --use_dia \
    --use_ogp \
    --use_whitening_adapter \
    --experiment_name Version3-All
```

---

## V3 Problem Analysis 
--
  심층 분석 결과

  1. 핵심 발견: 공유 파라미터 문제

  spatial_mixer가 모든 task에서 공유되고 학습됩니다:

  # mole_nf.py:529-530
  if self.spatial_mixer is not None and not reverse:
      x = self.spatial_mixer(x)

  # mole_nf.py:721-722 (get_fast_params)
  if self.spatial_mixer is not None:
      params.extend(self.spatial_mixer.parameters())  # 모든 task에서 학습!

  Forward 경로:
  Input → WhiteningAdapter[task] → SpatialMixer[SHARED] → BaseNF → DIA[task] → Output
                                         ↑
                              이것이 매 task마다 업데이트됨!

  ---
  2. Task 0 붕괴 메커니즘 상세 분석

  Phase 1: Task 0 학습 (정상)

  leather features → WhiteningAdapter_0 → SpatialMixer_v0 → BaseNF → DIA_0
                                                ↑
                                       이 시점의 파라미터
  - Image AUC = 1.0 달성

  Phase 2: Task 1-6 학습 (점진적 드리프트)

  grid/transistor/... → WhiteningAdapter_k → SpatialMixer_v1...v6 → BaseNF → DIA_k
                                                    ↑
                                           매 task마다 업데이트
  - leather 평가 시: SpatialMixer_v6이 사용됨 (원래 v0과 다름)
  - 하지만 OGP가 어느 정도 gradient를 제한하여 드리프트 최소화
  - leather Image AUC: 1.0 → 0.99 (미미한 감소)

  Phase 3: Task 7-8 학습 (붕괴 시작)

  metal_nut/screw → WhiteningAdapter_k → SpatialMixer_v7,v8 → BaseNF → DIA_k
                                                ↑
                                        OGP 제약이 포화 상태
  - OGP의 orthogonal 공간이 점점 줄어듦
  - SpatialMixer가 새로운 task에 적응하려면 이전 subspace에서 벗어나야 함
  - leather Image AUC: 0.99 → 0.78 → 0.55 (급격한 감소)

  Phase 4: Task 9+ 학습 (완전 붕괴)

  - OGP 제약 하에서도 SpatialMixer가 계속 변경됨
  - leather의 representation이 완전히 손상됨
  - leather Image AUC: 0.07 (거의 random)

  ---
  3. 알파벳 순서에서 더 빠른 붕괴 원인

  | 비교   | 원래 순서        | 알파벳 순서         |
  |--------|------------------|---------------------|
  | Task 0 | leather (텍스처) | bottle (형상)       |
  | Task 2 | transistor       | capsule             |
  | 유사성 | 낮음             | 높음 (둘 다 원통형) |

  bottle → capsule 유사성 문제:
  1. bottle의 feature로 학습된 SpatialMixer
  2. capsule 학습 시 비슷한 feature 공간에서 경쟁
  3. OGP가 보존하려는 subspace와 새 task가 필요한 공간이 크게 중첩
  4. 결과: bottle의 representation이 2개 task 만에 손상됨

  leather → grid 비유사성:
  1. leather의 텍스처 feature로 학습된 SpatialMixer
  2. grid (패턴), transistor (형상) 등은 다른 feature 공간
  3. OGP가 보존하는 subspace와 새 task 공간의 중첩이 적음
  4. 결과: 7개 task까지 상대적으로 안정

  ---
  4. Image AUC vs Pixel AUC 차이 분석

  leather (원래 순서)

  | Metric    | Task 0 | Task 14 | 설명     |
  |-----------|--------|---------|----------|
  | Pixel AUC | 0.97   | 0.66    | 32% 하락 |
  | Image AUC | 1.00   | 0.07    | 93% 하락 |

  해석:
  - Pixel-level은 patch별 anomaly score = -log p(z) - log|det J|
  - Image-level은 patch scores의 99th percentile
  - Pixel AUC > 0.5: patch들 간의 상대적 순서는 어느 정도 유지됨
  - Image AUC ≈ 0: 모든 이미지의 99th percentile 값이 전체적으로 shift되어 normal/anomaly 구분 불가

  의미: SpatialMixer 변경으로 전체 score 분포가 shift됨. 개별 patch의 상대적 차이는 유지되지만, image-level 통계가 무의미해짐.

  bottle (알파벳 순서)

  | Metric    | Task 0 | Task 14 | 설명         |
  |-----------|--------|---------|--------------|
  | Pixel AUC | 0.94   | 0.12    | Random 이하! |
  | Image AUC | 1.00   | 0.30    | 70% 하락     |

  해석:
  - Pixel AUC = 0.12 < 0.5: 모델이 반대로 예측
  - Anomaly patch에 낮은 score, Normal patch에 높은 score
  - 이는 단순한 distribution shift가 아니라 representation 자체의 inversion

  의미: SpatialMixer가 bottle의 feature space를 완전히 뒤집어버림. 더 심각한 손상.

  ---
  5. OGP의 한계

  OGP는 gradient를 이전 task의 중요 subspace에 orthogonal하게 투영:

  g' = g - Σ(basis_i @ basis_i^T @ g)

  문제점:
  1. 누적 제약: 15개 task × 50 rank = 750개의 orthogonal 방향 제약
  2. 공유 파라미터에 적용: SpatialMixer의 gradient도 투영됨
  3. 불완전한 보호: 투영 후에도 남은 gradient가 공유 파라미터를 변경

  Task 0 basis: [v1, v2, ..., v50]  (leather 중요 방향)
  Task 1 학습: g' = g - proj(g, Task0_basis)
                ↑
         이 g'가 SpatialMixer를 변경 → leather 손상

  ---
  6. 핵심 문제 요약

  | 문제                | 설명                                  | 심각도      |
  |---------------------|---------------------------------------|-------------|
  | 공유 SpatialMixer   | 모든 task에서 학습되어 이전 task 손상 | 🔴 Critical |
  | OGP 누적 제약       | Task 수 증가 시 학습 공간 축소        | 🟠 High     |
  | 클래스 순서 의존성  | 유사 클래스가 연속되면 더 빠른 붕괴   | 🟡 Medium   |
  | Task 0 Base NF 편향 | 첫 task에만 최적화된 Base NF          | 🟡 Medium   |

  ---
  7. Routing이 100% 정확한데도 붕괴하는 이유

  Evaluation: leather test image
    ↓
  Router: "This is leather" (100% 정확)
    ↓
  WhiteningAdapter_0: leather용 adapter ✓
    ↓
  SpatialMixer: Task 14 이후 버전 ✗  ← 여기서 문제!
    ↓
  BaseNF + LoRA_0: leather용 ✓
    ↓
  DIA_0: leather용 ✓
    ↓
  Wrong output due to SpatialMixer mismatch

  결론: Task-specific 컴포넌트는 정상이지만, 공유 컴포넌트(SpatialMixer)의 드리프트가 전체 파이프라인을 오염시킴.

---

## V3 근본적 문제와 해결 방향

### 근본적 문제 진단

V3의 가정:
> "Task 0에서 학습된 Base NF가 모든 task에 범용적으로 적용 가능하고, LoRA/DIA로 task-specific adaptation만 하면 된다"

**이 가정이 틀린 이유:**

1. **Base NF의 본질적 편향**
   - Base NF는 Task 0 (leather 또는 bottle)의 "normal = 무결함" 분포만 학습
   - 다른 task의 normal distribution과 근본적으로 다름
   - LoRA/DIA는 "fine-tuning"일 뿐, transformation 자체를 바꿀 수 없음

2. **공유 파라미터의 치명적 영향**
   - SpatialMixer가 모든 task에서 학습됨
   - OGP는 gradient를 제한할 뿐, 완전한 보호 불가
   - Task 수 증가 시 OGP 제약 공간 포화 → 이전 task 손상

3. **LoRA/DIA의 표현력 한계**
   - LoRA: `W + BA` (저차원 linear adaptation)
   - DIA: 작은 flow block (2 coupling layers)
   - Base NF가 잘못된 변환을 하면 이를 보정하기 어려움

### 가능한 해결 방향

| 접근법 | 설명 | 장점 | 단점 |
|--------|------|------|------|
| **사전학습 Base NF** | 다양한 domain에서 Base NF 사전학습 | Task-agnostic 표현 | 사전학습 데이터 필요 |
| **완전 분리** | 모든 trainable 파라미터 task-specific | 간섭 원천 차단 | 메모리 증가 |
| **Replay 기반** | 이전 task 데이터 일부 저장 | 직접적 forgetting 방지 | Privacy, 저장 비용 |

### 채택 방향: 완전 분리 (Complete Separation)

**선택 이유:**
1. 근본 원인(공유 파라미터 드리프트)을 원천 차단
2. 추가 데이터 수집/저장 불필요
3. 구현 복잡도 낮음
4. 확장성 보장 (task 수 증가에도 안정)

---

## V4 - Complete Separation Architecture

### 핵심 원칙
> "모든 학습 가능한 파라미터는 task-specific이어야 한다"

### Architecture Overview

```
V3 (문제):
Input → WhiteningAdapter[task] → SpatialMixer[SHARED+Trained] → BaseNF[frozen] + LoRA[task] → DIA[task] → Output
                                         ↑
                                    모든 task에서 학습 → 드리프트

V4 (해결):
Input → WhiteningAdapter[task] → SpatialMixer[FROZEN] → BaseNF[frozen] + LoRA[task] → DIA[task] → Output
                                         ↑
                                    Task 0 이후 완전 동결
```

### 핵심 변경사항

| 컴포넌트 | V3 | V4 | 변경 이유 |
|----------|-----|-----|----------|
| **SpatialMixer** | 모든 task에서 학습 | Task 0 이후 freeze | 공유 파라미터 드리프트 방지 |
| **OGP** | 활성화 | 제거 (불필요) | 공유 파라미터 없음 → 투영 불필요 |
| **WhiteningAdapter** | task별 | task별 (유지) | 이미 완전 분리됨 |
| **LoRA** | task별 | task별 (유지) | 이미 완전 분리됨 |
| **DIA** | task별 | task별 (유지) | 이미 완전 분리됨 |

### 학습 프로토콜

**Task 0 (Base Training)**:
```python
# 학습 대상: SpatialMixer + BaseNF + LoRA_0 + WhiteningAdapter_0 + DIA_0
trainable = [
    spatial_mixer.parameters(),      # Task 0에서만 학습
    base_nf.parameters(),            # Task 0에서만 학습
    lora_adapters["0"].parameters(),
    whitening_adapters["0"].parameters(),
    dia_adapters["0"].parameters()
]
```

**Task 1+ (Adapter Only)**:
```python
# 학습 대상: LoRA_t + WhiteningAdapter_t + DIA_t (공유 파라미터 완전 freeze)
trainable = [
    lora_adapters[str(task_id)].parameters(),
    whitening_adapters[str(task_id)].parameters(),
    dia_adapters[str(task_id)].parameters()
]
# SpatialMixer, BaseNF는 완전 freeze
```

### 구현 변경사항

#### 1. mole_nf.py - get_fast_params() 수정

**Before (V3)**:
```python
def get_fast_params(self, task_id: int) -> List[nn.Parameter]:
    params = []
    # ... LoRA, WhiteningAdapter, DIA params ...

    # SpatialMixer가 모든 task에서 학습됨 ← 문제!
    if self.spatial_mixer is not None:
        params.extend(self.spatial_mixer.parameters())

    return params
```

**After (V4)**:
```python
def get_fast_params(self, task_id: int) -> List[nn.Parameter]:
    params = []
    # ... LoRA, WhiteningAdapter, DIA params ...

    # V4: SpatialMixer는 Task 0에서만 학습, 이후 freeze
    if self.spatial_mixer is not None and task_id == 0:
        params.extend(self.spatial_mixer.parameters())

    return params
```

#### 2. continual_trainer.py - OGP 제거

**V4에서 OGP가 불필요한 이유:**
- OGP는 "공유 파라미터"의 gradient를 투영하여 이전 task 보호
- V4에서는 공유 파라미터가 모두 frozen → 보호할 대상 없음
- OGP 연산 오버헤드 제거 → 학습 속도 향상

```python
# V4: OGP 비활성화
if self.use_ogp:
    warnings.warn("V4 Complete Separation: OGP is unnecessary and will be disabled")
    self.use_ogp = False
```

#### 3. run.sh - V4 실험 설정

```bash
# V4: Complete Separation (WhiteningAdapter + DIA, no OGP, frozen SpatialMixer)
python run_moleflow.py \
    --task_classes leather grid transistor carpet zipper hazelnut \
                   toothbrush metal_nut screw wood tile capsule pill cable bottle \
    --use_whitening_adapter \
    --use_dia \
    --dia_n_blocks 2 \
    --no_ogp \                  # V4: OGP 비활성화
    --freeze_spatial_mixer \    # V4: SpatialMixer Task 0 이후 freeze
    --experiment_name Version4-CompleteSeparation
```

### 예상 결과

| 지표 | V3 (15 classes) | V4 예상 | 근거 |
|------|-----------------|---------|------|
| Task 0 Image AUC | 0.07~0.30 | 0.90+ | SpatialMixer 드리프트 없음 |
| Mean Image AUC | 0.72 | 0.85+ | 모든 task 안정 |
| Routing Acc | 99.76% | 99%+ | 유지 (Router는 변경 없음) |
| 학습 속도 | 1x | 1.2x+ | OGP 연산 제거 |

### V4 구현 체크리스트

- [x] `mole_nf.py`: `get_fast_params()`에서 SpatialMixer task_id 조건 추가
  - Line 720-725: `if self.spatial_mixer is not None and task_id == 0:`
- [x] `mole_nf.py`: `freeze_fast_params()` Task 0 조건 추가
  - Line 759-762: SpatialMixer freeze only for Task 0
- [x] `mole_nf.py`: `unfreeze_fast_params()` Task 0 조건 추가
  - Line 794-797: SpatialMixer unfreeze only for Task 0
- [x] `run.sh`: V4 실험 스크립트 추가
  - `--use_whitening_adapter --use_dia` (no OGP)
  - All 15 classes in alphabetical order

**Note**: 별도 config 옵션 불필요 - `task_id == 0` 조건으로 자동 처리됨

### V4 File Changes Summary

| File | Changes |
|------|---------|
| `moleflow/models/mole_nf.py` | `get_fast_params()`, `freeze_fast_params()`, `unfreeze_fast_params()` - SpatialMixer는 task_id == 0일 때만 학습 |
| `run.sh` | V4 실험 스크립트: `Version4-CompleteSeparation_all_classes_alphabet` |

---

## V4 Experiment Results (15 Classes)

### Catastrophic Forgetting 해결 ✅

| 순서 | Task 0 | V3 After Task 14 | V4 After Task 14 | 개선 |
|------|--------|------------------|------------------|------|
| Original | leather | 0.07 (-93%) | **1.00 (0%)** | ✅ 완전 해결 |
| Alphabet | bottle | 0.30 (-70%) | **0.999 (-0.08%)** | ✅ 완전 해결 |

### 전체 성능 비교

| 지표 | V3 Original | V4 Original | V4 Alphabet |
|------|-------------|-------------|-------------|
| Mean Image AUC | 0.7716 | **0.8636** | 0.8564 |
| Mean Pixel AUC | 0.9009 | **0.9272** | 0.9245 |
| Routing Acc | 99.76% | 99.76% | 99.76% |

**V4 vs V3: Mean Image AUC +12% 향상**

### 클래스별 상세 결과 (V4 Original Order)

| Class | Task ID | Image AUC | Pixel AUC | Gap |
|-------|---------|-----------|-----------|-----|
| leather | 0 | 1.000 | 0.972 | +0.03 |
| grid | 1 | 0.842 | 0.908 | -0.07 |
| transistor | 2 | 0.773 | 0.926 | -0.15 |
| carpet | 3 | 0.968 | 0.965 | +0.00 |
| zipper | 4 | 0.935 | 0.853 | +0.08 |
| hazelnut | 5 | 0.952 | 0.968 | -0.02 |
| toothbrush | 6 | 0.775 | 0.946 | -0.17 |
| metal_nut | 7 | 0.946 | 0.978 | -0.03 |
| screw | 8 | **0.420** | 0.856 | **-0.44** |
| wood | 9 | 0.979 | 0.895 | +0.08 |
| tile | 10 | 1.000 | 0.883 | +0.12 |
| capsule | 11 | 0.670 | 0.939 | -0.27 |
| pill | 12 | 0.819 | 0.951 | -0.13 |
| cable | 13 | 0.875 | 0.910 | -0.03 |
| bottle | 14 | 0.999 | 0.958 | +0.04 |

### 발견된 문제

#### 1. 미세 성능 변화 (context_conv 공유)

grid Image AUC 추적:
```
After Task  1: 0.8463
After Task 14: 0.8421 (-0.42%)
```

**원인**: `context_conv`와 `context_scale_param`이 여전히 모든 task에서 학습됨
```python
# mole_nf.py:706-710 (V4)
if hasattr(subnet, 'context_conv'):
    params.extend(subnet.context_conv.parameters())  # 모든 task에서 학습!
```

#### 2. Image AUC << Pixel AUC 문제

| Class | Image AUC | Pixel AUC | Gap | 원인 |
|-------|-----------|-----------|-----|------|
| screw | 0.42 | 0.86 | -0.44 | 미세 결함, normal도 high score |
| capsule | 0.67 | 0.94 | -0.27 | 형상 유사성 |
| toothbrush | 0.78 | 0.95 | -0.17 | 텍스처 유사 |

**원인 분석:**
- Image Score = max(patch scores) 또는 99th percentile
- Normal 이미지의 일부 패치가 높은 anomaly score를 가짐
- Anomaly/Normal 이미지의 max score 분포가 중첩
- Pixel-level은 개별 패치 단위로 평가되어 분리가 잘 됨

---

## V4.1 - True Complete Separation

### 변경 이유

V4에서 `context_conv`가 여전히 공유되어 미세 성능 저하 발생

### 핵심 변경

| 컴포넌트 | V4 | V4.1 |
|----------|-----|------|
| SpatialMixer | Task 0 이후 freeze | Task 0 이후 freeze |
| **context_conv** | 모든 task 학습 | **Task 0 이후 freeze** |
| **context_scale_param** | 모든 task 학습 | **Task 0 이후 freeze** |

### 구현 변경

#### mole_nf.py - get_fast_params()

```python
# V4.1: MoLEContextSubnet context parameters - only trained in Task 0
if task_id == 0:
    if hasattr(subnet, 'context_conv'):
        params.extend(subnet.context_conv.parameters())
    if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
        params.append(subnet.context_scale_param)
```

#### mole_nf.py - get_trainable_params() (Task > 0 블록)

```python
# V4.1: MoLEContextSubnet context parameters are frozen for Task > 0
# They are only trained in Task 0 (see the task_id == 0 block above)
```

### V4.1 File Changes

| File | Changes |
|------|---------|
| `moleflow/models/mole_nf.py` | `get_fast_params()`, `get_trainable_params()`, `freeze_fast_params()`, `unfreeze_fast_params()` - context_conv도 task_id == 0일 때만 학습 |
| `run.sh` | V4.1 실험 스크립트 추가 |

### 예상 결과

| 지표 | V4 | V4.1 예상 |
|------|-----|-----------|
| Task 성능 변화 | -0.42% | **0%** (완전 고정) |
| 학습 파라미터 | LoRA + DIA + WhiteningAdapter + context_conv | LoRA + DIA + WhiteningAdapter |

---

## V4,3 - Score Aggregation Improvements

### Motivation

V4.1에서 catastrophic forgetting은 해결되었지만, **Image AUC가 Pixel AUC보다 현저히 낮은 문제** 여전히 존재:

| Class | Image AUC | Pixel AUC | Gap |
|-------|-----------|-----------|-----|
| screw | 0.42 | 0.86 | -0.44 |
| capsule | 0.67 | 0.94 | -0.27 |
| toothbrush | 0.78 | 0.95 | -0.17 |
| **Mean** | **0.87** | **0.94** | **-0.07** |

**통계 분석:**
- Image AUC std: 0.1532 (높은 분산)
- Pixel AUC std: 0.0399 (낮은 분산)

### 문제 원인 분석

**현재 Image Score 계산:**
```python
# 기존: 99th percentile
patch_scores = -log_pz - log_det  # (B, H, W)
image_scores = torch.quantile(patch_scores.reshape(B, -1), 0.99, dim=1)
```

**문제:**
1. Normal 이미지에도 outlier 패치 존재 (높은 anomaly score)
2. 99th percentile은 이 outlier에 민감
3. Normal과 Anomaly 이미지의 image score 분포가 중첩

```
Normal Image:  패치 scores = [0.1, 0.2, 0.3, ..., 0.8, 0.9, 1.5(outlier)]
                                                            ↑ 99th percentile = 1.5
Anomaly Image: 패치 scores = [0.1, 0.2, 0.3, ..., 1.2, 1.4, 1.6(true anomaly)]
                                                            ↑ 99th percentile = 1.6
→ 분포 중첩으로 구분 어려움
```

### Solution: Configurable Score Aggregation

**Top-K Averaging** 접근:
```python
# Top-K 평균: outlier 영향 감소
top_k_scores, _ = torch.topk(patch_scores, k=10, dim=1)
image_score = top_k_scores.mean(dim=1)
```

**장점:**
- K개 패치 평균 → 단일 outlier 영향 희석
- Normal의 sporadic outlier와 Anomaly의 clustered anomaly 구분 가능

### Implementation

#### 1. AblationConfig (ablation.py)

새로운 config 옵션 추가:
```python
# V4.3 Score Aggregation
score_aggregation_mode: str = "percentile"  # percentile, top_k, top_k_percent, max, mean
score_aggregation_percentile: float = 0.99  # For percentile mode
score_aggregation_top_k: int = 10           # For top_k mode
score_aggregation_top_k_percent: float = 0.05  # For top_k_percent mode (5%)
```

#### 2. continual_trainer.py - _aggregate_patch_scores()

새로운 aggregation 메서드:
```python
def _aggregate_patch_scores(self, patch_scores: torch.Tensor) -> torch.Tensor:
    """
    Aggregate patch-level scores to image-level score.

    Args:
        patch_scores: (B, H, W) tensor of per-patch anomaly scores

    Returns:
        image_scores: (B,) tensor of per-image anomaly scores
    """
    B = patch_scores.shape[0]
    flat_scores = patch_scores.reshape(B, -1)  # (B, H*W)
    num_patches = flat_scores.shape[1]
    mode = self.score_aggregation_mode

    if mode == "percentile":
        p = self.score_aggregation_percentile
        image_scores = torch.quantile(flat_scores, p, dim=1)
    elif mode == "top_k":
        k = min(self.score_aggregation_top_k, num_patches)
        top_k_scores, _ = torch.topk(flat_scores, k, dim=1)
        image_scores = top_k_scores.mean(dim=1)
    elif mode == "top_k_percent":
        k = max(1, int(num_patches * self.score_aggregation_top_k_percent))
        top_k_scores, _ = torch.topk(flat_scores, k, dim=1)
        image_scores = top_k_scores.mean(dim=1)
    elif mode == "max":
        image_scores = flat_scores.max(dim=1)[0]
    elif mode == "mean":
        image_scores = flat_scores.mean(dim=1)
    else:
        # Fallback to percentile
        image_scores = torch.quantile(flat_scores, 0.99, dim=1)

    return image_scores
```

#### 3. CLI Arguments

```bash
python run_moleflow.py \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 10 \
    --experiment_name V4.3-TopK10
```

### Aggregation Modes 비교

| Mode | 수식 | 특성 | 예상 효과 |
|------|------|------|-----------|
| `percentile` | `quantile(scores, 0.99)` | 기존 방식, outlier 민감 | Baseline |
| `top_k` | `mean(top_k_scores)` | K개 평균, outlier 영향 감소 | **추천** |
| `top_k_percent` | `mean(top_5%_scores)` | 비율 기반, 해상도 무관 | Alternative |
| `max` | `max(scores)` | 가장 극단적, 가장 민감 | 특수 케이스 |
| `mean` | `mean(scores)` | 전체 평균, 가장 둔감 | 특수 케이스 |

### 실험 계획

**Pilot (3 classes: leather, grid, transistor):**
```bash
# GPU 0: Baseline (percentile 99%)
python run_moleflow.py --score_aggregation_mode percentile --score_aggregation_percentile 0.99 \
    --experiment_name Version5-ScoreAgg_percentile99

# GPU 1: Top-K (K=10)
python run_moleflow.py --score_aggregation_mode top_k --score_aggregation_top_k 10 \
    --experiment_name Version5-ScoreAgg_topk10
```

**추가 실험 (선택):**
- Top-K percent (5%)
- Lower percentile (95%)

### V4.3 File Changes Summary

| File | Changes |
|------|---------|
| `moleflow/config/ablation.py` | V5 Score Aggregation config options 추가 (lines 136-151), CLI arguments 추가 (lines 631-652), `parse_ablation_args()` 업데이트 (lines 817-827) |
| `moleflow/trainer/continual_trainer.py` | `_aggregate_patch_scores()` 메서드 추가, `_compute_anomaly_scores()` 수정하여 aggregation 호출 |
| `run.sh` | V4.3 실험 스크립트 추가 |

### 예상 결과

| 지표 | V4.1 (percentile) | V5 (top_k) 예상 |
|------|-------------------|-----------------|
| Image AUC (screw) | 0.42 | 0.55+ |
| Image AUC (capsule) | 0.67 | 0.75+ |
| Mean Image AUC | 0.87 | **0.90+** |
| Image AUC std | 0.1532 | 0.10 미만 |

---

## V4.4 - LayerNorm Ablation Study

### 배경

V4.2/V4.3 실험 후 Image AUC가 Pixel AUC보다 낮은 문제의 원인으로 **LayerNorm이 anomaly 신호를 약화시킨다**는 가설을 세움.

가설의 근거:
- LayerNorm은 patch별 에너지(||x||), 평균(mean), 분산(std)을 제거
- 이 정보들이 anomaly 탐지에 중요할 수 있음
- WhiteningAdapter가 `nn.LayerNorm(channels, elementwise_affine=False)` 사용

### 실험 설계

**공정한 비교를 위해 WhiteningAdapterNoLN 구현:**

```python
class WhiteningAdapterNoLN(nn.Module):
    """WhiteningAdapter WITHOUT LayerNorm"""

    def forward(self, x):
        # LayerNorm 없이 바로 gamma/beta 적용
        # ||x||, mean(x), std(x) 정보 보존
        return self.gamma * x + self.beta
```

비교 대상:
| Adapter | LayerNorm | gamma/beta |
|---------|-----------|------------|
| WhiteningAdapter | ✅ ON | constrained [0.5, 2.0] |
| WhiteningAdapterNoLN | ❌ OFF | constrained [0.5, 2.0] |

### 실험 결과

| 실험 | LayerNorm | Mean Image AUC | Mean Pixel AUC |
|------|-----------|----------------|----------------|
| V4.2-topk3 | ✅ ON | **0.8903** | **0.9357** |
| V4.4-whitening_no_ln | ❌ OFF | 0.8476 | 0.9222 |
| **차이** | | **-4.8%** | **-1.4%** |

클래스별 비교:
| Class | Image AUC (LN) | Image AUC (No LN) | 변화 |
|-------|----------------|-------------------|------|
| leather | 1.0000 | 1.0000 | 0% |
| grid | 0.8956 | 0.8145 | **-9.1%** |
| transistor | 0.7754 | 0.7283 | **-6.1%** |

### 결론

**가설 기각: LayerNorm은 병목이 아님**

1. LayerNorm 제거 시 성능 **하락** (특히 Image AUC -4.8%)
2. LayerNorm이 오히려 학습 안정성에 기여
3. Image AUC 하락이 Pixel AUC보다 큼 → 불안정한 patch score가 aggregation에서 더 큰 영향

### File Changes

| File | Changes |
|------|---------|
| `moleflow/models/adapters.py` | `WhiteningAdapterNoLN` 클래스 추가, `create_task_adapter()`에 `whitening_no_ln` 옵션 추가 |
| `moleflow/config/ablation.py` | CLI choices에 `whitening_no_ln` 추가 |

---

## V4.3 All Classes 분석 - 클래스별 성능 편차

### 15 클래스 전체 실험 결과

**V4.3-topk3_all_classes (기본 순서):**

| Task ID | Class | Image AUC | Pixel AUC | 비고 |
|---------|-------|-----------|-----------|------|
| 0 | leather | 1.0000 | 0.9720 | ✅ 최고 |
| 1 | grid | 0.8956 | 0.9082 | |
| 2 | transistor | 0.7754 | 0.9270 | ⚠️ 낮음 |
| 3 | carpet | 0.9755 | 0.9648 | ✅ 우수 |
| 4 | zipper | 0.9288 | 0.8550 | |
| 5 | hazelnut | 0.9529 | 0.9682 | ✅ 우수 |
| 6 | toothbrush | 0.7861 | 0.9459 | ⚠️ 낮음 |
| 7 | metal_nut | 0.9565 | 0.9776 | ✅ 우수 |
| 8 | **screw** | **0.4575** | 0.8573 | ❌ **매우 낮음** |
| 9 | wood | 0.9798 | 0.8949 | ✅ 우수 |
| 10 | tile | 1.0000 | 0.8843 | ✅ 최고 |
| 11 | capsule | 0.6881 | 0.9388 | ⚠️ 낮음 |
| 12 | pill | 0.8391 | 0.9513 | |
| 13 | cable | 0.8771 | 0.9102 | |
| 14 | bottle | 0.9992 | 0.9571 | ✅ 최고 |
| **Mean** | | **0.8741** | **0.9275** | |

### 성능 분포 분석

**Image AUC 기준 분류:**
- 🟢 **우수 (≥0.95)**: leather, tile, bottle, carpet, wood, metal_nut, hazelnut (7개)
- 🟡 **보통 (0.80~0.95)**: grid, zipper, pill, cable (4개)
- 🟠 **낮음 (0.65~0.80)**: transistor, toothbrush, capsule (3개)
- 🔴 **매우 낮음 (<0.65)**: **screw** (1개)

**통계:**
- Mean Image AUC: 0.8741
- Std: ~0.15 (높은 편차)
- Min: 0.4575 (screw)
- Max: 1.0000 (leather, tile)

### 문제 클래스 분석

#### 1. Screw (Image AUC: 0.4575) - 가장 심각

**특성:**
- 매우 작은 결함 (스크래치, 스레드 손상)
- 결함이 전체 이미지에서 매우 작은 비율 차지
- Normal과 Anomaly의 시각적 차이가 미미

**추정 원인:**
- Top-K(K=3) aggregation으로도 부족
- 작은 결함이 patch score에서 충분히 두드러지지 않음
- Pixel AUC (0.86)는 양호 → 위치는 찾지만 image-level 판단 실패

#### 2. Transistor (Image AUC: 0.7754)

**특성:**
- 다양한 결함 유형 (misplaced, bent, damaged)
- 결함 위치와 형태가 다양

**추정 원인:**
- Task 순서상 초기(Task 2)에 학습되어 Base NF와 함께 최적화
- 하지만 후속 task 학습 시 representation drift 가능성

#### 3. Capsule (Image AUC: 0.6881)

**특성:**
- 반투명한 객체, 내부 결함
- 미묘한 색상/텍스처 변화

**추정 원인:**
- ViT feature가 반투명 객체의 미묘한 차이를 포착하기 어려움
- 결함이 전역적 패턴보다 국소적 변화로 나타남

#### 4. Toothbrush (Image AUC: 0.7861)

**특성:**
- 가는 bristle 구조
- 결함이 매우 작은 영역에 집중

**추정 원인:**
- 고해상도 feature가 필요하지만 ViT patch size(16x16)로 인한 정보 손실

### 핵심 문제 정리

1. **Image AUC << Pixel AUC Gap**
   - Pixel은 잘 찾지만 Image-level 판단 실패
   - Aggregation 방식의 한계

2. **클래스별 편차가 큼**
   - Std ~0.15 (목표: 0.05 이하)
   - 특정 클래스(screw)가 전체 평균을 크게 낮춤

3. **작은 결함 탐지 어려움**
   - screw, capsule, toothbrush 공통점: 작거나 미묘한 결함
   - Top-K aggregation으로도 해결 안 됨

---

## Version 5 - 구조적 문제 해결을 위한 개선

### 근본 문제 분석

#### 1. 학습 목표 vs 평가 목표 불일치 (The Objective Gap)

**현상**:
- NF 학습: 평균적 피팅 (`log p(x)` 최대화) - 모든 패치의 합을 평균
- 평가: 분포의 꼬리(Tail)에 있는 극값(Top-k)으로 결정

**원인** (`continual_trainer.py:226-265`):
```python
# 학습: 모든 패치의 평균
log_px_image = log_px_patch.sum(dim=(1, 2))  # 전체 합
nll_loss = -log_px_image.mean()

# 평가: 극값 기반
image_scores = torch.quantile(flat_scores, 0.99, dim=1)  # percentile
# 또는
top_k_scores, _ = torch.topk(flat_scores, k, dim=1)      # top_k
```

**결과**: 평균적으로 정상 분포는 좋아졌지만 극값에 대한 대응이 없어 Image AUC가 낮음

#### 2. 기하학적 정렬 부재 (Geometric Misalignment) - Screw 문제

**현상**:
- Screw 클래스의 무작위 회전이 복잡한 매니폴드를 형성
- 모델이 결함 대신 회전(SE(2))을 학습하는 데 용량 소진

**원인**:
- 코드 전체에서 회전 불변성/등변성 처리 메커니즘 없음
- ViT feature, SpatialMixer, NF coupling 모두 회전에 민감

#### 3. 논리적 이상 미탐지 (Logical Anomaly) - Transistor 문제

**현상**:
- 부품 누락/오배치 등 텍스처는 정상이지만 전역 구조가 깨진 경우 탐지 실패

**원인** (`adapters.py:673`, `lora.py:249`):
```python
# SpatialContextMixer: 3x3 kernel
kernel_size: int = 3  # 3x3 receptive field

# LightweightMSContext
dilations = (1, 2, 4)  # 최대 9x9 effective RF
```

- 37x37 patches에서 9x9는 전체의 0.6%만 커버 → 전역 문맥 부재

#### 4. Pixel-Image AUC 격차 (Statistical Aggregation Error)

**현상**:
- Pixel AUC는 높으나 Image AUC가 현저히 낮음
- 정상 이미지에서도 outlier patch 발생 → image score 분포 overlap

**원인**:
- 기존 Max/Top-k 방식은 산발적 노이즈에 취약
- 실제 결함이 갖는 위상학적 군집성(Topological Clustering)을 반영하지 못함

#### 5. SpatialMixer 고정 문제

**현상**:
- Task 0에 최적화된 context filter가 이후 task에서 고정

**원인** (`mole_nf.py:719-726`):
```python
# V4 Complete Separation: Spatial mixer only trained in Task 0
if self.spatial_mixer is not None and task_id == 0:
    params.extend(self.spatial_mixer.parameters())
```

---

### Version 5 해결책

#### 문제-해결책 매핑

| 순위 | 문제점 | 해결책 | 난이도 | 기대 효과 |
|:----:|--------|--------|:------:|:---------:|
| **1** | 학습-평가 불일치 | Tail-Aware Loss | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **2** | Image AUC 붕괴 | Spatial Clustering Score | ⭐⭐ | ⭐⭐⭐⭐ |
| **3** | SpatialMixer 고정 | Task-Adaptive Context | ⭐⭐⭐ | ⭐⭐⭐ |
| **4** | Long-range Dependency | Global Context Module | ⭐⭐⭐ | ⭐⭐⭐ |
| **5** | Geometry-Semantic Entanglement | Semantic Projector | ⭐⭐~⭐⭐⭐ | ⭐⭐⭐⭐ |

---

### Solution 1: Tail-Aware Loss (Phase 1)

**핵심**: 학습 시에도 극값을 고려하는 손실 함수

```python
def _compute_tail_aware_loss(self, z, logdet_patch,
                              tail_weight=0.3, top_k_ratio=0.05):
    """
    L = (1 - λ) * L_mean + λ * L_tail
    """
    # Patch-wise NLL
    nll_patch = -(log_pz + logdet_patch)  # (B, H, W)

    # 1. Mean loss (기존)
    nll_mean = nll_patch.mean()

    # 2. Tail loss (상위 k% 패치)
    flat_nll = nll_patch.reshape(B, -1)
    k = max(1, int(flat_nll.shape[1] * top_k_ratio))
    top_k_nll, _ = torch.topk(flat_nll, k, dim=1)
    nll_tail = top_k_nll.mean()

    # Combined
    total_loss = (1 - tail_weight) * nll_mean + tail_weight * nll_tail
    return total_loss
```

**Config 옵션**:
```python
use_tail_aware_loss: bool = True
tail_weight: float = 0.3
tail_top_k_ratio: float = 0.05
```

---

### Solution 2: Spatial Clustering Score (Phase 1)

**핵심**: 산발적 노이즈 vs 실제 결함(cluster) 구분

```python
def _aggregate_with_spatial_clustering(self, patch_scores,
                                        cluster_weight=0.5):
    """실제 결함은 cluster 형성, 노이즈는 산발적"""
    # 1. 기본 top-k score
    top_k_score = torch.topk(flat_scores, k=10, dim=1)[0].mean(dim=1)

    # 2. High-score region의 connectivity 측정
    high_mask = patch_scores > threshold
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    dilated = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)

    cluster_ratio = dilated.sum() / mask.sum()

    # 3. Cluster bonus: 진짜 결함이면 점수 증폭
    image_score = top_k_score * (1 + cluster_weight * cluster_ratio)
    return image_score
```

**Config 옵션**:
```python
score_aggregation_mode: str = "spatial_cluster"
cluster_weight: float = 0.5
```

---

### Solution 3: Task-Adaptive Context (Phase 2)

**핵심**: Frozen base mixer + Task-specific lightweight adapter

```python
class TaskAdaptiveContextMixer(nn.Module):
    """
    Base SpatialMixer (frozen after Task 0) + Task-specific gate/scale/bias
    """
    def __init__(self, channels, base_mixer):
        self.base_mixer = base_mixer
        self.task_gates = nn.ParameterDict()
        self.task_scales = nn.ParameterDict()
        self.task_biases = nn.ParameterDict()

    def forward(self, x, task_id):
        base_out = self.base_mixer(x)
        gate = torch.sigmoid(self.task_gates[str(task_id)])
        scale = self.task_scales[str(task_id)]
        bias = self.task_biases[str(task_id)]

        adapted = scale * base_out + bias
        return (1 - gate) * x + gate * adapted
```

**Config 옵션**:
```python
use_task_adaptive_context: bool = True
```

---

### Solution 4: Global Context Module (Phase 3)

**핵심**: Regional pooling + Cross-attention으로 전역 문맥 추출

```python
class LightweightGlobalContext(nn.Module):
    """O(N * R²) 복잡도로 global context"""
    def __init__(self, channels, num_regions=4, reduction=4):
        self.region_proj = nn.Linear(channels, channels // reduction)
        self.query_proj = nn.Linear(channels, channels // reduction)
        self.out_proj = nn.Linear(channels // reduction, channels)
        self.gate = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x):
        # 1. Regional tokens via pooling
        regions = F.adaptive_avg_pool2d(x_4d, (R, R))

        # 2. Cross-attention
        Q = self.query_proj(x_flat)
        K, V = self.key_proj(regions), self.value_proj(regions)
        attn = softmax(Q @ K.T / sqrt(d))
        global_ctx = attn @ V

        # 3. Gated residual
        return x + sigmoid(self.gate) * global_ctx
```

**Config 옵션**:
```python
use_global_context: bool = True
global_context_regions: int = 4
```

---

### Solution 5: Semantic Projector (Phase 2)

**핵심**: Permutation-invariant pooling으로 positional info 제거, semantic 학습

```python
class SemanticProjector(nn.Module):
    """Position-agnostic semantic feature extraction"""
    def __init__(self, channels, bottleneck_ratio=0.5):
        self.patch_encoder = nn.Sequential(...)  # Per-patch
        self.global_encoder = nn.Sequential(...)  # Set function
        self.global_decoder = nn.Sequential(...)
        self.gate = nn.Parameter(torch.tensor([0.3]))

    def forward(self, x):
        # 1. Per-patch semantic
        x_semantic = self.patch_encoder(x)

        # 2. Global context (permutation-invariant)
        global_feat = self.global_encoder(x).mean(dim=1)  # Position 제거
        global_ctx = self.global_decoder(global_feat)

        # 3. Combine
        return x_semantic + sigmoid(self.gate) * global_ctx
```

**Config 옵션**:
```python
use_semantic_projector: bool = True
semantic_bottleneck_ratio: float = 0.5
```

---

### 구현 로드맵

```
Phase 1 (즉시 적용, 높은 효과)
├── Tail-Aware Loss
└── Spatial Clustering Score

Phase 2 (단기, 구조 수정)
├── Semantic Projector
└── Task-Adaptive Context

Phase 3 (중기)
└── Global Context Module
```

### 기대 효과

| 해결책 | Image AUC | Pixel AUC | 클래스 편차 | Screw | Transistor |
|--------|:---------:|:---------:|:----------:|:-----:|:----------:|
| Tail-Aware Loss | ⬆️⬆️⬆️ | ⬆️ | ⬆️⬆️ | ⬆️ | ⬆️ |
| Spatial Clustering | ⬆️⬆️⬆️ | - | ⬆️⬆️ | ⬆️ | ⬆️ |
| Task-Adaptive Ctx | ⬆️ | ⬆️ | ⬆️⬆️⬆️ | ⬆️ | ⬆️ |
| Global Context | ⬆️ | ⬆️ | ⬆️ | ⬆️ | ⬆️⬆️⬆️ |
| Semantic Projector | ⬆️⬆️ | ⬆️ | ⬆️⬆️ | ⬆️⬆️⬆️ | ⬆️⬆️ |

### File Changes Summary

| File | Changes |
|------|---------|
| `moleflow/config/ablation.py` | V5 config options 추가 |
| `moleflow/trainer/continual_trainer.py` | Tail-Aware Loss, Spatial Clustering Score |
| `moleflow/models/adapters.py` | SemanticProjector, TaskAdaptiveContextMixer, LightweightGlobalContext |
| `moleflow/models/mole_nf.py` | 새 모듈 통합 |
| `run.sh` | V5 실험 스크립트 |

---

## 이전 분석 (참고용)

### V4.3 이전 아키텍처 리뷰

```
ViT Backbone (frozen)
    ↓
Multi-block Feature Aggregation (blocks 8,9,10,11)
    ↓
Positional Embedding (sin/cos)
    ↓
WhiteningAdapter (task-specific, LayerNorm + gamma/beta)
    ↓
SpatialMixer (frozen after Task 0)
    ↓
Normalizing Flow + LoRA (task-specific)
    ↓
DIA (task-specific invertible adapter)
    ↓
Anomaly Score = -log p(z) - log|det J|
    ↓
Aggregation (top-k mean)
    ↓
Image Score
```

### 병목 후보 분석 (V4 기준)

#### A. Feature Extraction Level

| 요소 | 현재 상태 | 잠재적 문제 |
|------|----------|-------------|
| Coupling layers | 8 layers | 표현력 제한 가능 |
| LoRA rank | 64 | Task별 적응력 제한 가능 |
| DIA | 2 blocks | 분포 정렬 표현력 제한 |

#### D. Scoring Level

| 요소 | 현재 상태 | 잠재적 문제 |
|------|----------|-------------|
| Patch score | -log p(z) - log|det J| | 표준 NLL |
| Aggregation | top-k mean (k=3) | **작은 결함에 부족** |
| Calibration | 없음 | **Task별 score scale 불일치** |

### 개선 방향 제안

#### 방향 1: Adaptive Aggregation (클래스 난이도 기반)

**문제**: 고정 K값이 모든 클래스에 적합하지 않음
- Screw: 결함이 매우 작음 → K=1~2 필요
- Carpet: 결함이 넓음 → K=5~10 적합

**제안**:
```python
# 학습된 aggregation weight
class AdaptiveAggregation(nn.Module):
    def forward(self, patch_scores):
        # Attention-based weighted sum
        weights = self.attention(patch_scores)  # 학습
        return (weights * patch_scores).sum()
```

#### 방향 2: Score Calibration (Task별 정규화)

**문제**: Task별 score 분포가 다름
- Task 0 (leather): score range [0, 5]
- Task 8 (screw): score range [0, 20]

**제안**:
```python
# Task별 정상 score 통계 저장
class ScoreCalibrator:
    def __init__(self):
        self.task_stats = {}  # {task_id: (mean, std)}

    def calibrate(self, score, task_id):
        mean, std = self.task_stats[task_id]
        return (score - mean) / std  # Z-score 정규화
```

#### 방향 3: Multi-scale Patch Analysis

**문제**: 16x16 patch가 작은 결함을 놓침

**제안**: 다중 해상도 feature 사용
- 원본 patch (16x16)
- Overlapping patches
- 또는 더 작은 patch size backbone

#### 방향 4: Contrastive/Margin Loss 추가

**문제**: NLL만으로는 Normal/Anomaly 분리력 부족

**제안**:
```python
# Pseudo-anomaly로 margin loss 추가
loss = nll_loss + lambda * margin_loss(normal_scores, pseudo_anomaly_scores)
```

#### 방향 5: DIA 표현력 확대

**문제**: 2 blocks DIA가 분포 차이가 큰 task에 부족

**제안**:
- Task 난이도 기반 blocks 수 조정
- 또는 더 expressive한 flow 구조

### 우선순위 추천

| 순위 | 방향 | 기대 효과 | 구현 난이도 |
|------|------|----------|------------|
| 1 | Score Calibration | 클래스별 편차 완화 | 낮음 |
| 2 | Adaptive Aggregation | screw 등 개선 | 중간 |
| 3 | DIA 표현력 확대 | 전반적 향상 | 낮음 |
| 4 | Contrastive Loss | 분리력 향상 | 중간 |
| 5 | Multi-scale Patch | 작은 결함 탐지 | 높음 |

---
## V5.5 - Position-Agnostic Improvements (2025-12-28)

### Problem Identified

V5 experiments revealed screw class remains at ~0.40-0.44 Image AUC (worse than random).

**Root Cause Analysis**:
- Pixel AUC for screw: ~0.85 (decent - patch detection works)
- Image AUC for screw: ~0.41 (terrible - worse than random)
- **Key Insight**: The problem is NOT in anomaly detection, but in aggregation
- Screw has random orientations → normal rotated patches get high anomaly scores → false positive noise dominates top-k aggregation

**Fundamental Issue**: Position-dependent learning
- NF learns "pattern at position (x,y)" instead of "pattern regardless of position"
- Works for fixed-position objects (leather, grid) but breaks for rotated objects (screw)

### V5.5 Implementation: 3 Class-Agnostic Directions

All directions use V5.1a-TailAwareLoss as baseline and address the position problem without class-specific hacks.

#### Direction 1: Relative Position Encoding (`--use_relative_position`)
**Idea**: Replace absolute PE with relative position attention
- Instead of "what is at (5,5)?", ask "what is the relationship between neighboring patches?"
- Relative patterns (thread spacing) are rotation-invariant
- **Implementation**: `RelativePositionEmbedding` in adapters.py
  - Learnable relative position bias table
  - Query/Key projection for attention
  - Blend gate to combine with absolute PE

#### Direction 2: Dual Branch Scoring (`--use_dual_branch`)
**Idea**: Two parallel NF branches, learn when to trust each
- Position Branch: Standard NF with PE (good for aligned objects)
- No-Position Branch: NF without PE (good for rotated objects)
- Final score = α * pos_score + (1-α) * nopos_score
- α is learned per-patch based on local pattern consistency
- **Implementation**: `DualBranchScorer` in adapters.py
  - Alpha predictor network
  - Dual forward pass in `_compute_anomaly_scores()`

#### Direction 3: Local Consistency Calibration (`--use_local_consistency`)
**Idea**: Down-weight isolated high scores (likely rotation noise)
- Real anomalies have spatially consistent high scores
- False positives (rotation artifacts) are isolated
- **Implementation**: `LocalConsistencyCalibrator` in adapters.py
  - 3x3 consistency convolution
  - Learnable temperature and minimum weight

### Experiment Setup (run.sh)

```bash
# GPU 0: Dir1 - Relative Position
--use_relative_position --relative_position_max_dist 7

# GPU 1: Dir3 - Local Consistency
--use_local_consistency --local_consistency_kernel 3

# GPU 4: Dir1+Dir3 Combined (most promising)
--use_relative_position --use_local_consistency

# GPU 5: Dir2 - Dual Branch
--use_dual_branch
```

### Files Modified

1. **adapters.py**: Added 3 new modules
   - `RelativePositionEmbedding`
   - `DualBranchScorer`
   - `LocalConsistencyCalibrator`

2. **ablation.py**: Added V5.5 config options and CLI args

3. **mole_nf.py**: Integrated V5.5 modules in forward pass

4. **continual_trainer.py**: 
   - Added V5.5 settings
   - Integrated LocalConsistency in `_aggregate_patch_scores()`
   - Implemented dual-branch scoring in `_compute_anomaly_scores()`

---

### V5.5 실험 결과 (2025-12-28)

#### 결과 테이블 (Image AUC)

| Experiment | leather | grid | transistor | screw | Mean |
|------------|---------|------|------------|-------|------|
| Baseline (V4.3) | 1.00 | 0.90 | 0.78 | 0.46 | 0.87 |
| Dir1-RelativePosition | 1.00 | 0.87 | 0.76 | 0.39 | 0.75 |
| Dir2-DualBranch | 0.17 | 0.35 | 0.52 | **0.91** | 0.49 |
| **Dir3-LocalConsistency** | 1.00 | **0.92** | **0.81** | 0.43 | **0.79** |
| Dir1+Dir3-Combined | 1.00 | 0.84 | 0.77 | 0.40 | 0.75 |

#### 핵심 발견

**1. Dir2 (DualBranch)가 가설을 증명함**
```
Screw Image AUC: 0.39 → 0.91 (2.3배 향상!)

그러나 다른 클래스 붕괴:
- leather: 1.00 → 0.17
- grid: 0.90 → 0.35
- transistor: 0.78 → 0.52
```

**해석**:
- 위치 정보 제거가 screw 문제의 해결책임을 확인
- α predictor가 제대로 학습되지 않음
- no-position 브랜치가 과도하게 지배하면서 고정 방향 클래스 성능 붕괴

**2. Dir3 (LocalConsistency)가 가장 균형잡힌 접근법**
```
Grid: 0.90 → 0.92 (개선)
Transistor: 0.78 → 0.81 (개선)
Screw: 0.46 → 0.43 (미미한 변화)
Mean: 0.79 (baseline 0.87보다 낮지만 4클래스 중 가장 높음)
```

**3. Dir1 (RelativePosition)은 효과 없음**
- 상대 위치 인코딩만으로는 rotation invariance 달성 불가
- Screw 오히려 악화: 0.46 → 0.39

**4. Combined (Dir1+Dir3)는 Dir3 단독보다 나쁨**
- Dir1이 오히려 방해 요소로 작용

#### 실패 원인 분석

**Dir2 α predictor 실패 원인**:
```python
# 현재 구조
self.alpha_net = nn.Sequential(
    nn.Linear(channels * 2, channels // 2),
    nn.LayerNorm(channels // 2),
    nn.GELU(),
    nn.Linear(channels // 2, 1),
    nn.Sigmoid()  # α ∈ [0, 1]
)
# 초기화: α ≈ 0.5로 시작

문제:
1. 학습 초기 no-pos 브랜치 loss가 더 낮음 (위치 에러 없으므로)
2. Gradient가 α를 0 방향으로 빠르게 이동
3. 일단 α → 0이 되면 pos 브랜치 gradient 소실
4. 결과: α ≈ 0 고정 (no-pos만 사용)
```

#### 다음 단계 제안

1. **Dir2 개선안**:
   - α 초기값을 0.7-0.8로 설정 (pos 브랜치 선호)
   - α에 regularization 추가: loss += λ * |α - 0.5|
   - Warm-up: 초기 N epochs는 α 고정

2. **Dir3 확장**:
   - 커널 크기 실험: 5x5, 7x7
   - Temperature 조정 실험

3. **새로운 방향**:
   - Task-aware α: 각 task별로 다른 α 학습
   - Rotation augmentation + contrastive learning

---

## V5.6 - Improved Position-Agnostic Solutions (2025-12-28)

### 1. V5.5 실패 원인 분석

#### Dir2 (DualBranchScorer) 실패 분석

**현상**: Screw 0.91 달성했으나 다른 클래스 붕괴 (leather 0.17)

**원인 분석**:
```
학습 초기:
  - pos_score: 위치 정보 기반 → 일부 패치 높은 error
  - nopos_score: 위치 정보 없음 → 전반적으로 낮은 error
  
  → nopos 브랜치의 loss가 더 낮음
  → Gradient가 α를 0 방향으로 업데이트
  → α ≈ 0이 되면 pos 브랜치 gradient 소실
  → 결과: α → 0 고정 (no-position만 사용)

문제점:
  1. 초기값 α=0.5가 불안정
  2. α에 제약 없어 극단값으로 수렴
  3. 한 번 붕괴하면 회복 불가
```

#### Dir3 (LocalConsistency) 한계

**현상**: 가장 좋았으나 screw 개선 미미 (0.43)

**원인**:
- 단일 3x3 커널은 결함 크기 다양성 미반영
- 큰 결함은 5x5나 7x7 커널 필요
- 작은 결함은 3x3이 적합

### 2. V5.6 개선 방안

#### 2.1 ImprovedDualBranchScorer (Anti-Collapse)

**핵심 개선**:
```python
class ImprovedDualBranchScorer(nn.Module):
    def __init__(self, channels, init_alpha=0.7, min_alpha=0.3, max_alpha=0.9):
        # 1. 초기값 0.7 (pos 브랜치 선호로 시작)
        init_logit = log(init_alpha / (1 - init_alpha))
        nn.init.constant_(self.alpha_net[-1].bias, init_logit)
        
        # 2. α clamp로 collapse 방지
        self.min_alpha = min_alpha  # 최소 30% pos 사용
        self.max_alpha = max_alpha  # 최대 90% pos 사용
    
    def forward(self, z_pos, z_nopos, score_pos, score_nopos):
        # 3. Score 차이를 추가 입력으로 활용
        score_diff = (score_pos - score_nopos) / (|score_pos| + |score_nopos| + ε)
        combined_input = cat([z_pos, z_nopos, score_diff], dim=-1)
        
        # 4. α를 [min, max] 범위로 제한
        alpha_raw = sigmoid(self.alpha_net(combined_input))
        alpha = min_alpha + (max_alpha - min_alpha) * alpha_raw
        
        return alpha * score_pos + (1 - alpha) * score_nopos
```

**기대 효과**:
- α가 0으로 붕괴하지 않음 (min=0.3 보장)
- pos 브랜치가 항상 최소 30% 기여
- Score 차이 입력으로 더 informative한 α 예측

#### 2.2 ScoreGuidedDualBranch (Alternative)

**더 단순한 접근**:
```python
class ScoreGuidedDualBranch(nn.Module):
    """
    Latent 대신 score 차이로 직접 α 결정.
    
    아이디어:
    - score_pos < score_nopos → pos 브랜치가 더 좋음 → α ↑
    - score_pos > score_nopos → nopos 브랜치가 더 좋음 → α ↓
    """
    
    def forward(self, z_pos, z_nopos, score_pos, score_nopos):
        score_diff = score_pos - score_nopos
        normalized_diff = score_diff / score_magnitude
        
        # diff > 0 (pos가 worse) → α 감소
        # diff < 0 (pos가 better) → α 증가
        alpha = sigmoid(temp * (bias - normalized_diff))
        alpha = clamp(alpha, min=min_alpha, max=1-min_alpha)
        
        return alpha * score_pos + (1 - alpha) * score_nopos
```

**장점**:
- Latent 기반보다 직접적
- Gradient가 score로 직접 전파
- 더 interpretable

#### 2.3 MultiScaleLocalConsistency

**Multi-scale 분석**:
```python
class MultiScaleLocalConsistency(nn.Module):
    def __init__(self, kernel_sizes=[3, 5, 7], temperature=1.0):
        # 각 스케일별 learnable parameters
        self.temperatures = [Parameter for each kernel]
        self.min_weights = [Parameter for each kernel]
        self.scale_weights = Parameter([1/3, 1/3, 1/3])  # 융합 가중치
        
        # Score-adaptive fusion
        self.adaptive_net = Linear(n_scales, n_scales) + Softmax
    
    def forward(self, patch_scores):
        # 각 스케일에서 consistency 계산
        weights_3x3 = compute_consistency(scores, kernel=3)
        weights_5x5 = compute_consistency(scores, kernel=5)
        weights_7x7 = compute_consistency(scores, kernel=7)
        
        # Adaptive fusion (스케일별 중요도 학습)
        scale_means = stack([w.mean() for w in weights]).T
        fusion_weights = adaptive_net(scale_means)  # Softmax
        
        combined = sum(w * fw for w, fw in zip(all_weights, fusion_weights))
        return patch_scores * combined
```

**기대 효과**:
- 작은 결함 (3x3) + 중간 (5x5) + 큰 결함 (7x7) 모두 커버
- Learnable fusion으로 최적 조합 학습
- V5.5 Dir3 대비 다양한 결함 크기 대응

### 3. 실험 구성 (run.sh)

| GPU | 실험 | 핵심 설정 |
|-----|------|----------|
| 0 | ImprovedDualBranch | init=0.7, α∈[0.3, 0.9] |
| 1 | ScoreGuidedDual | temp=1.0, min_α=0.2 |
| 4 | MultiScaleConsistency | kernels=[3,5,7] |
| 5 | Combined | ImprovedDual + MultiScale |

### 4. 수정된 파일

1. **adapters.py**:
   - `ImprovedDualBranchScorer`: Anti-collapse dual branch
   - `ScoreGuidedDualBranch`: Score-guided alternative
   - `MultiScaleLocalConsistency`: Multi-scale consistency

2. **ablation.py**:
   - V5.6 config options 추가
   - CLI arguments 추가

3. **mole_nf.py**:
   - V5.6 모듈 imports
   - V5.6 settings 처리
   - 모듈 인스턴스 생성
   - get_trainable_params에 V5.6 파라미터 추가

4. **continual_trainer.py**:
   - V5.6 settings 처리
   - _compute_anomaly_scores에서 V5.6 dual branch 처리
   - _aggregate_patch_scores에서 multiscale consistency 처리

5. **run.sh**:
   - V5.6 실험 4개 구성

---

### V5.6 실험 결과 (2025-12-28)

#### 결과 테이블 (Image AUC)

| Experiment | leather | grid | transistor | screw | Mean |
|------------|---------|------|------------|-------|------|
| Baseline (V5.5-Dir3) | 1.00 | 0.92 | 0.81 | 0.43 | **0.79** |
| V5.6-ImprovedDualBranch | 0.47 | 0.40 | 0.60 | **0.90** | 0.59 |
| V5.6-ScoreGuidedDual | 0.14 | 0.49 | 0.57 | **0.90** | 0.52 |
| **V5.6-MultiScaleConsistency** | **1.00** | **0.92** | **0.81** | 0.39 | **0.78** |
| V5.6-Combined | 0.12 | 0.44 | 0.54 | **0.90** | 0.50 |

#### 분석

**1. Dual Branch 개선 실패**
- α clamping [0.3, 0.9]도 collapse 방지 실패
- Screw는 0.90으로 좋지만 다른 클래스 붕괴
- **근본 원인**: 정상 데이터만으로는 pos/nopos 구분 학습 불가

**2. MultiScaleConsistency 유지**
- V5.5-Dir3와 거의 동일 (0.78 vs 0.79)
- Multi-scale이 single-scale 대비 큰 개선 없음
- Screw는 여전히 0.39

**3. Dual Branch 실패 근본 원인**
```
학습 데이터: 정상 이미지만 사용
  ↓
pos_score ≈ nopos_score (정상은 둘 다 낮음)
  ↓
α 학습에 유의미한 신호 없음
  ↓
α가 임의 방향으로 수렴
  ↓
테스트 시 의미있는 선택 불가
```

#### 결론

- **Dual Branch 접근법 포기**: 정상 데이터만으로는 pos/nopos 선택 학습 불가
- **MultiScaleConsistency**: V5.5-Dir3와 동등, 추가 개선 없음
- **새로운 방향 필요**:
  1. Task-level α (각 task마다 고정 α 학습)
  2. Pseudo-anomaly 기반 contrastive learning
  3. Position encoding 자체를 rotation-invariant하게 설계

---

## V5.7 - Rotation-Invariant Position Encoding

### V5.7-DirC-MultiOrientation 결과 (All Classes)

| Task ID | Class | Routing Acc | Image AUC | Pixel AUC |
|---------|-------|-------------|-----------|-----------|
| 0 | bottle | 100.00 | 1.0000 | 0.9469 |
| 1 | cable | 100.00 | 0.9162 | 0.9042 |
| 2 | capsule | 100.00 | 0.7276 | 0.9203 |
| 3 | carpet | 100.00 | 0.9755 | 0.9643 |
| 4 | grid | 100.00 | 0.8989 | 0.8937 |
| 5 | hazelnut | 100.00 | 0.9625 | 0.9646 |
| 6 | leather | 100.00 | 1.0000 | 0.9699 |
| 7 | metal_nut | 100.00 | 0.9717 | 0.9654 |
| 8 | pill | 98.80 | 0.8568 | 0.9438 |
| 9 | screw | 100.00 | **0.3484** | 0.8168 |
| 10 | tile | 100.00 | 1.0000 | 0.8794 |
| 11 | toothbrush | 97.62 | 0.8417 | 0.9414 |
| 12 | transistor | 100.00 | 0.7967 | 0.9456 |
| 13 | wood | 100.00 | 0.9553 | 0.8811 |
| 14 | zipper | 100.00 | 0.9278 | 0.8629 |
| **Mean** | Overall | 99.76 | **0.8786** | 0.9200 |

### V5.7 분석

**Multi-Orientation Ensemble 효과 없음**:
- Mean 0.88로 좋아 보이지만 **Screw 0.35로 여전히 문제**
- Feature 회전 ≠ 의미있는 다른 시점
- Position Embedding이 이미 feature에 baked-in 되어 있음
- 4배 inference cost만 발생, 개선 없음

**ContentBasedPE/HybridPE**:
- Pilot 실험에서 0.75 mean으로 baseline보다 나쁨
- 학습 없이 inference-time에 prototype 매칭은 불안정

---

## V5.8 - TAPE (Task-Adaptive Position Encoding) 구현

### 핵심 아이디어

이전 접근법의 실패 원인 분석:
```
V5.5/V5.6 Dual Branch:
  - Patch-level α decision
  - 정상 데이터: pos_score ≈ nopos_score → No gradient signal
  - α가 collapse하거나 랜덤하게 수렴

V5.7 Multi-Orientation:
  - Inference-time rotation
  - 학습에 반영 안됨 → Cannot learn
  - Feature에 PE가 이미 적용되어 있어 rotation 무의미
```

**TAPE 해결책**:
```
Task-level PE strength + Training-time learning
  ↓
NLL loss provides direct gradient
  ↓
각 task가 최적의 PE strength 자동 학습
```

### 설계

```python
class TaskAdaptivePositionEncoding(nn.Module):
    """
    V5.8: TAPE - Task별 PE 강도 학습

    - pe_gates: {task_id: learnable gate}
    - alpha = sigmoid(gate) → PE strength (0~1)
    - features_with_pe = raw_features + alpha * grid_pe
    """
    def __init__(self, init_value: float = 0.0):
        self.init_value = init_value
        self.pe_gates = nn.ParameterDict()

    def add_task(self, task_id: int):
        self.pe_gates[str(task_id)] = nn.Parameter(
            torch.tensor([self.init_value])
        )

    def forward(self, features, grid_pe, task_id):
        gate = self.pe_gates[str(task_id)]
        alpha = torch.sigmoid(gate)  # 0~1
        return features + alpha * grid_pe
```

### 기대 효과

| Task | 예상 PE Strength | 이유 |
|------|------------------|------|
| Screw | ~0.1-0.3 (낮음) | 회전 불변 → PE 약하게 |
| Leather | ~0.8-1.0 (높음) | 공간 일관성 중요 → PE 강하게 |
| Grid | ~0.5-0.7 (중간) | 어느 정도 위치 정보 필요 |

### 수정된 파일

1. **adapters.py**: `TaskAdaptivePositionEncoding` 클래스 추가
2. **ablation.py**: `use_tape`, `tape_init_value` config 추가
3. **mole_nf.py**: TAPE 통합 (초기화, add_task, forward)
4. **continual_trainer.py**:
   - TAPE 활성화 시 raw features 전달 (PE는 NF 내부에서 적용)
   - Task 훈련 후 PE strength 로깅

### 실행 방법

```bash
# TAPE 기본 실험
python run_moleflow.py --run_diagnostics \
    --use_tape \
    --tape_init_value 0.0 \
    --experiment_name Version5.8-TAPE

# TAPE + LocalConsistency
python run_moleflow.py --run_diagnostics \
    --use_tape \
    --tape_init_value 0.0 \
    --use_local_consistency \
    --local_consistency_kernel 3 \
    --experiment_name Version5.8-TAPE-LocalConsistency
```

### 핵심 차별점 (vs 이전 접근법)

| 측면 | V5.5/V5.6 | V5.7 | V5.8 TAPE |
|------|-----------|------|-----------|
| Decision Level | Patch | Image | **Task** |
| Learning | Training | Inference | **Training** |
| Gradient | None (normal≈normal) | None | **Clear (NLL)** |
| 복잡도 | Moderate | High (4x inference) | **Low** |

---

### V5.8-TAPE v1 실험 결과 (Pilot)

| Task | Class | Image AUC | PE Strength |
|------|-------|-----------|-------------|
| 0 | leather | 1.0000 | 0.5028 |
| 1 | grid | 0.9365 | 0.5000 |
| 2 | transistor | 0.8087 | 0.5000 |
| 3 | screw | 0.3900 | 0.5000 |
| **Mean** | | **0.7838** | |

### 문제 발견: PE Strength가 학습되지 않음

**증상**: 모든 Task의 PE strength가 초기값 0.5에서 거의 변하지 않음

**원인 분석**:
1. TAPE gate는 **단일 스칼라** 파라미터
2. 다른 수천 개 파라미터와 **동일한 learning rate** 사용
3. NLL loss에서 PE 기여도가 다른 파라미터에 비해 작음
4. Gradient가 너무 작아서 학습이 일어나지 않음

### V5.8-TAPE v2: LR Multiplier 추가

**해결책**: TAPE gate에 별도의 높은 learning rate 적용

**수정 내용**:

1. **ablation.py**:
   - `tape_lr_multiplier: float = 100.0` 추가
   - CLI argument `--tape_lr_multiplier` 추가

2. **continual_trainer.py**:
   - `_train_base_task`: Parameter groups로 분리, TAPE에 100x LR
   - `_train_fast_stage`: 동일하게 적용
   - Warmup도 각 그룹별로 적절히 처리

```python
# 수정된 optimizer 생성 코드
if self.use_tape:
    tape_params = self.nf_model.tape.get_trainable_params(task_id)
    other_params = [p for p in trainable_params if id(p) not in tape_param_ids]

    param_groups = [
        {'params': other_params, 'lr': lr},
        {'params': tape_params, 'lr': lr * self.tape_lr_multiplier}  # 100x
    ]
    optimizer = create_optimizer(param_groups, lr=lr)
```

**기대 효과**:
- 기본 LR이 1e-4면, TAPE gate는 1e-2로 학습
- PE strength가 실제로 각 task 특성에 맞게 변화할 것
- Screw: 0.5 → ~0.2 (PE 감소), Leather: 0.5 → ~0.8 (PE 유지/증가)

### V5.8-TAPE v2 실험 결과

| Task | Class | PE Strength | Image AUC | vs v1 |
|------|-------|-------------|-----------|-------|
| 0 | leather | 0.3257 | 1.0000 | = |
| 1 | grid | 0.9748 | 0.9165 | ↓0.02 |
| 2 | transistor | 0.9426 | 0.7963 | ↓0.01 |
| 3 | screw | 0.3082 | 0.3753 | ↓0.01 |
| **Mean** | | | **0.7720** | ↓0.01 |

### 분석: TAPE 학습 방향 문제

**발견 1**: PE strength가 이제 학습됨 (v1의 0.5에서 변화)
- Screw: 0.31 (낮음) ← 의도대로!
- Leather: 0.33 (낮음) ← **반대로 학습됨!**
- Grid/Transistor: 0.94-0.97 (높음)

**발견 2**: 성능은 오히려 저하됨
- v1 Mean: 0.7838 → v2 Mean: 0.7720 (↓)
- 모든 metric에서 소폭 하락

**근본 원인: NLL Loss ≠ Anomaly Detection**

```
NLL Loss 최소화 방향:
  - 낮은 PE → 더 자유로운 fit → 더 낮은 NLL
  - 모델은 PE를 낮추는 방향으로 학습

Anomaly Detection 최적화 방향:
  - 클래스 특성에 맞는 PE 필요
  - Leather: 높은 PE (공간 구조 중요)
  - Screw: 낮은 PE (회전 불변 필요)

→ 두 목표가 정렬되지 않음!
```

**Leather가 낮은 PE로 학습된 이유**:
- Leather의 정상 이미지는 texture가 균일
- PE 없이도 쉽게 fit 가능 → 낮은 NLL
- 하지만 anomaly detection에서는 위치 정보가 필요할 수 있음

### 결론: TAPE 접근법의 한계

**Normal-only training의 근본적 한계**:
1. V5.5/V5.6: Patch-level α → No gradient (pos≈nopos for normal)
2. V5.8 TAPE: Task-level α → Gradient exists but **wrong direction**

NLL loss만으로는 anomaly detection에 최적인 PE strength를 학습할 수 없음.

**가능한 대안**:
1. **Prior knowledge 주입**: 클래스 타입별 PE 고정 (texture→high, object→low)
2. **Pseudo-anomaly 사용**: 가짜 anomaly로 contrastive learning
3. **Validation-based tuning**: Anomaly detection 성능으로 PE 튜닝

---

## Version 5 최종 정리

### Version 5 실험 요약

| Version | 접근법 | 결과 | 문제점 |
|---------|--------|------|--------|
| V5.1a | Tail-Aware Loss + Top-K | **Best baseline** | Screw 여전히 낮음 |
| V5.5 | Dual Branch (pos/nopos) | 실패 | No gradient signal |
| V5.6 | Improved Dual Branch | 실패 | Collapse to one branch |
| V5.7 | Multi-Orientation Ensemble | 개선 없음 | Feature rotation ≠ viewpoint |
| V5.8 | TAPE (Task-Adaptive PE) | 역효과 | NLL ≠ AD performance |

### 핵심 교훈: Normal-Only Training의 한계

**Position Encoding 최적화 시도 실패 원인**:

```
문제 정의:
  - Screw: 회전에 불변해야 함 → PE 약하게
  - Leather: 공간 구조 중요 → PE 강하게

시도한 접근법들:
  1. Patch-level α (V5.5/V5.6)
     - 정상 데이터: pos_score ≈ nopos_score
     - → α에 gradient 신호 없음
     - → 학습 불가

  2. Inference-time 조정 (V5.7)
     - Feature에 PE가 이미 baked-in
     - → rotation 무의미
     - → 개선 없음

  3. Task-level 학습 (V5.8)
     - NLL loss는 "정상 fit" 최적화
     - → PE 낮추는 방향으로 학습 (더 자유로운 fit)
     - → Anomaly detection과 역방향
```

**근본적 한계**:
- Anomaly detection에 최적인 PE를 찾으려면 **anomaly 정보 필요**
- Normal-only training으로는 불가능

### Best Configuration (Version 5 Final)

```bash
python run_moleflow.py \
    --use_whitening_adapter \
    --use_dia \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --use_tail_aware_loss \
    --tail_weight 0.3 \
    --experiment_name Version5-Final
```

### 남은 과제

1. **Screw 클래스 성능 개선**: PE 외 다른 접근 필요
2. **Pseudo-anomaly training**: CutPaste 등으로 anomaly 신호 제공
3. **Class-specific 처리**: Object vs Texture 클래스 구분

---

## Screw 클래스 근본 원인 재분석 (2025-12-29)

### Diagnostics 데이터 분석

**Version5-RotationAug (screw as task 3) 분석**:

| Metric | Normal | Anomaly | 해석 |
|--------|--------|---------|------|
| logdet_std | 72.28 | 57.18 | **Anomaly가 더 uniform** |
| log_pz var | 75,495 | 6,764 | **Normal이 11x 더 diverse** |
| ||z|| vs logdet corr | 0.27 | 0.26 | 유사 |
| ratio (anom/norm) | - | 0.79 | **"DEAD" scale 진단** |

### 핵심 발견: Normal > Anomaly Variance

**Screw의 특이점**: Normal 이미지들이 Anomaly 이미지들보다 **더 높은 분산**을 가짐

이것은 NF 기반 Anomaly Detection에서 치명적인 문제:
1. NF는 "Normal 분포를 학습하고 그 분포에서 벗어난 것을 Anomaly로 탐지"
2. Normal의 분산이 높으면 → 넓은 분포 학습 → Anomaly도 그 안에 포함
3. Anomaly가 더 일관적이면 → 오히려 "더 normal"하게 보임

**MVTec Screw 데이터셋 특성**:
- Train/Test Normal: 다양한 각도/조명/위치에서 촬영
- Test Anomaly: 특정 결함 유형 (scratch_head, thread_side 등)이 더 일관적인 조건에서 촬영

```
Normal 이미지 다양성:
- 320장 학습 이미지
- 다양한 회전 각도
- 다양한 조명 조건
- log_pz variance = 75,495 (매우 큼)

Anomaly 이미지 일관성:
- 결함 유형별로 군집화된 촬영
- 더 통제된 환경
- log_pz variance = 6,764 (상대적으로 작음)
```

### Screw vs 다른 클래스 비교

**Leather, Grid (잘 작동하는 클래스)**:
- Texture 클래스 → 위치 불변 패턴
- Normal/Anomaly 모두 일관적
- Anomaly가 확실한 분포 이탈

**Screw (작동 안하는 클래스)**:
- Object 클래스 + 회전
- Normal 자체가 매우 다양 (회전)
- Anomaly가 오히려 일관적

### 왜 이전 접근법들이 실패했는가

| 접근법 | 실패 이유 |
|--------|-----------|
| V5.5/V5.6 Dual Branch | Normal만으로는 pos/nopos 구분 학습 불가 |
| V5.7 Multi-Orientation | Feature에 이미 PE baked-in |
| V5.8 TAPE | NLL ≠ AD, PE 낮추는 방향으로만 학습 |
| V6 Rotation Aug | PE 충돌 + Normal이 이미 다양해서 효과 없음 |
| V6 No PE | PE가 오히려 도움 주고 있었음 (0.39→0.31) |

### ~~가설: Screw 문제는 "데이터 특성" 문제~~ (수정됨)

**이 가설은 틀렸음** - 아래 "Screw 문제 재분석: V5 컴포넌트가 원인" 섹션 참조

baseline_v8.1 (단순 구조, img_size 518)에서 screw **0.67** 달성!
→ 데이터 문제가 아닌 **V5 컴포넌트 (WhiteningAdapter, SpatialMixer, DIA)가 원인**

### 다음 단계 제안

1. **데이터 분석 심화**:
   - 실제 screw 이미지들을 시각적으로 분석
   - Normal/Anomaly 간 feature 분포 시각화

2. **접근법 전환 고려**:
   - NF 대신 Reconstruction-based 방법 (AutoEncoder 등)
   - 또는 Contrastive Learning으로 anomaly 신호 직접 학습

3. **Screw 전용 처리**:
   - Task-specific preprocessing
   - Rotation alignment (테스트 시 canonical orientation 정렬)

---

## V6 - Rotation Augmentation

### 아이디어

이전 접근법 (V5.5-V5.8)이 실패한 이유:
- Normal-only training에서 PE 최적화 방향을 학습 불가
- Model-level 변경보다 **Data-level** 접근이 더 효과적일 수 있음

**V6 접근법**: Random rotation augmentation
- 학습 시 이미지를 ±180° 랜덤 회전
- 모델이 자연스럽게 rotation-invariant 특성 학습
- Position Encoding은 유지 (회전된 이미지에 PE 적용)

### 구현

**수정된 파일**:

1. **moleflow/data/mvtec.py**:
   - `use_rotation_aug`, `rotation_degrees` 파라미터 추가
   - Training transform에 `T.RandomRotation` 추가

2. **moleflow/data/datasets.py**:
   - `create_task_dataset`에 rotation 설정 전달

3. **moleflow/config/ablation.py**:
   - `use_rotation_aug: bool = False`
   - `rotation_degrees: float = 180.0`
   - CLI arguments 추가

4. **run_moleflow.py**:
   - `create_task_dataset` 호출 시 rotation 설정 전달

### 사용법

```bash
# Rotation augmentation 활성화 (±180°)
python run_moleflow.py \
    --use_rotation_aug \
    --rotation_degrees 180.0 \
    --experiment_name Version6-RotationAug

# 다른 회전 범위 (±90°)
python run_moleflow.py \
    --use_rotation_aug \
    --rotation_degrees 90.0 \
    --experiment_name Version6-RotationAug-90
```

### 기대 효과

| 클래스 | 예상 | 이유 |
|--------|------|------|
| Screw | 개선 | 회전된 정상 이미지로 학습 → 회전에 불변 |
| Leather | 유지/소폭 하락 | Texture는 회전에 원래 불변 |
| Grid | 유지 | 주기적 패턴 |
| Transistor | ? | Component 위치에 따라 다를 수 있음 |

### V6 실험 결과

| Class | Baseline | RotationAug | 변화 |
|-------|----------|-------------|------|
| leather | 1.0000 | 1.0000 | = |
| grid | 0.9365 | 0.8797 | ↓0.06 |
| transistor | 0.8087 | 0.6558 | ↓0.15 |
| screw | 0.3900 | 0.3898 | ≈ |
| **Mean** | 0.7838 | 0.7313 | ↓0.05 |

### 분석: Rotation Augmentation 실패

**결과**: Screw 개선 없음, Grid/Transistor 오히려 성능 저하

**원인: Position Encoding과의 충돌**

```
Rotation Augmentation + Fixed PE = 모순

이미지: 90° 회전됨
  - 원래 (0,0)에 있던 패치 → (0,13)으로 이동

PE: 고정 그리드
  - (0,13) 위치에 (0,13)의 PE 적용

문제:
  - 같은 패치가 다른 위치에서 다른 PE를 받음
  - 모델: "이 패치는 (0,0)에서 본 적 있는데 왜 (0,13) PE가 붙어있지?"
  - → 학습 혼란 → 성능 저하
```

**Screw가 개선되지 않은 이유**:
- Rotation augmentation이 "rotation invariance"를 주지 않음
- PE 불일치로 인해 오히려 학습이 방해됨
- 근본적으로 screw 문제는 rotation이 아닌 다른 원인일 수 있음

### 결론

**Rotation augmentation은 PE와 함께 사용 시 역효과**

가능한 방향:
1. **Rotation Aug + No PE**: PE 비활성화하고 rotation만 사용
2. **Rotation Aug + Rotated PE**: PE도 함께 회전 (구현 복잡)
3. **Rotation 포기**: 다른 접근법 탐색

---

## Screw 문제 재분석: V5 컴포넌트가 원인 (2025-12-29)

### 중요 발견: baseline_v8.1이 screw에서 0.67 달성!

기존 모든 V5/V6 실험에서 screw는 0.44-0.47 수준이었는데,
**baseline_v8.1_lora_rank64_all_classes**에서 **0.6741** 달성!

### 설정 비교

| Component | baseline_v8.1 (screw 0.67) | V5 (screw 0.44) |
|-----------|----------------------------|-----------------|
| img_size | **518** | 224 |
| WhiteningAdapter | ❌ 없음 | ✅ 사용 |
| SpatialContextMixer | ❌ 없음 | ✅ 사용 |
| DIA | ❌ 없음 | ✅ 사용 |
| scale_context | ❌ 없음 | ✅ 사용 |

### 결론: V5 "개선" 컴포넌트들이 screw 성능 저하의 원인

**1. WhiteningAdapter (LayerNorm)**
- LayerNorm이 patch 간 상대적 크기 정보를 정규화
- 작은 결함의 미세한 차이가 희석됨
- Screw의 미세 결함에 치명적

**2. SpatialContextMixer (3x3 context)**
- 3x3 영역 평균/집계
- 작은 결함이 주변과 섞여 blur됨
- Screw의 scratch_head, thread_side 같은 작은 결함 탐지 실패

**3. DIA (Deep Invertible Adapter)**
- Task별 분포 정렬 목적
- Screw의 다양한 정상 분포를 오히려 왜곡할 수 있음

**4. Image Size 224 vs 518**
- 518 해상도에서 더 많은 세부 정보 보존
- Screw 나사산 패턴이 224에서 손실

### 해결 방안

**Option 1: Screw-specific config**
```bash
# Screw 학습 시 V5 컴포넌트 비활성화
python run_moleflow.py \
    --task_classes screw \
    --no_whitening_adapter \
    --no_spatial_context \
    --no_dia \
    --img_size 518
```

**Option 2: 전체 구조 단순화**
- V5 컴포넌트들의 효과 재검증 필요
- 다른 클래스에서도 실제로 도움이 되는지 확인
- 불필요한 복잡성 제거

**Option 3: Adaptive Components**
- 클래스 특성(texture vs object, large vs small defect)에 따라 컴포넌트 활성화
- 학습된 gate로 컴포넌트 사용 여부 결정

### 실험 계획

1. **baseline_v8.1 스타일로 screw 재실험** (img_size 518, no extra components)
2. **각 컴포넌트별 ablation** (WhiteningAdapter만, SpatialMixer만, DIA만 테스트)
3. **img_size 효과 분리 테스트** (518 vs 224, 동일 컴포넌트)

---

## Version 6.1 - Spatial Transformer Network (STN) 실험 결과 (2025-12-29)

### 실험 목적
- Screw 클래스의 rotation 문제를 해결하기 위해 STN 도입
- 이미지 레벨에서 자동 정렬 → PE와 일관성 유지

### 설정
```bash
CUDA_VISIBLE_DEVICES=0 python run_moleflow.py --run_diagnostics \
    --task_classes leather grid transistor screw \
    --use_whitening_adapter --use_dia \
    --score_aggregation_mode top_k --score_aggregation_top_k 3 \
    --use_tail_aware_loss --tail_weight 0.3 \
    --use_stn --stn_mode rotation \
    --stn_hidden_dim 128 --stn_rotation_reg_weight 0.01 \
    --experiment_name Version6.1-STN
```

### 결과: STN 실패 ❌

| Class | V5 Baseline | V6.1-STN | 차이 |
|-------|-------------|----------|------|
| leather | **1.000** | 1.000 | 0.00 |
| grid | **0.942** | 0.923 | **-1.9%** |
| transistor | **0.824** | 0.795 | **-2.9%** |
| screw | **0.443** | 0.416 | **-2.7%** |
| **Average** | **0.802** | 0.784 | **-1.8%** |

**결론: STN이 성능을 오히려 저하시킴**

### 실패 원인 분석

1. **Normal-only Training의 한계**
   - STN이 정상 이미지만으로 학습됨
   - "Canonical orientation"이 무엇인지 명확한 supervision 없음
   - Anomaly detection loss가 STN에 유용한 gradient 제공하지 못함

2. **End-to-end 학습 문제**
   - Rotation이 anomaly score에 직접적 영향 미미
   - NLL loss 최소화와 rotation alignment가 직접 연결되지 않음

3. **Identity 초기화 + Regularization 역효과**
   - rotation_reg_weight=0.01로 변환 최소화 유도
   - 결과적으로 거의 변환이 일어나지 않았을 가능성
   - 그러나 STN 연산 자체가 feature에 노이즈 추가

4. **추가 파라미터의 부작용**
   - Localization network가 이미지에 불필요한 변형 추가
   - Feature extractor 입력이 오염됨

### 시사점

- **이미지 레벨 변환은 근본적 해결책이 아님**
- **Screw 문제의 근본 원인은 rotation이 아닐 수 있음**
- 이전 분석에서 발견한 것처럼 **V5 컴포넌트들(WhiteningAdapter, SpatialMixer, DIA)이 진짜 원인**
- baseline_v8.1(단순 구조)이 screw 0.67 달성한 것이 증거

### 다음 방향

1. **V5 컴포넌트 제거 실험**: WhiteningAdapter, SpatialMixer, DIA 없이 학습
2. **img_size 518로 변경**: 더 높은 해상도에서 세부 정보 보존
3. **단순한 baseline으로 회귀**: 복잡한 컴포넌트가 오히려 해로울 수 있음

---

## Hyperparameter Tuning 실험 결과 (2025-12-30)

### 실험 목적
V5-Final baseline을 기준으로 screw 성능 개선을 위한 hyperparameter 탐색

### 실험 구성 (24개 실험, GPU 0/1/4/5 병렬)

| Round | 변경 요소 |
|-------|----------|
| 1 | V5 Components 제거 (NoWhitening, NoDIA, Simple, Baseline) |
| 2 | Score Aggregation (TopK 1/5/10, Mean) |
| 3 | Tail-aware Loss (NoTail, 0.1, 0.5, 0.7) |
| 4 | Model Capacity (Coupling 12/16, LoRA 32/128) |
| 5 | Learning Rate & Epochs (LR 5e-5/2e-4, Epochs 60/80) |
| 6 | Combined (Simple + 조합) |

### 결과: Screw AUC Top 5

| Rank | Experiment | Screw AUC | Avg AUC | 비고 |
|------|------------|-----------|---------|------|
| 1 | **HP-NoDIA** | **0.508** | 0.699 | Grid/Transistor 망가짐 |
| 2 | **HP-NoTail** | **0.504** | 0.784 | **균형 좋음 ✓** |
| 3 | HP-Epochs80 | 0.482 | 0.810 | 최고 평균 |
| 4 | HP-LR2e-4 | 0.477 | 0.806 | |
| 5 | HP-TopK1 | 0.475 | 0.791 | |
| - | V5-Final (기준) | 0.443 | 0.802 | |

### 클래스별 상세 비교

| Experiment | Leather | Grid | Transistor | Screw | Avg |
|------------|---------|------|------------|-------|-----|
| V5-Final | 1.00 | **0.94** | **0.82** | 0.44 | 0.80 |
| HP-NoTail | 1.00 | 0.88 | 0.75 | **0.50** | 0.78 |
| HP-NoDIA | 1.00 | 0.75 | 0.55 | **0.51** | 0.70 |
| HP-Epochs80 | 1.00 | 0.91 | 0.80 | 0.48 | **0.81** |

### 핵심 발견

1. **DIA 제거** → Screw ↑6% but 다른 클래스 크게 하락
2. **Tail-aware loss 제거** → Screw ↑6%, 균형 유지 ✓
3. **Mean aggregation** → 완전 실패 (Screw 0.04-0.07)
4. **TopK 증가 (5, 10)** → Screw 하락
5. **Epochs 80** → 전체 성능 향상

### 분석

**Tail-aware loss가 screw에 해로운 이유:**
- Tail loss는 상위 5% high-loss patch에 집중
- Screw는 정상 이미지도 variation이 큼 (rotation, position)
- High-loss patch가 반드시 anomaly가 아님 → 잘못된 신호로 학습
- 결과적으로 정상/비정상 구분 능력 저하

**DIA가 screw에 해로운 이유:**
- DIA는 task별 nonlinear adaptation 제공
- 다른 클래스에서는 도움이 되지만
- Screw의 높은 intra-class variance에서는 overfitting 유발
- 정상 분포를 너무 tight하게 학습 → 정상도 anomaly로 판정

### 추천 설정

**Best Trade-off: HP-NoTail**
```bash
--use_whitening_adapter --use_dia \
--score_aggregation_mode top_k --score_aggregation_top_k 3
# tail-aware loss 제거 (--use_tail_aware_loss 없음)
```

- Screw: 0.443 → **0.504** (+13.8% 상대 개선)
- Average: 0.802 → 0.784 (-2.2%)
- 다른 클래스 성능은 약간 하락하지만 screw 개선 효과가 더 큼

---

## V6 Ablation Experiments - Architecture Fundamentals

### 배경

HP 튜닝 결과 분석 후, 아키텍처 근본적인 변경을 통한 ablation 실험 진행.

### 실험 설계

| Exp | Name | 설명 |
|-----|------|------|
| 1 | **V6-NoLoRA** | NF subnet의 LoRA를 일반 Linear로 대체 |
| 2 | **V6-TaskSeparated** | Task별 완전 분리 학습 (base 공유 없음) |
| 1+2 | **V6-NoLoRA-TaskSep** | 위 두 가지 조합 |
| 3 | **V6-SpectralNorm** | Subnet에 Spectral Normalization 적용 |

### 수정된 파일

1. **moleflow/config/ablation.py**
   - `use_regular_linear`: LoRA 대신 일반 Linear 사용
   - `use_task_separated`: Task별 독립 훈련
   - `use_spectral_norm`: Spectral Normalization 적용

2. **moleflow/models/lora.py (MoLESubnet)**
   - `use_regular_linear=True`: nn.Linear로 대체, task별 별도 layer 생성
   - `use_spectral_norm=True`: nn.utils.spectral_norm 적용

3. **moleflow/models/mole_nf.py**
   - make_subnet에 V6 플래그 전달
   - add_task에서 task-separated 모드 처리

4. **moleflow/trainer/continual_trainer.py**
   - Task-separated 모드: task > 0도 _train_base_task 스타일로 훈련

### 기대 효과

1. **V6-NoLoRA**: Low-rank constraint 제거로 표현력 증가
2. **V6-TaskSeparated**: Task 간 간섭 완전 제거 (upper bound 측정)
3. **V6-SpectralNorm**: Lipschitz 제약으로 더 안정적인 flow

### 실행 스크립트

```bash
./run.sh  # GPU 0, 1, 4, 5에서 4개 실험 병렬 실행
```

### 결과

(실험 완료 후 기록 예정)

---

## Dataset Support - VISA & MPDD

### 개요

MVTec AD 외에 VisA(Visual Anomaly)와 MPDD(Metal Parts Defect Detection) 데이터셋 지원 추가.

### 데이터셋 구조

#### VisA Dataset (/Data/VISA)
- **Classes (12개)**: candle, capsules, cashew, chewinggum, fryum, macaroni1, macaroni2, pcb1, pcb2, pcb3, pcb4, pipe_fryum
- **구조**: CSV 기반 split (`split_csv/1cls.csv`)
- **이미지**: `{class}/Data/Images/{Normal|Anomaly}/*.JPG`
- **마스크**: `{class}/Data/Masks/Anomaly/*.png`

#### MPDD Dataset (/Data/mpdd)
- **Classes (6개)**: bracket_black, bracket_brown, bracket_white, connector, metal_plate, tubes
- **구조**: MVTec-AD 스타일 디렉토리 구조
- **이미지**: `{class}/{train|test}/{good|defect_type}/*.png`
- **마스크**: `{class}/ground_truth/{defect_type}/*_mask.png`

### 새로 추가된 파일

1. **moleflow/data/visa.py**
   - `VISA` 클래스: CSV 기반 데이터 로딩
   - `VISA_CLASS_NAMES`: 12개 클래스 목록

2. **moleflow/data/mpdd.py**
   - `MPDD` 클래스: MVTec-AD 스타일 디렉토리 스캔
   - `MPDD_CLASS_NAMES`: 6개 클래스 목록

### 수정된 파일

1. **moleflow/data/datasets.py**
   - `DATASET_REGISTRY`: 데이터셋 클래스 레지스트리
   - `get_dataset_class(name)`: 이름으로 데이터셋 클래스 반환
   - `get_class_names(name)`: 데이터셋의 클래스 목록 반환
   - `create_task_dataset()`: `args.dataset` 기반으로 자동 선택

2. **moleflow/data/__init__.py**
   - VISA, MPDD 관련 export 추가

3. **moleflow/__init__.py**
   - VISA, MPDD, 유틸리티 함수 export

4. **run_moleflow.py**
   - `--dataset` 인자 추가 (mvtec, visa, mpdd)
   - 로그에 dataset 정보 출력

### 사용법

```bash
# VisA 데이터셋으로 실험
python run_moleflow.py \
    --dataset visa \
    --data_path /Data/VISA \
    --task_classes candle capsules cashew

# MPDD 데이터셋으로 실험
python run_moleflow.py \
    --dataset mpdd \
    --data_path /Data/mpdd \
    --task_classes bracket_black bracket_brown connector

# 기본 MVTec (변경 없음)
python run_moleflow.py \
    --dataset mvtec \
    --data_path /Data/MVTecAD \
    --task_classes leather grid transistor
```

### 검증 결과

```
VISA candle train samples: 900
VISA candle test samples: 200
MPDD bracket_black train samples: 289
MPDD bracket_black test samples: 79
```

---

## Bug Fix - VISA/MPDD 데이터셋 평가 오류 수정 (2025-12-31)

### 문제
- VISA/MPDD 데이터셋으로 학습 후 **test 단계에서 에러 발생**
- 평가 함수가 MVTEC 데이터셋을 하드코딩하여 사용

### 원인
`moleflow/evaluation/evaluator.py`의 `evaluate_class`와 `evaluate_routing_performance` 함수가 `args.dataset` 값과 관계없이 항상 MVTEC 데이터셋 클래스를 사용:

```python
# 문제 코드
from moleflow.data.mvtec import MVTEC
test_dataset = MVTEC(args.data_path, class_name=class_name, ...)
```

### 해결책
`args.dataset`에 따라 적절한 데이터셋 클래스를 동적으로 선택하도록 수정:

```python
# 수정된 코드
from moleflow.data.datasets import get_dataset_class

dataset_name = getattr(args, 'dataset', 'mvtec')
DatasetClass = get_dataset_class(dataset_name)
test_dataset = DatasetClass(args.data_path, class_name=class_name, ...)
```

### 수정된 파일
- `moleflow/evaluation/evaluator.py`
  - `evaluate_class()` 함수
  - `evaluate_routing_performance()` 함수

---
