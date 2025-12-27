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