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
# moleflow/models/whitening_adapter.py
class WhiteningAdapter(nn.Module):
    """
    Whitening + Constrained De-whitening for distribution alignment.

    Forward: x → whiten → constrained de-whiten → x'
    - Whitening uses running statistics (updated during training)
    - De-whitening uses learnable but constrained parameters
    """
    def __init__(self, channels, constraint_scale=0.1):
        # Running statistics for whitening
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))

        # Constrained de-whitening parameters
        # γ ∈ [1-δ, 1+δ], β ∈ [-δ, δ]
        self.dewhiten_gamma = nn.Parameter(torch.ones(channels))
        self.dewhiten_beta = nn.Parameter(torch.zeros(channels))
        self.constraint_scale = constraint_scale

    def forward(self, x):
        # Whitening: (x - μ) / σ
        x_whitened = (x - self.running_mean) / (self.running_var.sqrt() + 1e-5)

        # Constrained de-whitening
        gamma = 1.0 + self.constraint_scale * torch.tanh(self.dewhiten_gamma - 1.0)
        beta = self.constraint_scale * torch.tanh(self.dewhiten_beta)

        return gamma * x_whitened + beta
```

### Key Design
1. **Running Statistics**: Task별 업데이트, 현재 task의 분포 반영
2. **Constrained Parameters**: tanh로 범위 제한 → 안정적 학습
3. **Per-Task Adapter**: 각 task마다 별도 WhiteningAdapter

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
# moleflow/models/dia.py
class DeepInvertibleAdapter(nn.Module):
    """
    Task-specific mini normalizing flow after base NF.

    Provides nonlinear manifold adaptation while maintaining invertibility.
    """
    def __init__(self, channels, n_blocks=2, hidden_ratio=0.5):
        self.blocks = nn.ModuleList([
            InvertibleBlock(channels, hidden_ratio) for _ in range(n_blocks)
        ])

    def forward(self, z, reverse=False):
        logdet = 0
        blocks = reversed(self.blocks) if reverse else self.blocks

        for block in blocks:
            z, ld = block(z, reverse=reverse)
            logdet = logdet + ld

        return z, logdet

class InvertibleBlock(nn.Module):
    """Affine coupling block for DIA."""
    def __init__(self, channels, hidden_ratio=0.5):
        hidden_dim = int(channels * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(channels // 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)  # s and t
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=-1)
        st = self.net(x1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * 0.5  # Bounded scale

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            logdet = s.sum(dim=-1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            logdet = -s.sum(dim=-1)

        return torch.cat([x1, y2], dim=-1), logdet
```

### Integration in mole_nf.py
```python
class MoLESpatialAwareNF(nn.Module):
    def __init__(self, ...):
        # ...
        self.dia_adapters = nn.ModuleDict()  # Per-task DIA

    def add_task_adapter(self, task_id):
        if self.use_dia:
            self.dia_adapters[str(task_id)] = DeepInvertibleAdapter(
                self.c_in, self.dia_n_blocks, self.dia_hidden_ratio
            )

    def forward(self, x, reverse=False):
        # Base NF forward
        z, logdet = self.inn(x)

        # DIA forward
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
g' = g - Σᵢ (g · vᵢ) vᵢ
```
where vᵢ = important directions from previous tasks

### Implementation
```python
# moleflow/models/ogp.py
class OrthogonalGradientProjection:
    """
    Projects gradients to the null space of important subspaces
    from previous tasks to prevent catastrophic forgetting.
    """
    def __init__(self, threshold=0.99, max_rank_per_task=50, device='cuda'):
        self.threshold = threshold
        self.max_rank = max_rank_per_task
        self.device = device
        self.projection_matrices = {}  # Per-parameter projection

    def compute_basis(self, model, dataloader, task_id):
        """Compute important gradient directions for current task."""
        gradients = self._collect_gradients(model, dataloader)

        for name, grads in gradients.items():
            # SVD to find important directions
            G = torch.stack(grads)  # (N, D)
            U, S, V = torch.svd(G)

            # Select top-k directions (explain threshold% variance)
            cumsum = torch.cumsum(S**2, 0) / (S**2).sum()
            k = (cumsum < self.threshold).sum() + 1
            k = min(k, self.max_rank)

            # Store projection matrix: I - V_k @ V_k.T
            V_k = V[:, :k]
            if name not in self.projection_matrices:
                self.projection_matrices[name] = V_k
            else:
                # Merge with existing basis
                combined = torch.cat([self.projection_matrices[name], V_k], dim=1)
                U_c, S_c, V_c = torch.svd(combined)
                self.projection_matrices[name] = V_c[:, :self.max_rank]

    def project_gradient(self, model):
        """Project current gradients to null space of stored basis."""
        for name, param in model.named_parameters():
            if param.grad is None or name not in self.projection_matrices:
                continue

            V = self.projection_matrices[name]
            g = param.grad.view(-1)

            # g' = g - V @ V.T @ g
            g_proj = g - V @ (V.T @ g)
            param.grad = g_proj.view(param.grad.shape)
```

### Integration in Trainer
```python
class MoLEContinualTrainer:
    def __init__(self, ...):
        if self.use_ogp:
            self.ogp = OrthogonalGradientProjection(
                threshold=self.ogp_threshold,
                max_rank_per_task=self.ogp_max_rank
            )

    def _train_fast_stage(self, task_id, ...):
        for batch in dataloader:
            loss.backward()

            # OGP: Project gradients
            if self.use_ogp and self.ogp.is_initialized:
                self.ogp.project_gradient(self.nf_model)

            optimizer.step()

    def _after_task_training(self, task_id, dataloader):
        # Compute OGP basis after task training
        if self.use_ogp:
            self.ogp.compute_basis(self.nf_model, dataloader, task_id)
```

### Command Line
```bash
python run_moleflow.py \
    --use_ogp \
    --ogp_threshold 0.99 \
    --ogp_max_rank 50
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
| `moleflow/models/whitening_adapter.py` | NEW: WhiteningAdapter module |
| `moleflow/models/ms_context.py` | NEW: LightweightMSContext module |
| `moleflow/models/dia.py` | NEW: DeepInvertibleAdapter module |
| `moleflow/models/ogp.py` | NEW: OrthogonalGradientProjection |
| `moleflow/models/routing.py` | TwoStageHybridRouter 추가 |
| `moleflow/models/mole_nf.py` | DIA integration, V3 options |
| `moleflow/config/ablation.py` | V3 options: use_dia, use_ogp, use_whitening_adapter, use_ms_context |
| `moleflow/trainer/continual_trainer.py` | OGP integration |
| `run_moleflow.py` | V3 CLI arguments, config saving |
| `run_v3_experiments.sh` | V3 ablation experiment script |

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
