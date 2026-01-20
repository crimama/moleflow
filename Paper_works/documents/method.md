# MoLE-Flow: Method Description (Version 5-Final)

## Overview

MoLE-Flow (Mixture of LoRA Experts for Normalizing Flow)는 **Continual Anomaly Detection** 문제를 해결하기 위한 프레임워크입니다. 여러 제품 클래스(Task)를 순차적으로 학습하면서, 이전에 학습한 Task의 성능을 유지합니다.

**핵심 문제**: Task 0 (예: leather) 학습 → Task 1 (예: grid) 학습 → Task 2 (예: transistor) 학습... 이 과정에서 catastrophic forgetting 없이 모든 Task에서 좋은 anomaly detection 성능을 유지해야 합니다.

**지원 데이터셋**:
- MVTec AD (15 classes)
- VisA (12 classes)
- MPDD (6 classes)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              MoLE-Flow Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Input Image (224×224×3)                                                        │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  ViTPatchCoreExtractor (Frozen)                                         │   │
│  │  • Backbone: vit_base_patch16_224.augreg2_in21k_ft_in1k                 │   │
│  │  • Multi-layer aggregation: blocks [1, 3, 5, 11]                        │   │
│  │  • Output: (B, 14, 14, 768) patch features                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  PositionalEmbeddingGenerator                                           │   │
│  │  • 2D sinusoidal positional encoding                                    │   │
│  │  • positionalencoding2d(D, H, W): (D, H, W) → concat with features     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  WhiteningAdapter (Task-specific) [moleflow/models/adapters.py:437]     │   │
│  │  • Whitening: LayerNorm(x, elementwise_affine=False)                   │   │
│  │  • De-whitening: γ * whitened + β                                      │   │
│  │  • γ ∈ [0.5, 2.0] via sigmoid, β ∈ [-2.0, 2.0] via tanh               │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  SpatialContextMixer (Frozen after Task 0) [adapters.py]                │   │
│  │  • 3×3 depthwise conv for local context                                 │   │
│  │  • Learnable residual gate for context blending                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  MoLESpatialAwareNF (Base NF) [moleflow/models/mole_nf.py:31]          │   │
│  │  • 8 Affine Coupling Layers (FrEIA AllInOneBlock)                       │   │
│  │  • MoLESubnet with LoRALinear layers                                    │   │
│  │  • Base weights: Frozen after Task 0                                    │   │
│  │  • Task-specific: LoRA adapters (rank=64) + Task biases                │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  DeepInvertibleAdapter (DIA) [moleflow/models/lora.py:774]             │   │
│  │  • 2 additional coupling blocks per task                                │   │
│  │  • AffineCouplingBlock with SimpleSubnet                                │   │
│  │  • Near-identity initialization                                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│       │                                                                         │
│       ▼                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  Anomaly Score Computation                                              │   │
│  │  • Patch score: -log p(z) - log|det J| = 0.5*||z||² - logdet           │   │
│  │  • Image score: Top-K aggregation (k=3)                                 │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components (Actual Implementation)

### 1. Feature Extractor: ViTPatchCoreExtractor

**File**: `moleflow/extractors/vit_extractor.py`

```python
class ViTPatchCoreExtractor:
    """
    ViT-based patch feature extractor with multi-layer aggregation.

    Key Parameters:
    - backbone_name: "vit_base_patch16_224.augreg2_in21k_ft_in1k"
    - blocks_to_extract: [1, 3, 5, 11]  # Multi-scale features
    - target_embed_dimension: 768
    - remove_cls_token: True  # Only patch tokens
    """

    def __init__(self, backbone_name, blocks_to_extract, input_shape,
                 target_embed_dimension, device, remove_cls_token=True):
        # Create timm model with feature extraction
        self.model = timm.create_model(backbone_name, pretrained=True)
        self.model.eval()  # Always frozen

        # Register forward hooks for intermediate layers
        for block_idx in blocks_to_extract:
            self.model.blocks[block_idx].register_forward_hook(...)
```

**출력 형태**:
- Input: (B, 3, 224, 224) RGB image
- Output: (B, 196, 768) = (B, 14×14, 768) patch features

---

### 2. Positional Embedding: PositionalEmbeddingGenerator

**File**: `moleflow/models/position_embedding.py`

```python
def positionalencoding2d(d_model: int, height: int, width: int, device='cuda'):
    """
    2D sinusoidal positional encoding.

    pe[0::4, :, :] = sin(x / 10000^(4i/d))
    pe[1::4, :, :] = cos(x / 10000^(4i/d))
    pe[2::4, :, :] = sin(y / 10000^(4i/d))
    pe[3::4, :, :] = cos(y / 10000^(4i/d))

    Returns: (d_model, height, width)
    """
    pe = torch.zeros(d_model, height, width, device=device)
    d_model_quarter = d_model // 4

    div_term = torch.exp(torch.arange(0., d_model_quarter, device=device) *
                        -(math.log(10000.0) / d_model_quarter))

    pos_w = torch.arange(0., width, device=device).unsqueeze(1)
    pos_h = torch.arange(0., height, device=device).unsqueeze(1)

    pe[0::4, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[1::4, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, height, 1)
    pe[2::4, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)
    pe[3::4, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, width)

    return pe
```

---

### 3. WhiteningAdapter (Task Input Adapter)

**File**: `moleflow/models/adapters.py:437-552`

```python
class WhiteningAdapter(nn.Module):
    """
    Whitening-based Task Adapter.

    Design Principles:
    1. All tasks go through Whitening first (mean=0, std=1)
    2. Task-specific de-whitening with constrained parameters
    3. Task 0 stays close to identity (anchor point)

    Parameters:
    - gamma: constrained to [0.5, 2.0] via sigmoid
    - beta: constrained to [-2.0, 2.0] via tanh
    """

    def __init__(self, channels: int, task_id: int = 0,
                 gamma_range: tuple = (0.5, 2.0), beta_max: float = 2.0):
        super().__init__()

        self.gamma_min, self.gamma_max = gamma_range
        self.beta_max = beta_max

        # Whitening layer (NO learnable affine - pure normalization)
        self.whiten = nn.LayerNorm(channels, elementwise_affine=False)

        if task_id == 0:
            # Task 0: Initialize close to identity (γ≈1.0, β≈0)
            # sigmoid(-0.7) ≈ 0.33 → γ = 0.5 + 1.5*0.33 ≈ 1.0
            init_gamma_raw = -0.7 * torch.ones(1, 1, 1, channels)
            self.gamma_raw = nn.Parameter(init_gamma_raw)
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.1  # Regularize toward identity
        else:
            # Task 1+: Start at midpoint
            self.gamma_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.identity_reg_weight = 0.0

    @property
    def gamma(self):
        """Constrained gamma in [gamma_min, gamma_max]."""
        gamma_range = self.gamma_max - self.gamma_min
        return self.gamma_min + gamma_range * torch.sigmoid(self.gamma_raw)

    @property
    def beta(self):
        """Constrained beta in [-beta_max, beta_max]."""
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, D) input features
        Returns:
            Transformed features with same shape
        """
        B, H, W, D = x.shape

        # Step 1: Whitening (mean=0, std=1)
        x_flat = x.reshape(-1, D)
        x_white = self.whiten(x_flat).reshape(B, H, W, D)

        # Step 2: De-whitening with constrained params
        x_out = self.gamma * x_white + self.beta

        return x_out
```

**설계 의도**:
- **Whitening**: 모든 Task가 동일한 normalized representation을 공유
- **Constrained De-whitening**: Task별 adaptation이 제한적 범위 내에서만 가능
- **Identity Regularization**: Task 0가 base anchor가 됨

---

### 4. Normalizing Flow: MoLESpatialAwareNF

**File**: `moleflow/models/mole_nf.py:31-1212`

```python
class MoLESpatialAwareNF(nn.Module):
    """
    Main NF model with LoRA adapters.

    Key Attributes:
    - coupling_layers: 8 (default)
    - lora_rank: 64 (default)
    - clamp_alpha: 1.9 (numerical stability)
    """

    def __init__(self, embed_dim=512, coupling_layers=8, clamp_alpha=1.9,
                 lora_rank=32, lora_alpha=1.0, device='cuda', ablation_config=None):
        super().__init__()

        self.embed_dim = embed_dim
        self.coupling_layers = coupling_layers
        self.subnets: List[MoLESubnet] = []

        # Build flow using FrEIA
        self.flow = self._build_flow()

        # Task-specific components
        self.input_adapters = nn.ModuleDict()   # WhiteningAdapter per task
        self.dia_adapters = nn.ModuleDict()     # DIA per task

    def _build_flow(self) -> Ff.SequenceINN:
        """Build normalizing flow with MoLE coupling blocks."""

        def make_subnet(dims_in, dims_out):
            subnet = MoLESubnet(
                dims_in, dims_out,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                use_lora=self.use_lora,
                use_task_bias=self.use_task_bias
            )
            self.subnets.append(subnet)
            return subnet

        coder = Ff.SequenceINN(self.embed_dim)

        for k in range(self.coupling_layers):
            coder.append(
                Fm.AllInOneBlock,
                subnet_constructor=make_subnet,
                affine_clamping=self.clamp_alpha,
                global_affine_type='SOFTPLUS',
                permute_soft=False  # Hard permutation
            )

        return coder

    def add_task(self, task_id: int):
        """Add LoRA adapters and input adapter for a new task."""
        task_key = str(task_id)

        if task_id == 0:
            # Task 0: Train base weights + LoRA + InputAdapter
            for subnet in self.subnets:
                subnet.unfreeze_base()
                subnet.add_task_adapter(task_id)

            # Add WhiteningAdapter
            self.input_adapters[task_key] = WhiteningAdapter(
                channels=self.embed_dim, task_id=task_id
            ).to(self.device)

            # Add DIA
            self.dia_adapters[task_key] = DeepInvertibleAdapter(
                channels=self.embed_dim, task_id=task_id, n_blocks=2
            ).to(self.device)
        else:
            # Task > 0: Freeze base, add only task-specific components
            for subnet in self.subnets:
                subnet.freeze_base()
                subnet.add_task_adapter(task_id)

            # Create WhiteningAdapter with reference stats
            self.input_adapters[task_key] = WhiteningAdapter(
                channels=self.embed_dim, task_id=task_id,
                reference_mean=ref_mean, reference_std=ref_std
            ).to(self.device)

            # Add DIA
            self.dia_adapters[task_key] = DeepInvertibleAdapter(
                channels=self.embed_dim, task_id=task_id, n_blocks=2
            ).to(self.device)

        self.num_tasks = task_id + 1
        self.set_active_task(task_id)

    def forward(self, patch_embeddings_with_pos: torch.Tensor, reverse: bool = False):
        """
        Forward transformation.

        Args:
            patch_embeddings_with_pos: (B, H, W, D)
            reverse: If True, generate samples from latent

        Returns:
            z: (B, H, W, D) latent
            logdet_patch: (B, H, W) patch-wise log determinant
        """
        B, H, W, D = patch_embeddings_with_pos.shape
        x = patch_embeddings_with_pos

        # Apply task-specific input adapter
        if self.use_task_adapter and not reverse and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.input_adapters:
                x = self.input_adapters[task_key](x)

        # Apply spatial context mixing
        if self.spatial_mixer is not None and not reverse:
            x = self.spatial_mixer(x)

        # Flatten for flow: (B, H, W, D) -> (BHW, D)
        x_flat = x.reshape(B * H * W, D)

        # Flow transformation
        z_flat, log_jac_det_flat = self.flow(x_flat)

        # Reshape: (BHW,) -> (B, H, W)
        logdet_patch = log_jac_det_flat.reshape(B, H, W)
        z = z_flat.reshape(B, H, W, D)

        # Apply DIA (Deep Invertible Adapter)
        if self.use_dia and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.dia_adapters:
                z, dia_logdet = self.dia_adapters[task_key](z, reverse=reverse)
                logdet_patch = logdet_patch + dia_logdet

        return z, logdet_patch
```

---

### 5. LoRALinear Layer

**File**: `moleflow/models/lora.py:13-166`

```python
class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear Layer with Task-specific Bias.

    Output: h(x) = W_base @ x + scaling * (B @ A) @ x + (base_bias + task_bias)

    Key Design:
    - scaling = alpha / rank (e.g., 1.0/64 = 0.0156)
    - A: Xavier uniform initialization
    - B: Zero initialization (ΔW = 0 at start)
    - task_bias: Zero initialization
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4,
                 alpha: float = 1.0, bias: bool = True,
                 use_lora: bool = True, use_task_bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank  # LoRA contribution scaling

        # Base weight (frozen after Task 0)
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapters: Dict[task_id -> (A, B)]
        self.lora_A = nn.ParameterDict()  # (rank, in_features)
        self.lora_B = nn.ParameterDict()  # (out_features, rank)

        # Task-specific biases
        self.task_biases = nn.ParameterDict()

        self.active_task_id: Optional[int] = None
        self.base_frozen = False

    def add_task_adapter(self, task_id: int):
        """Add LoRA adapter and task-specific bias for a new task."""
        task_key = str(task_id)
        device = self.base_linear.weight.device

        if self.use_lora:
            # A: Xavier uniform (good gradient flow)
            A = nn.Parameter(torch.zeros(self.rank, self.in_features, device=device))
            nn.init.xavier_uniform_(A)

            # B: Zero (ΔW = 0 at start, pure identity for LoRA)
            B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device))

            self.lora_A[task_key] = A
            self.lora_B[task_key] = B

        if self.use_bias and self.use_task_bias:
            task_bias = nn.Parameter(torch.zeros(self.out_features, device=device))
            self.task_biases[task_key] = task_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: h(x) = W_base @ x + scaling * (B @ A) @ x + bias
        """
        if self.active_task_id is not None:
            task_key = str(self.active_task_id)
            has_lora = task_key in self.lora_A
            has_task_bias = task_key in self.task_biases

            if has_lora or has_task_bias:
                # Base transformation (without bias)
                output = F.linear(x, self.base_linear.weight, bias=None)

                # Add LoRA contribution
                if has_lora and self.use_lora:
                    A = self.lora_A[task_key]
                    B = self.lora_B[task_key]
                    lora_output = F.linear(F.linear(x, A), B)
                    output = output + self.scaling * lora_output

                # Add combined bias
                if self.use_bias and self.base_linear.bias is not None:
                    if has_task_bias:
                        total_bias = self.base_linear.bias + self.task_biases[task_key]
                    else:
                        total_bias = self.base_linear.bias
                    output = output + total_bias

                return output

        return self.base_linear(x)
```

---

### 6. MoLESubnet (Coupling Block Subnet)

**File**: `moleflow/models/lora.py:168-283`

```python
class MoLESubnet(nn.Module):
    """
    Subnet for NF Coupling Blocks.

    Architecture: Linear(D→2D) → ReLU → Linear(2D→D)
    With LoRA adapters on both linear layers.
    """

    def __init__(self, dims_in: int, dims_out: int, rank: int = 4, alpha: float = 1.0,
                 use_lora: bool = True, use_task_bias: bool = True):
        super().__init__()

        hidden_dim = 2 * dims_in

        self.layer1 = LoRALinear(dims_in, hidden_dim, rank=rank, alpha=alpha,
                                  use_lora=use_lora, use_task_bias=use_task_bias)
        self.layer2 = LoRALinear(hidden_dim, dims_out, rank=rank, alpha=alpha,
                                  use_lora=use_lora, use_task_bias=use_task_bias)
        self.relu = nn.ReLU()

    def add_task_adapter(self, task_id: int):
        self.layer1.add_task_adapter(task_id)
        self.layer2.add_task_adapter(task_id)

    def freeze_base(self):
        self.layer1.freeze_base()
        self.layer2.freeze_base()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.relu(self.layer1(x)))
```

---

### 7. Deep Invertible Adapter (DIA)

**File**: `moleflow/models/lora.py:774-974`

```python
class DeepInvertibleAdapter(nn.Module):
    """
    Deep Invertible Adapter (DIA).

    Key Insight:
    Instead of linear LoRA (W + BA), we add a small task-specific Flow
    AFTER the base NF. This allows nonlinear manifold adaptation.

    Architecture:
    - 2 AffineCouplingBlock (alternating split direction)
    - Near-identity initialization for stable start

    Mathematical Formulation:
    - z_base = f_base(x)
    - z_final = f_DIA_t(z_base)
    - log p(x) = log p(z_final) + log|det J_base| + log|det J_DIA|
    """

    def __init__(self, channels: int, task_id: int, n_blocks: int = 2,
                 hidden_ratio: float = 0.5, clamp_alpha: float = 1.9):
        super().__init__()

        hidden_dim = int(channels * hidden_ratio)

        self.coupling_blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.coupling_blocks.append(
                AffineCouplingBlock(
                    channels=channels,
                    hidden_dim=hidden_dim,
                    clamp_alpha=clamp_alpha,
                    reverse=(i % 2 == 1)  # Alternate split direction
                )
            )

        # Initialize to near-identity
        self._initialize_near_identity()

    def forward(self, x: torch.Tensor, reverse: bool = False):
        """
        Args:
            x: (B, H, W, D)
            reverse: If True, compute inverse

        Returns:
            y: Transformed output
            log_det: (B, H, W) log determinant
        """
        B, H, W, D = x.shape
        log_det = torch.zeros(B, H, W, device=x.device)

        blocks = self.coupling_blocks if not reverse else reversed(self.coupling_blocks)

        for block in blocks:
            x, block_log_det = block(x, reverse=reverse)
            log_det = log_det + block_log_det

        return x, log_det


class AffineCouplingBlock(nn.Module):
    """
    Affine Coupling Block for DIA.

    Split: x = [x1, x2]
    Transform:
        y1 = x1
        y2 = x2 * exp(s(x1)) + t(x1)

    Log-determinant: sum(s(x1))
    """

    def __init__(self, channels: int, hidden_dim: int,
                 clamp_alpha: float = 1.9, reverse: bool = False):
        super().__init__()

        self.split_dim = channels // 2
        self.clamp_alpha = clamp_alpha
        self.reverse_split = reverse

        # Scale network
        self.s_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)
        # Translation network
        self.t_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        B, H, W, D = x.shape

        # Split
        if self.reverse_split:
            x2, x1 = x[..., :self.split_dim], x[..., self.split_dim:]
        else:
            x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        x1_flat = x1.reshape(-1, self.split_dim)
        s = self.s_net(x1_flat).reshape(B, H, W, self.split_dim)
        t = self.t_net(x1_flat).reshape(B, H, W, self.split_dim)

        # Clamp scale for numerical stability
        s = self.clamp_alpha * torch.tanh(s / self.clamp_alpha)

        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)
        else:
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)

        # Reconstruct
        if self.reverse_split:
            y = torch.cat([y2, x1], dim=-1)
        else:
            y = torch.cat([x1, y2], dim=-1)

        return y, log_det
```

---

### 8. Prototype Router

**File**: `moleflow/models/routing.py:24-184`

```python
class TaskPrototype:
    """
    Task Prototype for Mahalanobis distance-based routing.

    Stores:
    - mu_t: Mean of image-level features (D,)
    - Sigma_t^{-1}: Precision matrix (D, D)
    """

    def __init__(self, task_id: int, task_classes: List[str], device: str = 'cuda'):
        self.task_id = task_id
        self.task_classes = task_classes
        self.mean: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.covariance: Optional[torch.Tensor] = None

    def update(self, features: torch.Tensor):
        """Update prototype with new features using Welford's algorithm."""
        features = features.detach()

        if self.mean is None:
            self.mean = features.mean(dim=0)
            centered = features - self.mean.unsqueeze(0)
            self.covariance = (centered.T @ centered) / (features.shape[0] - 1)
            # Regularization for numerical stability
            reg = 1e-5 * torch.eye(features.shape[1], device=features.device)
            self.covariance = self.covariance + reg
        else:
            # Incremental update (Welford's online algorithm)
            ...

    def finalize(self):
        """Compute precision matrix after all updates."""
        self.precision = torch.linalg.inv(self.covariance)

    def mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        D_M = sqrt((x - μ)^T Σ^{-1} (x - μ))
        """
        centered = features - self.mean.unsqueeze(0)
        distances = torch.sqrt(torch.sum(centered @ self.precision * centered, dim=1))
        return distances


class PrototypeRouter:
    """Prototype-based Router for Task Selection."""

    def __init__(self, device: str = 'cuda', use_mahalanobis: bool = True):
        self.prototypes: Dict[int, TaskPrototype] = {}
        self.use_mahalanobis = use_mahalanobis

    def route(self, features: torch.Tensor) -> torch.Tensor:
        """Route features to best task based on minimum distance."""
        all_distances = []
        task_ids = sorted(self.prototypes.keys())

        for task_id in task_ids:
            if self.use_mahalanobis:
                dist = self.prototypes[task_id].mahalanobis_distance(features)
            else:
                dist = self.prototypes[task_id].euclidean_distance(features)
            all_distances.append(dist)

        all_distances = torch.stack(all_distances, dim=0)
        min_indices = torch.argmin(all_distances, dim=0)
        predicted_tasks = torch.tensor([task_ids[idx] for idx in min_indices],
                                        device=features.device)
        return predicted_tasks
```

---

## Training Procedure

### Task 0 (Base Task)

```python
# From moleflow/trainer/continual_trainer.py

def _train_base_task(self, train_loader, num_epochs, lr):
    """Train Task 0: Base NF + LoRA + WhiteningAdapter + DIA"""

    # All base weights are trainable
    self.nf_model.add_task(task_id=0)

    # Get trainable parameters
    params = self.nf_model.get_trainable_params(task_id=0)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for images, labels, masks, names, paths in train_loader:
            # 1. Extract features
            patch_embeddings, spatial_shape = self.vit_extractor(images)

            # 2. Add positional embedding
            features = self.pos_embed_generator(spatial_shape, patch_embeddings)

            # 3. Collect statistics for reference (used by Task 1+)
            self.nf_model.update_reference_stats(features)

            # 4. Forward through NF
            z, logdet_patch = self.nf_model(features)

            # 5. Compute loss
            loss = self._compute_nll_loss(z, logdet_patch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Finalize reference statistics
    self.nf_model.finalize_reference_stats()

    # Build prototype for routing
    self._build_prototype(task_id=0, train_loader)
```

### Task 1+ (Continual Tasks)

```python
def _train_continual_task(self, task_id, train_loader, num_epochs, lr):
    """Train Task > 0: Freeze base, train LoRA + Adapter + DIA"""

    # Add task-specific components (base is frozen)
    self.nf_model.add_task(task_id=task_id)

    # Only task-specific parameters are trainable
    params = self.nf_model.get_trainable_params(task_id=task_id)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    for epoch in range(num_epochs):
        for images, labels, masks, names, paths in train_loader:
            # Same forward pass
            patch_embeddings, spatial_shape = self.vit_extractor(images)
            features = self.pos_embed_generator(spatial_shape, patch_embeddings)
            z, logdet_patch = self.nf_model(features)
            loss = self._compute_nll_loss(z, logdet_patch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Build prototype for routing
    self._build_prototype(task_id, train_loader)
```

---

## Loss Function

### Base NLL Loss

```python
def _compute_nll_loss(self, z, logdet_patch):
    """
    Negative Log-Likelihood Loss.

    NLL = -log p(z) - log|det J|
        = 0.5 * ||z||² + D/2 * log(2π) - log|det J|
    """
    B, H, W, D = z.shape

    # Patch-wise log p(z)
    log_pz_patch = -0.5 * (z ** 2).sum(dim=-1)  # (B, H, W)

    # Patch-wise NLL = -log p(z) - log|det J|
    nll_patch = -log_pz_patch - logdet_patch  # (B, H, W)

    # Mean over patches
    loss = nll_patch.mean()

    return loss
```

### Tail-Aware Loss (V5)

```python
def _compute_tail_aware_loss(self, patch_losses, tail_weight=0.3, tail_top_k_ratio=0.05):
    """
    Tail-Aware Loss: Focus on extreme patches.

    L = (1-λ) * mean_loss + λ * tail_loss

    tail_loss = mean of top 5% highest-loss patches
    """
    mean_loss = patch_losses.mean()

    k = int(patch_losses.numel() * tail_top_k_ratio)
    k = max(k, 1)

    flat_losses = patch_losses.flatten()
    tail_loss = flat_losses.topk(k).values.mean()

    return (1 - tail_weight) * mean_loss + tail_weight * tail_loss
```

---

## Score Aggregation

### Top-K Aggregation (V5-Final)

```python
def _aggregate_scores(self, patch_scores, mode='top_k', top_k=3):
    """
    Aggregate patch-level scores to image-level.

    Args:
        patch_scores: (B, H, W) or (H, W)
        mode: 'mean', 'max', 'top_k'
        top_k: Number of top patches for top_k mode

    Returns:
        image_score: scalar or (B,)
    """
    if mode == 'mean':
        return patch_scores.mean()
    elif mode == 'max':
        return patch_scores.max()
    elif mode == 'top_k':
        flat = patch_scores.flatten()
        return flat.topk(top_k).values.mean()
```

---

## Inference Pipeline

```python
def inference(image, model, router, vit_extractor, pos_embed_generator):
    """Complete inference pipeline."""

    # 1. Feature extraction (frozen ViT)
    patch_embeddings, spatial_shape = vit_extractor(image)  # (B, 196, 768)

    # 2. Add positional embedding
    features = pos_embed_generator(spatial_shape, patch_embeddings)  # (B, 14, 14, 768)

    # 3. Route to correct task (Mahalanobis distance)
    image_features = features.mean(dim=(1, 2))  # (B, 768)
    task_id = router.route(image_features)  # (B,)

    # 4. Set active task
    model.set_active_task(task_id.item())

    # 5. Forward through NF with task's adapters
    z, logdet_patch = model(features)  # (B, 14, 14, 768), (B, 14, 14)

    # 6. Compute patch-level anomaly scores
    patch_scores = 0.5 * (z ** 2).sum(dim=-1) - logdet_patch  # (B, 14, 14)

    # 7. Aggregate to image-level score (Top-K)
    image_score = patch_scores.flatten(-2).topk(3).values.mean()

    return image_score, patch_scores
```

---

## Version 5-Final Configuration

```bash
# V5-Final Baseline Configuration
python run_moleflow.py \
    --dataset mvtec \
    --data_path /Data/MVTecAD \
    --task_classes leather grid transistor screw \
    --use_whitening_adapter \
    --use_dia \
    --score_aggregation_mode top_k \
    --score_aggregation_top_k 3 \
    --use_tail_aware_loss \
    --tail_weight 0.3 \
    --num_epochs 40 \
    --lr 1e-4 \
    --lora_rank 64
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| backbone_name | vit_base_patch16_224 | ViT backbone |
| img_size | 224 | Input image size |
| embed_dim | 768 | Feature dimension |
| num_coupling_layers | 8 | NF depth |
| lora_rank | 64 | LoRA rank |
| lora_alpha | 1.0 | LoRA scaling |
| use_whitening_adapter | True | WhiteningAdapter |
| use_dia | True | DIA (2 blocks) |
| score_aggregation_mode | top_k | Top-K aggregation |
| score_aggregation_top_k | 3 | k=3 |
| use_tail_aware_loss | True | Tail-aware loss |
| tail_weight | 0.3 | Tail loss weight |
| num_epochs | 40 | Training epochs |
| batch_size | 16 | Batch size |
| lr | 1e-4 | Learning rate |

---

## Parameter Count

### Per-Task Parameter Breakdown

| Component | Parameters | Notes |
|-----------|------------|-------|
| **LoRA per layer** | 2 × rank × dim = 2 × 64 × 768 = 98K | A: (64, 768), B: (768, 64) |
| **LoRA total** | 8 layers × 2 subnets × 98K = 1.57M | Per task |
| **Task Bias** | 8 × 2 × 768 = 12K | Per task |
| **WhiteningAdapter** | 2 × 768 = 1.5K | γ, β |
| **DIA (2 blocks)** | ~500K | Per task |
| **Total per task** | ~2.1M | |

### Shared Parameters (Frozen after Task 0)

| Component | Parameters |
|-----------|------------|
| Base NF weights | 8 layers × 2 × (768×1536 + 1536×768) = 18.9M |
| SpatialContextMixer | ~600K |
| **Total shared** | ~19.5M |

---

## Key Design Decisions

### 1. Why LoRA instead of full fine-tuning?

- **Parameter Efficiency**: rank=64로 전체 weight의 일부만 사용
- **Catastrophic Forgetting 방지**: Base weight frozen, task별 독립적 adapter
- **Scalability**: Task 수 증가해도 memory 효율적

### 2. Why WhiteningAdapter?

- **Distribution Alignment**: 모든 Task가 일관된 분포로 NF에 입력
- **Stable Training**: LayerNorm으로 정규화 후 제약된 de-whitening
- **Task Isolation**: Task별 γ, β로 독립적 적응

### 3. Why DIA?

- **Nonlinear Adaptation**: Linear LoRA의 한계 극복
- **Invertibility**: Flow의 density estimation property 유지
- **Complete Isolation**: Task별 완전한 parameter 분리

### 4. Why Top-K Score Aggregation?

- **Anomaly Localization**: 이상치는 소수 patch에 집중
- **Noise Robustness**: 단일 patch보다 안정적
- **Interpretability**: 어떤 patch가 기여했는지 추적 가능

---

## File Structure

```
moleflow/
├── __init__.py                    # Package exports
├── models/
│   ├── mole_nf.py                 # MoLESpatialAwareNF (main model)
│   ├── lora.py                    # LoRALinear, MoLESubnet, DIA
│   ├── adapters.py                # WhiteningAdapter, SpatialContextMixer
│   ├── routing.py                 # TaskPrototype, PrototypeRouter
│   └── position_embedding.py      # Positional encoding
├── extractors/
│   └── vit_extractor.py           # ViTPatchCoreExtractor
├── trainer/
│   └── continual_trainer.py       # MoLEContinualTrainer
├── evaluation/
│   └── evaluator.py               # Evaluation functions
├── data/
│   ├── datasets.py                # Dataset utilities
│   ├── mvtec.py                   # MVTec AD dataset
│   ├── visa.py                    # VisA dataset
│   └── mpdd.py                    # MPDD dataset
├── config/
│   └── ablation.py                # AblationConfig
└── utils/
    ├── logger.py                  # TrainingLogger
    ├── helpers.py                 # Utility functions
    └── diagnostics.py             # FlowDiagnostics
```

---

## References

- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
- Normalizing Flows for Probabilistic Modeling and Inference (Papamakarios et al., 2021)
- CFLOW-AD: Real-Time Unsupervised Anomaly Detection with Localization via Conditional Normalizing Flows (Gudovskiy et al., 2022)
- MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection (Bergmann et al., 2019)
- VisA: A Visual Anomaly Detection Dataset (Zou et al., 2022)
