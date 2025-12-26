# MoLE-Flow: Mixture of LoRA Experts for Continual Anomaly Detection

## 1. Overview

MoLE-Flow는 **Continual Learning** 환경에서 **Anomaly Detection**을 수행하는 프레임워크입니다. 핵심 아이디어는 Normalizing Flow 기반의 anomaly detector에 **LoRA (Low-Rank Adaptation)** 를 적용하여, 새로운 task를 학습할 때 기존 지식을 보존하면서 효율적으로 adaptation하는 것입니다.

### 1.1 Problem Setting

```
Task 0: leather 학습 → Task 1: grid 학습 → Task 2: transistor 학습 → ...
```

- 각 task는 하나의 클래스(제품 유형)에 대한 anomaly detection
- **Catastrophic Forgetting 방지**: Task 1 학습 시 Task 0 성능 유지 필요
- **Inference 시 Task ID 모름**: Router를 통해 자동으로 task 선택

### 1.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           MoLE-Flow Pipeline                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input Image ──► ViT Feature Extractor ──► Positional Embedding         │
│       │              (DINOv2, frozen)              │                    │
│       │                    │                       │                    │
│       │                    ▼                       ▼                    │
│       │         [B, H, W, 512] patches    + 2D sinusoidal PE            │
│       │                    │                       │                    │
│       │                    └───────────┬───────────┘                    │
│       │                                ▼                                │
│       │                    ┌───────────────────────┐                    │
│       │                    │   TaskInputAdapter    │ ◄─── Task-specific │
│       │                    │   (FiLM-style)        │      distribution  │
│       │                    └───────────────────────┘      alignment     │
│       │                                │                                │
│       │                                ▼                                │
│       │                    ┌───────────────────────┐                    │
│       │                    │   Normalizing Flow    │                    │
│       │                    │   ┌─────────────────┐ │                    │
│       │                    │   │ Base Weights    │ │ ◄─── Shared        │
│       │                    │   │ (frozen T>0)   │ │                    │
│       │                    │   ├─────────────────┤ │                    │
│       │                    │   │ LoRA Adapters   │ │ ◄─── Task-specific │
│       │                    │   │ (per task)      │ │                    │
│       │                    │   ├─────────────────┤ │                    │
│       │                    │   │ Task Biases     │ │ ◄─── Task-specific │
│       │                    │   └─────────────────┘ │                    │
│       │                    └───────────────────────┘                    │
│       │                                │                                │
│       │                                ▼                                │
│       │                    Latent z ~ N(0, I)                           │
│       │                                │                                │
│       │                                ▼                                │
│       │              Anomaly Score = -log p(x) = -log p(z) - log|det J| │
│       │                                                                 │
│       └─► PrototypeRouter ──► Task Selection (Mahalanobis distance)     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Components

### 2.1 ViT Feature Extractor

DINOv2 ViT를 사용하여 이미지에서 patch-level feature를 추출합니다.

**Motivation**:
- DINOv2는 self-supervised learning으로 학습되어 범용적인 visual representation 제공
- Patch 단위 feature는 pixel-level anomaly localization에 필수

```python
class ViTPatchCoreExtractor(nn.Module):
    """
    ViT-based PatchCore Feature Extractor.
    Uses transformer block outputs directly without additional patchification.
    """

    def __init__(self,
                 backbone_name="vit_base_patch14_dinov2.lvd142m",
                 blocks_to_extract=[8, 9, 10, 11],
                 input_shape=(3, 224, 224),
                 target_embed_dimension=1024,
                 device='cuda',
                 remove_cls_token=True):
        super(ViTPatchCoreExtractor, self).__init__()

        self.device = device
        self.blocks_to_extract_from = blocks_to_extract
        self.remove_cls_token = remove_cls_token

        # Load backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.backbone.to(device)
        self.backbone.eval()

        # Setup feature aggregator
        self.feature_aggregator = ViTFeatureAggregator(
            self.backbone, self.blocks_to_extract_from
        )

        # Calculate embedding dimensions
        feature_dims = self.feature_aggregator.feature_dimensions(input_shape, device)

        # Setup preprocessing (dimension alignment)
        self.preprocessing = nn.ModuleList()
        for dim in feature_dims:
            if dim != target_embed_dimension:
                self.preprocessing.append(nn.Linear(dim, target_embed_dimension))
            else:
                self.preprocessing.append(nn.Identity())
        self.preprocessing.to(device)

        self.target_embed_dimension = target_embed_dimension
        self.num_blocks = len(blocks_to_extract)

    def forward(self, images, return_spatial_shape=False):
        """
        Extract patch embeddings from images.

        Args:
            images: (B, C, H, W) input images
            return_spatial_shape: If True, return spatial shape as well

        Returns:
            features: (B, H_patch, W_patch, Target_Embed_Dimension)
            spatial_shape (optional): (H_patch, W_patch) tuple
        """
        images = images.to(self.device)
        batch_size = images.shape[0]

        # 1. Extract features from transformer blocks
        features_dict = self.feature_aggregator(images)

        # 2. Collect features from each block (B, N_tokens, D)
        features = []
        for block_idx in self.blocks_to_extract_from:
            feat = features_dict[f"block_{block_idx}"]

            # Remove CLS token
            if self.remove_cls_token:
                feat = feat[:, 1:, :]  # (B, N_patches, D)

            features.append(feat)

        # 3. Apply preprocessing (dimension adjustment)
        processed_features = []
        for i, feat in enumerate(features):
            feat = self.preprocessing[i](feat)
            processed_features.append(feat)

        # 4. Aggregate features from multiple blocks
        stacked_features = torch.stack(processed_features, dim=2)
        aggregated_features = stacked_features.mean(dim=2)

        # 5. Reshape to spatial structure
        num_patches = aggregated_features.shape[1]
        patch_grid_size = int(np.sqrt(num_patches))

        output = aggregated_features.reshape(
            batch_size, patch_grid_size, patch_grid_size, self.target_embed_dimension
        )

        if return_spatial_shape:
            return output, (patch_grid_size, patch_grid_size)

        return output

    def get_image_level_features(self, images):
        """
        Return image-level features via global average pooling.
        Used for prototype-based routing.

        Returns:
            features: (Batch_Size, Target_Embed_Dimension)
        """
        spatial_embeddings = self.forward(images, return_spatial_shape=False)
        # Global average pooling: (B, H, W, D) -> (B, D)
        image_features = spatial_embeddings.mean(dim=(1, 2))
        return image_features
```

---

### 2.2 Positional Embedding

2D sinusoidal positional encoding을 patch embedding에 추가합니다.

**Motivation**:
- Normalizing Flow는 patch의 위치 정보를 모름
- 위치에 따라 normal/anomaly 패턴이 다를 수 있음 (예: 제품 가장자리 vs 중앙)

```python
def positionalencoding2d(D, H, W, device=None):
    """
    Generate 2D positional encoding.

    Args:
        D: dimension of the model
        H: height of the positions
        W: width of the positions
        device: device to put the tensor on

    Returns:
        DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W, device=device)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2, device=device) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W, device=device).unsqueeze(1)
    pos_h = torch.arange(0.0, H, device=device).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


class PositionalEmbeddingGenerator:
    """Generate and add 2D positional encodings to patch embeddings."""

    def __init__(self, device):
        self.device = device

    def __call__(self, spatial_shape, patch_embeddings):
        """
        Add positional embeddings to patch embeddings.

        Args:
            spatial_shape: (H_patch, W_patch) spatial dimensions
            patch_embeddings: (B, H, W, D) patch embeddings

        Returns:
            (B, H, W, D) patch embeddings with positional encoding
        """
        H_patch, W_patch = spatial_shape
        pos_embed_dim = patch_embeddings.shape[-1]

        # Generate positional encoding
        pos_embed = positionalencoding2d(pos_embed_dim, H_patch, W_patch, device=self.device)

        # Reshape and expand for batch
        pos_embed_expanded = pos_embed.unsqueeze(0).permute(0, 2, 3, 1)
        batch_size = patch_embeddings.shape[0]
        pos_embed_batch = pos_embed_expanded.repeat(batch_size, 1, 1, 1)

        # Add positional embedding
        patch_embeddings_with_pos = patch_embeddings + pos_embed_batch

        return patch_embeddings_with_pos
```

---

### 2.3 LoRA Linear Layer

**Motivation**:
- 전체 weight를 fine-tuning하면 이전 task 지식이 손실됨
- LoRA는 low-rank delta만 학습하여 parameter-efficient하면서 이전 지식 보존

```python
class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear Layer with Task-specific Bias.

    Output: h(x) = W_base @ x + scaling * (B @ A) @ x + (base_bias + task_bias)

    Key Design Principles:
    1. SMALL SCALING: Use smaller alpha/rank ratio for stable initial adaptation
    2. ZERO-INIT B: Ensures delta_W = 0 at start (pure identity mapping for LoRA part)
    3. XAVIER-INIT A: Better than Kaiming for symmetric distributions
    4. TASK BIAS: Handles distribution shift without modifying base model
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4,
                 alpha: float = 1.0, bias: bool = True,
                 use_lora: bool = True, use_task_bias: bool = True):
        super(LoRALinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        # Increased scaling for stronger LoRA contribution
        # Changed from alpha/(2*rank) to alpha/rank for 2x effect
        self.scaling = alpha / rank
        self.use_bias = bias

        # Ablation flags
        self.use_lora = use_lora
        self.use_task_bias = use_task_bias

        # Base weight (frozen after Task 1)
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapters storage: Dict[task_id -> (A, B)]
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()

        # Task-specific biases (critical for handling distribution shift)
        self.task_biases = nn.ParameterDict()

        # Current active task
        self.active_task_id: Optional[int] = None
        self.base_frozen = False

    def add_task_adapter(self, task_id: int):
        """
        Add LoRA adapter and task-specific bias for a new task.

        Initialization Strategy:
        - A: Xavier uniform (better for maintaining gradient flow)
        - B: Zero (ensures delta_W = 0 at start)
        - task_bias: Zero (starts at base_bias + 0)
        """
        task_key = str(task_id)
        device = self.base_linear.weight.device

        # Add LoRA adapters only if enabled
        if self.use_lora:
            # A: Xavier uniform initialization (better gradient flow than Kaiming)
            A = nn.Parameter(torch.zeros(self.rank, self.in_features, device=device))
            nn.init.xavier_uniform_(A)

            # B: Zero initialization (ensures delta_W = 0 at start, pure identity for LoRA)
            B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device))

            self.lora_A[task_key] = A
            self.lora_B[task_key] = B

        # Task-specific bias: Initialize to zero (starts at base_bias + 0)
        if self.use_bias and self.use_task_bias:
            task_bias = nn.Parameter(torch.zeros(self.out_features, device=device))
            self.task_biases[task_key] = task_bias

    def freeze_base(self):
        """Freeze base weights after Task 1 (but NOT the base bias for reference)."""
        self.base_linear.weight.requires_grad = False
        if self.use_bias and self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False
        self.base_frozen = True

    def unfreeze_base(self):
        """Unfreeze base weights (for Task 1 training)."""
        for param in self.base_linear.parameters():
            param.requires_grad = True
        self.base_frozen = False

    def set_active_task(self, task_id: Optional[int]):
        """Set the currently active LoRA adapter."""
        self.active_task_id = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional LoRA adapter and task-specific bias.

        h(x) = W_base @ x + scaling * (B @ A) @ x + (base_bias + task_bias)
        """
        # Check if task adapter is active
        if self.active_task_id is not None:
            task_key = str(self.active_task_id)
            has_lora = task_key in self.lora_A and task_key in self.lora_B
            has_task_bias = task_key in self.task_biases

            # If we have any task-specific components
            if has_lora or has_task_bias:
                # Compute W_base @ x (without bias)
                output = F.linear(x, self.base_linear.weight, bias=None)

                # Add LoRA contribution: scaling * (B @ A) @ x
                if has_lora and self.use_lora:
                    A = self.lora_A[task_key]
                    B = self.lora_B[task_key]
                    lora_output = F.linear(F.linear(x, A), B)
                    output = output + self.scaling * lora_output

                # Add bias
                if self.use_bias and self.base_linear.bias is not None:
                    if has_task_bias and self.use_task_bias:
                        # Combined bias: base_bias + task_bias
                        total_bias = self.base_linear.bias + self.task_biases[task_key]
                        output = output + total_bias
                    else:
                        # Only base bias
                        output = output + self.base_linear.bias

                return output

        # Default: use base linear (Task 0 or no adapter)
        return self.base_linear(x)

    def get_merged_weight(self, task_id: int) -> torch.Tensor:
        """Get merged weight W' = W_base + scaling * B @ A for a specific task."""
        task_key = str(task_id)
        merged = self.base_linear.weight.data.clone()

        if task_key in self.lora_A and task_key in self.lora_B:
            A = self.lora_A[task_key]
            B = self.lora_B[task_key]
            merged = merged + self.scaling * (B @ A)

        return merged
```

---

### 2.4 MoLE Subnet

NF Coupling Block에서 사용되는 s/t network입니다.

```python
class MoLESubnet(nn.Module):
    """
    MoLE Subnet for NF Coupling Blocks.

    Architecture: Linear -> ReLU -> Linear
    With LoRA adapters on both linear layers.
    """

    def __init__(self, dims_in: int, dims_out: int, rank: int = 4, alpha: float = 1.0,
                 use_lora: bool = True, use_task_bias: bool = True):
        super(MoLESubnet, self).__init__()

        hidden_dim = 2 * dims_in

        self.use_lora = use_lora
        self.use_task_bias = use_task_bias

        self.layer1 = LoRALinear(dims_in, hidden_dim, rank=rank, alpha=alpha,
                                  use_lora=use_lora, use_task_bias=use_task_bias)
        self.relu = nn.ReLU()
        self.layer2 = LoRALinear(hidden_dim, dims_out, rank=rank, alpha=alpha,
                                  use_lora=use_lora, use_task_bias=use_task_bias)

    def add_task_adapter(self, task_id: int):
        """Add LoRA adapters for a new task."""
        self.layer1.add_task_adapter(task_id)
        self.layer2.add_task_adapter(task_id)

    def freeze_base(self):
        """Freeze base weights."""
        self.layer1.freeze_base()
        self.layer2.freeze_base()

    def unfreeze_base(self):
        """Unfreeze base weights."""
        self.layer1.unfreeze_base()
        self.layer2.unfreeze_base()

    def set_active_task(self, task_id: Optional[int]):
        """Set active LoRA adapter."""
        self.layer1.set_active_task(task_id)
        self.layer2.set_active_task(task_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer2(self.relu(self.layer1(x)))
```

---

### 2.5 Task Input Adapter (FiLM-style)

**Motivation**:
- 각 task(제품 유형)마다 feature distribution이 다름
- Base NF가 학습한 Task 0 distribution에 맞춰 입력을 정규화

```python
class TaskInputAdapter(nn.Module):
    """
    FiLM-style Task Input Adapter (v3).

    Key Design Principles:
    1. FiLM (Feature-wise Linear Modulation): y = gamma * x + beta
    2. Layer Norm option to preserve spatial information
    3. Larger MLP capacity with active residual gate
    4. Can work with or without reference statistics

    v3 Changes from v2:
    - Instance Norm → Layer Norm (preserves spatial info)
    - residual_gate: 0 → 0.5 (MLP actively used from start)
    - Larger hidden_dim for more capacity
    - Optional use_norm flag for Task 0 self-adaptation
    """

    def __init__(self, channels: int, reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None, use_norm: bool = True):
        super(TaskInputAdapter, self).__init__()

        self.channels = channels
        self.eps = 1e-6
        self.use_norm = use_norm

        # Store reference statistics (from Task 0) - used for target distribution
        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
            self.has_reference = True
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))
            self.has_reference = False

        # FiLM parameters: y = gamma * x + beta
        # Initialize gamma=1, beta=0 for identity start
        self.film_gamma = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.film_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

        # Larger MLP for stronger feature transformation
        hidden_dim = max(channels // 2, 128)  # Increased from channels//4
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )

        # Initialize last layer to small values (not zero) for faster learning
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

        # Residual gate - starts at 0.5 for active MLP contribution
        # Changed from 0.0 to enable MLP from the beginning
        self.residual_gate = nn.Parameter(torch.tensor([0.5]))

        # Optional Layer Norm (preserves spatial info better than Instance Norm)
        if use_norm:
            self.layer_norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FiLM-style feature modulation with residual MLP.

        Flow: x -> (optional LayerNorm) -> FiLM -> MLP residual

        Args:
            x: (B, H, W, D) input features

        Returns:
            Transformed features with same shape
        """
        B, H, W, D = x.shape
        identity = x

        # 1. Optional normalization (Layer Norm preserves spatial structure)
        if self.use_norm:
            x_flat = x.reshape(-1, D)
            x_normed = self.layer_norm(x_flat)
            x = x_normed.reshape(B, H, W, D)

        # 2. FiLM modulation: y = gamma * x + beta
        x = self.film_gamma * x + self.film_beta

        # 3. MLP residual with learnable gate
        gate = torch.sigmoid(self.residual_gate)  # 0 ~ 1
        x_flat = x.reshape(-1, D)
        mlp_out = self.mlp(x_flat).reshape(B, H, W, D)
        x = x + gate * mlp_out

        # 4. Optional: blend with identity for stability
        # This helps Task 0 where we want minimal transformation
        if not self.has_reference:
            # For Task 0 self-adapter: stronger identity connection
            x = 0.9 * identity + 0.1 * x

        return x
```

---

### 2.6 Feature Statistics

Task 0에서 학습된 reference distribution 통계를 저장합니다.

```python
class FeatureStatistics:
    """
    Store reference feature statistics from Task 0 for distribution alignment.

    Key Insight: We DON'T normalize new task features using Task 0 statistics directly.
    Instead, we store these statistics and pass them to TaskInputAdapter,
    which learns to transform new task features to match the reference distribution.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.n_samples = 0

        # Welford's online algorithm for stable mean/variance computation
        self._M2: Optional[torch.Tensor] = None  # Sum of squared deviations

    def update(self, features: torch.Tensor):
        """
        Update running statistics using Welford's online algorithm.
        More stable than simple EMA for computing variance.

        Args:
            features: (B, H, W, D) or (N, D) features
        """
        # Flatten to (N, D)
        if features.dim() == 4:
            B, H, W, D = features.shape
            features = features.reshape(-1, D)

        features = features.detach()
        batch_size = features.shape[0]

        if self.mean is None:
            self.mean = features.mean(dim=0)
            self._M2 = ((features - self.mean.unsqueeze(0)) ** 2).sum(dim=0)
            self.n_samples = batch_size
        else:
            # Welford's online update
            for i in range(batch_size):
                self.n_samples += 1
                delta = features[i] - self.mean
                self.mean = self.mean + delta / self.n_samples
                delta2 = features[i] - self.mean
                self._M2 = self._M2 + delta * delta2

    def finalize(self):
        """Compute final std and mark as initialized."""
        if self._M2 is not None and self.n_samples > 1:
            variance = self._M2 / (self.n_samples - 1)
            self.std = torch.sqrt(variance + 1e-6)
        else:
            self.std = torch.ones_like(self.mean)

        self.is_initialized = True

    def get_reference_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get reference mean and std for TaskInputAdapter initialization."""
        if not self.is_initialized:
            raise ValueError("Statistics not finalized. Call finalize() first.")
        return self.mean.clone(), self.std.clone()
```

---

### 2.7 MoLE Spatial-Aware NF (Main Model)

```python
class MoLESpatialAwareNF(nn.Module):
    """
    MoLE-Flow: Mixture of LoRA Experts for Normalizing Flow.

    Key Features:
    - Base NF weights shared across all tasks
    - Task-specific LoRA adapters for distribution shift
    - Task-specific biases for mean shift adaptation
    - Task-specific input adapters for feature normalization
    - Feature statistics alignment for cross-task consistency
    - Zero-initialization for stable adaptation
    """

    def __init__(self,
                 embed_dim: int = 512,
                 coupling_layers: int = 8,
                 clamp_alpha: float = 1.9,
                 lora_rank: int = 32,
                 lora_alpha: float = 1.0,
                 device: str = 'cuda',
                 ablation_config: 'AblationConfig' = None):
        super(MoLESpatialAwareNF, self).__init__()

        self.embed_dim = embed_dim
        self.coupling_layers = coupling_layers
        self.clamp_alpha = clamp_alpha
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.device = device

        # Ablation configuration
        if ablation_config is None:
            self.use_lora = True
            self.use_task_adapter = True
            self.use_task_bias = True
        else:
            self.use_lora = ablation_config.use_lora
            self.use_task_adapter = ablation_config.use_task_adapter
            self.use_task_bias = ablation_config.use_task_bias

        # Track tasks
        self.num_tasks = 0
        self.current_task_id: Optional[int] = None

        # Build flow
        self.subnets: List[MoLESubnet] = []
        self.flow = self._build_flow()

        # Feature statistics from Task 0 (reference distribution)
        self.reference_stats = FeatureStatistics(device=device)

        # Task-specific input adapters for pre-conditioning
        self.input_adapters = nn.ModuleDict()

        self.to(device)

    def _build_flow(self) -> Ff.SequenceINN:
        """Build normalizing flow with MoLE coupling blocks."""

        def make_subnet(dims_in, dims_out):
            subnet = MoLESubnet(dims_in, dims_out,
                               rank=self.lora_rank,
                               alpha=self.lora_alpha,
                               use_lora=self.use_lora,
                               use_task_bias=self.use_task_bias)
            self.subnets.append(subnet)
            return subnet

        coder = Ff.SequenceINN(self.embed_dim)

        for k in range(self.coupling_layers):
            coder.append(
                Fm.AllInOneBlock,
                subnet_constructor=make_subnet,
                affine_clamping=self.clamp_alpha,
                global_affine_type='SOFTPLUS',
                permute_soft=True
            )

        return coder

    def add_task(self, task_id: int):
        """
        Add LoRA adapters and input adapter for a new task.

        NEW Design (Task 0 also uses LoRA):
        - Task 0: Train base weights + LoRA adapter (equal treatment)
        - Task > 0: Freeze base, add LoRA + task bias + input adapter

        Key Benefits:
        - Task 0 is no longer "special" - all tasks use LoRA for adaptation
        - Base NF learns general feature transformation
        - LoRA handles task-specific adaptation uniformly
        """
        task_key = str(task_id)

        if task_id == 0:
            # Task 0: Train base weights + LoRA adapter + InputAdapter (self-adaptation)
            for subnet in self.subnets:
                subnet.unfreeze_base()
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(task_id)

            # NEW v3: Add InputAdapter for Task 0 as well (self-adaptation)
            if self.use_task_adapter:
                self.input_adapters[task_key] = TaskInputAdapter(
                    self.embed_dim,
                    reference_mean=None,
                    reference_std=None,
                    use_norm=True
                ).to(self.device)

        else:
            # Task > 0: Freeze base, add LoRA adapters + task biases
            for subnet in self.subnets:
                subnet.freeze_base()
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(task_id)

            # Create input adapter with reference statistics
            if self.use_task_adapter:
                if self.reference_stats.is_initialized:
                    ref_mean, ref_std = self.reference_stats.get_reference_params()
                else:
                    ref_mean, ref_std = None, None

                self.input_adapters[task_key] = TaskInputAdapter(
                    self.embed_dim,
                    reference_mean=ref_mean,
                    reference_std=ref_std
                ).to(self.device)

        self.num_tasks = task_id + 1
        self.set_active_task(task_id)

    def forward(self, patch_embeddings_with_pos: torch.Tensor, reverse: bool = False):
        """
        Forward or inverse transformation with task-specific input pre-conditioning.

        Args:
            patch_embeddings_with_pos: (B, H, W, D) spatial patch embeddings
            reverse: If True, generate samples from latent

        Returns:
            z: (B, H, W, D) latent or reconstructed embeddings
            log_jac_det: (B,) log determinant of Jacobian
        """
        B, H, W, D = patch_embeddings_with_pos.shape
        x = patch_embeddings_with_pos

        # Apply task-specific input adapter for ALL tasks (if enabled)
        if self.use_task_adapter and not reverse and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.input_adapters:
                x = self.input_adapters[task_key](x)

        # Flatten spatial dimensions
        x_flat = x.reshape(B * H * W, D)

        # Flow transformation
        if not reverse:
            z_flat, log_jac_det = self.flow(x_flat)
            log_jac_det = log_jac_det.reshape(B, H * W).sum(dim=1)
        else:
            z_flat, log_jac_det = self.flow(x_flat, rev=True)
            log_jac_det = log_jac_det.reshape(B, H * W).sum(dim=1)

        # Reshape back to spatial
        z = z_flat.reshape(B, H, W, D)

        return z, log_jac_det

    def log_prob(self, patch_embeddings_with_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of patch embeddings.

        Returns:
            log_prob: (B,) log probability for each sample
        """
        B, H, W, D = patch_embeddings_with_pos.shape

        # Forward transformation
        z, log_jac_det = self.forward(patch_embeddings_with_pos, reverse=False)

        # Base distribution: N(0, I)
        z_flat = z.reshape(B, -1)
        log_pz = -0.5 * (z_flat ** 2).sum(dim=1) - 0.5 * (H * W * D) * math.log(2 * math.pi)

        # Change of variables: log p(x) = log p(z) + log|det J|
        log_px = log_pz + log_jac_det

        return log_px
```

---

### 2.8 Prototype Router

**Motivation**:
- Inference 시 task ID를 모름
- Image-level feature의 Mahalanobis distance로 가장 가까운 task 선택

```python
class TaskPrototype:
    """
    Task Prototype for Distance-based Routing.

    Stores:
    - mu_t: Mean of image-level features
    - Sigma_t^{-1}: Precision matrix (inverse of covariance) for Mahalanobis
    """

    def __init__(self, task_id: int, task_classes: List[str], device: str = 'cuda'):
        self.task_id = task_id
        self.task_classes = task_classes
        self.device = device

        self.mean: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None
        self.covariance: Optional[torch.Tensor] = None
        self.n_samples: int = 0

    def update(self, features: torch.Tensor):
        """Update prototype statistics with new features."""
        features = features.detach()

        if self.mean is None:
            self.mean = features.mean(dim=0)
            self.n_samples = features.shape[0]

            # Compute covariance with regularization
            centered = features - self.mean.unsqueeze(0)
            self.covariance = (centered.T @ centered) / (features.shape[0] - 1)

            # Add regularization for numerical stability
            reg = 1e-5 * torch.eye(features.shape[1], device=features.device)
            self.covariance = self.covariance + reg
        else:
            # Incremental update (Welford's online algorithm)
            n_new = features.shape[0]
            n_total = self.n_samples + n_new

            new_mean = features.mean(dim=0)
            delta = new_mean - self.mean

            self.mean = (self.n_samples * self.mean + n_new * new_mean) / n_total

            centered_new = features - new_mean.unsqueeze(0)
            cov_new = (centered_new.T @ centered_new) / (n_new - 1) if n_new > 1 else torch.zeros_like(self.covariance)

            self.covariance = (self.n_samples * self.covariance + n_new * cov_new +
                              (self.n_samples * n_new / n_total) * torch.outer(delta, delta)) / n_total

            self.n_samples = n_total

    def finalize(self):
        """Compute precision matrix after all updates."""
        if self.covariance is not None:
            try:
                self.precision = torch.linalg.inv(self.covariance)
            except:
                self.precision = torch.linalg.pinv(self.covariance)

    def mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance from prototype.

        D_M = sqrt((x - mu)^T Sigma^{-1} (x - mu))
        """
        centered = features - self.mean.unsqueeze(0)
        distances = torch.sqrt(torch.sum(centered @ self.precision * centered, dim=1))
        return distances


class PrototypeRouter:
    """
    Prototype-based Router for Task Selection.

    Uses Mahalanobis distance to select the best LoRA expert.
    """

    def __init__(self, device: str = 'cuda', use_mahalanobis: bool = True):
        self.device = device
        self.use_mahalanobis = use_mahalanobis
        self.prototypes: Dict[int, TaskPrototype] = {}

    def add_prototype(self, task_id: int, prototype: TaskPrototype):
        """Add a task prototype."""
        self.prototypes[task_id] = prototype

    def route(self, features: torch.Tensor) -> torch.Tensor:
        """
        Route features to the best task based on distance.

        Returns:
            task_ids: (N,) predicted task IDs
        """
        if not self.prototypes:
            return torch.zeros(features.shape[0], dtype=torch.long, device=features.device)

        all_distances = []
        task_ids = sorted(self.prototypes.keys())

        for task_id in task_ids:
            if self.use_mahalanobis:
                distances = self.prototypes[task_id].mahalanobis_distance(features)
            else:
                distances = self.prototypes[task_id].euclidean_distance(features)
            all_distances.append(distances)

        all_distances = torch.stack(all_distances, dim=0)
        min_indices = torch.argmin(all_distances, dim=0)
        predicted_tasks = torch.tensor([task_ids[idx] for idx in min_indices],
                                        device=features.device)

        return predicted_tasks
```

---

## 3. Training Strategy

### 3.1 Task 0: Base + LoRA Training

```python
def _train_base_task(self, task_id, task_classes, train_loader, num_epochs, lr, log_interval):
    """Train base NF + LoRA for Task 0."""
    trainable_params = self.nf_model.get_trainable_params(task_id)

    optimizer = create_optimizer(list(trainable_params), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    for epoch in range(num_epochs):
        for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
            images = images.to(self.device)

            with torch.no_grad():
                patch_embeddings, spatial_shape = self.vit_extractor(
                    images, return_spatial_shape=True
                )
                patch_embeddings_with_pos = self.pos_embed_generator(
                    spatial_shape, patch_embeddings
                )
                # Collect feature statistics for future tasks
                self.nf_model.update_reference_stats(patch_embeddings_with_pos)

            # Forward through NF
            z, log_jac_det = self.nf_model.forward(patch_embeddings_with_pos)

            # NLL Loss
            nll_loss = self._compute_nll_loss(z, log_jac_det)

            optimizer.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(trainable_params), max_norm=0.5)
            optimizer.step()

        scheduler.step()

    # Finalize feature statistics
    self.nf_model.finalize_reference_stats()
```

### 3.2 Task > 0: FAST Adaptation

```python
def _train_fast_stage(self, task_id, task_classes, train_loader, num_epochs, lr, log_interval):
    """Stage 1: FAST Adaptation (LoRA + InputAdapter, base frozen)."""
    self.nf_model.freeze_all_base()  # Freeze base weights

    fast_params = self.nf_model.get_fast_params(task_id)  # Only LoRA + InputAdapter

    optimizer = create_optimizer(list(fast_params), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=lr * 0.01
    )

    warmup_epochs = 2

    for epoch in range(num_epochs):
        # Warmup learning rate
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * warmup_factor

        for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
            images = images.to(self.device)

            with torch.no_grad():
                patch_embeddings, spatial_shape = self.vit_extractor(
                    images, return_spatial_shape=True
                )
                patch_embeddings_with_pos = self.pos_embed_generator(
                    spatial_shape, patch_embeddings
                )

            z, log_jac_det = self.nf_model.forward(patch_embeddings_with_pos)
            nll_loss = self._compute_nll_loss(z, log_jac_det)

            optimizer.zero_grad()
            nll_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(fast_params), max_norm=0.5)
            optimizer.step()

        if epoch >= warmup_epochs:
            scheduler.step()
```

### 3.3 NLL Loss

```python
def _compute_nll_loss(self, z: torch.Tensor, log_jac_det: torch.Tensor) -> torch.Tensor:
    """
    Compute NLL loss.

    Standard NLL: L = -log p(x) = -log p(z) - log |det J|

    Args:
        z: Latent tensor [B, H, W, D]
        log_jac_det: Log Jacobian determinant [B]

    Returns:
        NLL loss (scalar)
    """
    B, H, W, D = z.shape

    # Standard NLL (averaged over all patches)
    log_pz = -0.5 * (z ** 2).sum() / (B * H * W)
    log_jac = log_jac_det.mean() / (H * W)
    nll_loss = -(log_pz + log_jac)

    return nll_loss
```

---

## 4. Inference

```python
def inference(self, images: torch.Tensor, task_id: Optional[int] = None):
    """
    Inference with automatic routing or specified task.

    Args:
        images: (B, C, H, W) input images
        task_id: If None, use router to predict task

    Returns:
        anomaly_scores: (B, H_patch, W_patch) spatial anomaly map
        image_scores: (B,) image-level scores
        predicted_tasks: (B,) predicted task IDs
    """
    self.nf_model.eval()
    self.vit_extractor.eval()

    with torch.no_grad():
        images = images.to(self.device)

        # Route if task_id not specified
        if task_id is None and self.router is not None:
            image_features = self.vit_extractor.get_image_level_features(images)
            predicted_tasks = self.router.route(image_features)
        else:
            predicted_tasks = torch.full((images.shape[0],),
                                        task_id if task_id is not None else 0,
                                        dtype=torch.long, device=self.device)

        # Process each unique task
        unique_tasks = predicted_tasks.unique()

        for t_id in unique_tasks:
            mask = (predicted_tasks == t_id)
            task_images = images[mask]

            # Set active task (LoRA adapter)
            self.nf_model.set_active_task(t_id.item())

            # Extract features
            patch_embeddings, spatial_shape = self.vit_extractor(
                task_images, return_spatial_shape=True
            )
            patch_embeddings_with_pos = self.pos_embed_generator(
                spatial_shape, patch_embeddings
            )

            # Compute anomaly scores
            task_anomaly_scores, task_image_scores = self._compute_anomaly_scores(
                patch_embeddings_with_pos
            )

    return anomaly_scores, image_scores, predicted_tasks


def _compute_anomaly_scores(self, patch_embeddings_with_pos: torch.Tensor):
    """Compute anomaly scores from patch embeddings."""
    B, H, W, D = patch_embeddings_with_pos.shape

    # Forward through NF
    z, log_jac_det = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

    # Compute patch-level log p(z)
    z_reshaped = z.reshape(B, H, W, -1)
    log_pz_per_patch = -0.5 * (z_reshaped ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

    # Distribute log_jac_det across patches
    log_jac_per_patch = log_jac_det.reshape(B, 1, 1).expand(B, H, W) / (H * W)

    # Patch-level log-likelihood
    patch_log_prob = log_pz_per_patch + log_jac_per_patch

    # Anomaly score = -log p(x)
    anomaly_scores = -patch_log_prob

    # Image-level score (99th percentile)
    image_scores = torch.quantile(anomaly_scores.reshape(B, -1), 0.99, dim=1)

    return anomaly_scores, image_scores
```

---

## 5. Key Design Decisions

### 5.1 왜 LoRA인가?

| 방법 | 장점 | 단점 |
|------|------|------|
| Full Fine-tuning | 최대 표현력 | Catastrophic Forgetting |
| Feature Replay | Forgetting 방지 | 메모리 사용량, Privacy |
| EWC | Regularization 기반 | 복잡한 Fisher 계산 |
| **LoRA** | **Parameter Efficient + Forgetting 방지** | Capacity 제한 |

### 5.2 왜 Task 0도 LoRA를 사용하는가? (v2 → v3)

**v1/v2 문제**: Task 0에서는 Base만 학습 → Base가 Task 0에 편향

**v3 해결**: Task 0도 Base + LoRA 동시 학습
- Base: 범용 feature transformation
- LoRA: Task-specific adaptation
- 모든 Task가 동등한 구조

### 5.3 왜 FiLM-style InputAdapter인가? (v3)

**v2 문제**: Instance Norm이 spatial 정보 손실

**v3 해결**:
- Layer Norm (spatial 보존)
- FiLM modulation (gamma * x + beta)
- Active residual gate (0.5로 시작)

### 5.4 LoRA Scaling

```python
# v1: scaling = alpha / (2 * rank)  # 1.56% contribution
# v2+: scaling = alpha / rank        # 3.125% contribution (2x stronger)
```

---

## 6. Experiment Results (v3 Best)

| Version | Image AUC | Pixel AUC | Key Change |
|---------|-----------|-----------|------------|
| v1 | 0.8286 | 0.8628 | Baseline |
| v2 | 0.9307 | 0.9238 | LoRA scaling 2x, Task 0 LoRA |
| **v3** | **0.9504** | **0.9313** | FiLM InputAdapter, Task 0 self-adapt |

---

## 7. Usage

```bash
python run_moleflow.py \
    --task_classes leather grid transistor \
    --num_epochs 40 \
    --backbone_name vit_base_patch14_dinov2.lvd142m \
    --img_size 518 \
    --num_coupling_layers 8 \
    --lora_rank 32 \
    --experiment_name my_experiment
```
