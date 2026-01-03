"""
MoLE-Flow Integrated Model (Ultimate-Combo2 Default Configuration)

This file contains all components of MoLE-Flow (Mixture of LoRA Experts for Normalizing Flow)
integrated into a single file for easier understanding and deployment.

Default Configuration (Ultimate-Combo2):
- Backbone: wide_resnet50_2 (CNN)
- Embedding Dimension: 768
- LoRA Rank: 64
- Coupling Layers: 8
- Adapter Mode: whitening
- Spatial Context: depthwise_residual (kernel=3)
- Scale Context: enabled (kernel=3)
- DIA: 6 blocks
- LogDet Regularization: 1e-4

Key Features:
- Continual learning without catastrophic forgetting
- Task-specific LoRA adaptation
- Prototype-based routing for task selection
- Deep Invertible Adapters (DIA) for nonlinear manifold adaptation
"""

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import FrEIA.framework as Ff
import FrEIA.modules as Fm


# =============================================================================
# Default Configuration (Ultimate-Combo2)
# =============================================================================

@dataclass
class MoLEFlowConfig:
    """Default configuration for MoLE-Flow (Ultimate-Combo2 settings)."""

    # Model Architecture
    backbone_name: str = "wide_resnet50_2"
    backbone_type: str = "cnn"
    embed_dim: int = 768
    img_size: int = 224

    # Normalizing Flow
    num_coupling_layers: int = 8
    clamp_alpha: float = 1.9

    # LoRA Configuration
    lora_rank: int = 64
    lora_alpha: float = 1.0

    # Component Toggles (Ultimate-Combo2: all enabled)
    use_lora: bool = True
    use_task_adapter: bool = True
    use_task_bias: bool = True

    # Context Configuration
    use_spatial_context: bool = True
    spatial_context_kernel: int = 3
    use_scale_context: bool = True
    scale_context_kernel: int = 3
    scale_context_init_scale: float = 0.1
    scale_context_max_alpha: float = 0.2

    # DIA (Deep Invertible Adapter) Configuration
    use_dia: bool = True
    dia_n_blocks: int = 6
    dia_hidden_ratio: float = 0.5

    # Training Configuration
    lambda_logdet: float = 0.0001

    # Device
    device: str = "cuda"


# =============================================================================
# Position Embedding
# =============================================================================

def positionalencoding2d(D: int, H: int, W: int, device=None) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional encoding.

    Args:
        D: Embedding dimension (must be divisible by 4)
        H: Height of the position grid
        W: Width of the position grid
        device: Target device

    Returns:
        (D, H, W) positional encoding tensor
    """
    if D % 4 != 0:
        raise ValueError(f"Cannot use sin/cos PE with odd dimension (got dim={D})")

    P = torch.zeros(D, H, W, device=device)
    D_half = D // 2

    div_term = torch.exp(torch.arange(0.0, D_half, 2, device=device) * -(math.log(1e4) / D_half))
    pos_w = torch.arange(0.0, W, device=device).unsqueeze(1)
    pos_h = torch.arange(0.0, H, device=device).unsqueeze(1)

    P[0:D_half:2, :, :] = torch.sin(pos_w * div_term).T.unsqueeze(1).repeat(1, H, 1)
    P[1:D_half:2, :, :] = torch.cos(pos_w * div_term).T.unsqueeze(1).repeat(1, H, 1)
    P[D_half::2, :, :] = torch.sin(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, W)
    P[D_half+1::2, :, :] = torch.cos(pos_h * div_term).T.unsqueeze(2).repeat(1, 1, W)

    return P


class PositionalEmbeddingGenerator:
    """Generate and add 2D positional encodings to patch embeddings."""

    def __init__(self, device: str):
        self.device = device

    def __call__(self, spatial_shape: Tuple[int, int],
                 patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Add positional embeddings to patch embeddings.

        Args:
            spatial_shape: (H_patch, W_patch) spatial dimensions
            patch_embeddings: (B, H, W, D) patch embeddings

        Returns:
            (B, H, W, D) patch embeddings with positional encoding
        """
        H_patch, W_patch = spatial_shape
        D = patch_embeddings.shape[-1]
        B = patch_embeddings.shape[0]

        pos_embed = positionalencoding2d(D, H_patch, W_patch, device=self.device)
        pos_embed_expanded = pos_embed.unsqueeze(0).permute(0, 2, 3, 1)
        pos_embed_batch = pos_embed_expanded.repeat(B, 1, 1, 1)

        return patch_embeddings + pos_embed_batch


# =============================================================================
# Feature Extraction (CNN Backbone)
# =============================================================================

# Default layer config for wide_resnet50_2
CNN_LAYERS = ('layer2', 'layer3')


class LastLayerToExtractReachedException(Exception):
    """Exception to break forward pass after extracting last layer."""
    pass


class ForwardHook:
    """Hook to extract intermediate layer outputs."""

    def __init__(self, hook_dict: dict, layer_name: str, last_layer: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception = (layer_name == last_layer)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception:
            raise LastLayerToExtractReachedException()


class NetworkFeatureAggregator(nn.Module):
    """Extract features from specific layers of a backbone network."""

    def __init__(self, backbone: nn.Module, layers: tuple):
        super().__init__()
        self.layers = layers
        self.backbone = backbone
        self.outputs = {}

        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()

        for layer_name in layers:
            hook = ForwardHook(self.outputs, layer_name, layers[-1])

            if "." in layer_name:
                block, idx = layer_name.split(".")
                network_layer = backbone._modules[block]
                network_layer = network_layer[int(idx)] if idx.isnumeric() else network_layer._modules[idx]
            else:
                network_layer = backbone._modules[layer_name]

            target = network_layer[-1] if isinstance(network_layer, nn.Sequential) else network_layer
            self.backbone.hook_handles.append(target.register_forward_hook(hook))

    def forward(self, images: torch.Tensor) -> dict:
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape: tuple, device: str) -> List[int]:
        dummy = torch.ones([1] + list(input_shape)).to(device)
        output = self(dummy)
        return [output[layer].shape[1] for layer in self.layers]


class PatchMaker:
    """Convert feature maps to patches using unfold operation."""

    def __init__(self, patchsize: int, stride: int = 1):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features: torch.Tensor,
                 return_spatial_info: bool = False) -> Tuple[torch.Tensor, tuple]:
        """Convert (B, C, H, W) to (B*Grid, C, P, P) format."""
        padding = (self.patchsize - 1) // 2
        unfolder = nn.Unfold(kernel_size=self.patchsize, stride=self.stride,
                             padding=padding, dilation=1)
        unfolded = unfolder(features)

        num_patches = []
        for side in features.shape[-2:]:
            n = (side + 2 * padding - (self.patchsize - 1) - 1) / self.stride + 1
            num_patches.append(int(n))

        unfolded = unfolded.reshape(*features.shape[:2], self.patchsize, self.patchsize, -1)
        unfolded = unfolded.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded, num_patches
        return unfolded


class MeanMapper(nn.Module):
    """Reduce feature dimensions using adaptive average pooling."""

    def __init__(self, target_dim: int):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.target_dim).squeeze(1)


class Preprocessing(nn.Module):
    """Apply dimensionality reduction to multiple feature layers."""

    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.modules_list = nn.ModuleList([MeanMapper(output_dim) for _ in input_dims])

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        processed = [m(f) for m, f in zip(self.modules_list, features)]
        return torch.stack(processed, dim=1)


class CNNPatchCoreExtractor(nn.Module):
    """
    CNN-based PatchCore Feature Extractor.

    Extracts multi-scale features from CNN backbones (e.g., WideResNet50)
    and produces patch-level embeddings.
    """

    def __init__(self,
                 backbone_name: str = "wide_resnet50_2",
                 layers: tuple = None,
                 input_shape: tuple = (3, 224, 224),
                 patch_size: int = 3,
                 patch_stride: int = 1,
                 target_embed_dim: int = 768,
                 device: str = 'cuda'):
        super().__init__()

        self.device = device
        self.backbone_name = backbone_name

        if layers is None:
            layers = CNN_LAYERS
        self.layers = layers

        # Load pretrained backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=False)
        self.backbone.to(device)
        self.backbone.eval()

        # Feature extraction setup
        self.aggregator = NetworkFeatureAggregator(self.backbone, layers)
        feature_dims = self.aggregator.feature_dimensions(input_shape, device)

        self.patch_maker = PatchMaker(patch_size, stride=patch_stride)
        self.preprocessing = Preprocessing(feature_dims, target_embed_dim)
        self.preprocessing.to(device)

        self.target_embed_dim = target_embed_dim

        # Cache spatial shape
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape).to(device)
            features_dict = self.aggregator(dummy)
            feat = features_dict[layers[0]]
            _, (h, w) = self.patch_maker.patchify(feat, return_spatial_info=True)
            self._spatial_shape = (h, w)

    def _extract_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, tuple, int]:
        """Internal feature extraction."""
        images = images.to(self.device)
        batch_size = images.shape[0]

        features_dict = self.aggregator(images)
        features = [features_dict[layer] for layer in self.layers]

        features_info = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features_info]
        features = [x[0] for x in features_info]

        ref_shape = patch_shapes[0]

        # Align feature maps via interpolation
        for i in range(1, len(features)):
            f = features[i]
            ps = patch_shapes[i]
            f = f.reshape(f.shape[0], ps[0], ps[1], *f.shape[2:])
            f = f.permute(0, -3, -2, -1, 1, 2)
            base_shape = f.shape
            f = f.reshape(-1, *f.shape[-2:])
            f = F.interpolate(f.unsqueeze(1), size=ref_shape, mode="bilinear", align_corners=False)
            f = f.squeeze(1).reshape(*base_shape[:-2], *ref_shape)
            f = f.permute(0, -2, -1, 1, 2, 3).reshape(len(f), -1, *f.shape[3:6])
            features[i] = f

        features = [x.reshape(-1, *x.shape[-3:]) for x in features]
        features = self.preprocessing(features)
        features = features.mean(dim=1)

        return features, ref_shape, batch_size

    def forward(self, images: torch.Tensor,
                return_spatial_shape: bool = False) -> torch.Tensor:
        """
        Extract patch embeddings.

        Args:
            images: (B, C, H, W) input images
            return_spatial_shape: If True, also return spatial shape

        Returns:
            features: (B, H_patch, W_patch, D) patch embeddings
        """
        features, ref_shape, batch_size = self._extract_features(images)
        H, W = ref_shape
        features = features.reshape(batch_size, H, W, self.target_embed_dim)

        if return_spatial_shape:
            return features, (H, W)
        return features

    def get_image_level_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return image-level features via global average pooling."""
        spatial = self.forward(images)
        return spatial.mean(dim=(1, 2))


# =============================================================================
# LoRA (Low-Rank Adaptation) Module
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear Layer with Task-specific Bias.

    Output: h(x) = W_base @ x + scaling * (B @ A) @ x + (base_bias + task_bias)

    Key Design:
    - SMALL SCALING: Use alpha/rank ratio for stable adaptation
    - ZERO-INIT B: Ensures delta_W = 0 at start
    - XAVIER-INIT A: Better gradient flow
    - TASK BIAS: Handles distribution shift
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 64,
                 alpha: float = 1.0, bias: bool = True,
                 use_lora: bool = True, use_task_bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.use_bias = bias
        self.use_lora = use_lora
        self.use_task_bias = use_task_bias

        self.base_linear = nn.Linear(in_features, out_features, bias=bias)

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        self.task_biases = nn.ParameterDict()

        self.active_task_id: Optional[int] = None
        self.base_frozen = False

    def add_task_adapter(self, task_id: int):
        """Add LoRA adapter and task-specific bias for a new task."""
        task_key = str(task_id)
        device = self.base_linear.weight.device

        if self.use_lora:
            A = nn.Parameter(torch.zeros(self.rank, self.in_features, device=device))
            nn.init.xavier_uniform_(A)
            B = nn.Parameter(torch.zeros(self.out_features, self.rank, device=device))
            self.lora_A[task_key] = A
            self.lora_B[task_key] = B

        if self.use_bias and self.use_task_bias:
            self.task_biases[task_key] = nn.Parameter(torch.zeros(self.out_features, device=device))

    def freeze_base(self):
        """Freeze base weights after Task 0."""
        self.base_linear.weight.requires_grad = False
        if self.use_bias and self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False
        self.base_frozen = True

    def unfreeze_base(self):
        """Unfreeze base weights."""
        for param in self.base_linear.parameters():
            param.requires_grad = True
        self.base_frozen = False

    def set_active_task(self, task_id: Optional[int]):
        """Set the currently active LoRA adapter."""
        self.active_task_id = task_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional LoRA and task-specific bias."""
        if self.active_task_id is not None:
            task_key = str(self.active_task_id)
            has_lora = task_key in self.lora_A and task_key in self.lora_B
            has_bias = task_key in self.task_biases

            if has_lora or has_bias:
                output = F.linear(x, self.base_linear.weight, bias=None)

                if has_lora and self.use_lora:
                    A, B = self.lora_A[task_key], self.lora_B[task_key]
                    lora_out = F.linear(F.linear(x, A), B)
                    output = output + self.scaling * lora_out

                if self.use_bias and self.base_linear.bias is not None:
                    if has_bias and self.use_task_bias:
                        output = output + self.base_linear.bias + self.task_biases[task_key]
                    else:
                        output = output + self.base_linear.bias

                return output

        return self.base_linear(x)


class MoLEContextSubnet(nn.Module):
    """
    MoLE Subnet with Context-Aware Scale.

    Key Innovation:
    - s-network: concat(x, local_context) -> anomaly-sensitive scale
    - t-network: x only -> density-preserving shift

    This ensures scale(s) can detect "patches different from neighbors"
    while shift(t) preserves density estimation.
    """

    _spatial_info = None  # Class-level storage for spatial info

    def __init__(self, dims_in: int, dims_out: int, rank: int = 64, alpha: float = 1.0,
                 use_lora: bool = True, use_task_bias: bool = True,
                 context_kernel: int = 3, context_init_scale: float = 0.1,
                 context_max_alpha: float = 0.2):
        super().__init__()

        hidden_dim = 2 * dims_in
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.use_lora = use_lora
        self.use_task_bias = use_task_bias
        self.context_max_alpha = context_max_alpha

        # Context extraction (3x3 depthwise conv)
        self.context_conv = nn.Conv2d(dims_in, dims_in, kernel_size=context_kernel,
                                       padding=context_kernel // 2, groups=dims_in, bias=True)
        nn.init.zeros_(self.context_conv.weight)
        nn.init.zeros_(self.context_conv.bias)

        # Global alpha with sigmoid upper bound
        p = min(max(context_init_scale / context_max_alpha, 0.01), 0.99)
        init_param = torch.log(torch.tensor([p / (1 - p)]))
        self.context_scale_param = nn.Parameter(init_param)

        # s-network: context-aware (input: 2D)
        self.s_layer1 = LoRALinear(dims_in * 2, hidden_dim, rank=rank, alpha=alpha,
                                    use_lora=use_lora, use_task_bias=use_task_bias)
        self.s_layer2 = LoRALinear(hidden_dim, dims_out // 2, rank=rank, alpha=alpha,
                                    use_lora=use_lora, use_task_bias=use_task_bias)

        # t-network: context-free (input: D)
        self.t_layer1 = LoRALinear(dims_in, hidden_dim, rank=rank, alpha=alpha,
                                    use_lora=use_lora, use_task_bias=use_task_bias)
        self.t_layer2 = LoRALinear(hidden_dim, dims_out // 2, rank=rank, alpha=alpha,
                                    use_lora=use_lora, use_task_bias=use_task_bias)

        self.relu = nn.ReLU()

    def add_task_adapter(self, task_id: int):
        """Add LoRA adapters for all layers."""
        for layer in [self.s_layer1, self.s_layer2, self.t_layer1, self.t_layer2]:
            layer.add_task_adapter(task_id)

    def freeze_base(self):
        for layer in [self.s_layer1, self.s_layer2, self.t_layer1, self.t_layer2]:
            layer.freeze_base()

    def unfreeze_base(self):
        for layer in [self.s_layer1, self.s_layer2, self.t_layer1, self.t_layer2]:
            layer.unfreeze_base()

    def set_active_task(self, task_id: Optional[int]):
        for layer in [self.s_layer1, self.s_layer2, self.t_layer1, self.t_layer2]:
            layer.set_active_task(task_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with context-aware scale."""
        BHW, D = x.shape

        if MoLEContextSubnet._spatial_info is not None:
            B, H, W = MoLEContextSubnet._spatial_info
        else:
            B = 1
            H = W = int(math.sqrt(BHW))
            B = BHW // (H * W)

        # Extract local context
        x_spatial = x.view(B, H, W, D).permute(0, 3, 1, 2)
        ctx = self.context_conv(x_spatial)
        ctx = ctx.permute(0, 2, 3, 1).reshape(BHW, D)

        # Apply alpha scaling
        alpha = self.context_max_alpha * torch.sigmoid(self.context_scale_param)
        ctx = alpha * ctx

        # s-network: context-aware
        s_input = torch.cat([x, ctx], dim=-1)
        s = self.s_layer2(self.relu(self.s_layer1(s_input)))

        # t-network: context-free
        t = self.t_layer2(self.relu(self.t_layer1(x)))

        return torch.cat([s, t], dim=-1)

    def get_context_alpha(self) -> float:
        """Get current context alpha value."""
        with torch.no_grad():
            return (self.context_max_alpha * torch.sigmoid(self.context_scale_param)).item()


# =============================================================================
# Deep Invertible Adapter (DIA)
# =============================================================================

class SimpleSubnet(nn.Module):
    """Simple MLP subnet for DIA coupling blocks."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


class AffineCouplingBlock(nn.Module):
    """
    Affine Coupling Block for DIA.

    Split input into two halves, transform one conditioned on the other:
    y1 = x1
    y2 = x2 * exp(s(x1)) + t(x1)
    """

    def __init__(self, channels: int, hidden_dim: int,
                 clamp_alpha: float = 1.9, reverse: bool = False):
        super().__init__()

        self.channels = channels
        self.clamp_alpha = clamp_alpha
        self.reverse_split = reverse
        self.split_dim = channels // 2

        self.s_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)
        self.t_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
            y = torch.cat([y2, x1], dim=-1)
        else:
            y = torch.cat([x1, y2], dim=-1)

        return y, log_det


class DeepInvertibleAdapter(nn.Module):
    """
    Deep Invertible Adapter (DIA) - Task-specific nonlinear manifold adapter.

    Applied AFTER base NF for nonlinear manifold adaptation.
    Each task has its own DIA, achieving complete parameter isolation.
    """

    def __init__(self, channels: int, task_id: int, n_blocks: int = 6,
                 hidden_ratio: float = 0.5, clamp_alpha: float = 1.9):
        super().__init__()

        self.channels = channels
        self.task_id = task_id
        self.n_blocks = n_blocks
        self.clamp_alpha = clamp_alpha

        hidden_dim = int(channels * hidden_ratio)

        self.coupling_blocks = nn.ModuleList([
            AffineCouplingBlock(channels, hidden_dim, clamp_alpha, reverse=(i % 2 == 1))
            for i in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, W, D = x.shape
        log_det = torch.zeros(B, H, W, device=x.device)

        blocks = reversed(self.coupling_blocks) if reverse else self.coupling_blocks
        for block in blocks:
            x, block_log_det = block(x, reverse=reverse)
            log_det = log_det + block_log_det

        return x, log_det


# =============================================================================
# Adapters (Feature Statistics, Whitening, Spatial Context)
# =============================================================================

class FeatureStatistics:
    """Store reference feature statistics from Task 0 for distribution alignment."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.is_initialized = False
        self.n_samples = 0
        self._M2: Optional[torch.Tensor] = None

    def update(self, features: torch.Tensor):
        """Update running statistics using Welford's online algorithm."""
        if features.dim() == 4:
            features = features.reshape(-1, features.shape[-1])
        features = features.detach()
        batch_size = features.shape[0]

        if self.mean is None:
            self.mean = features.mean(dim=0)
            self._M2 = ((features - self.mean.unsqueeze(0)) ** 2).sum(dim=0)
            self.n_samples = batch_size
        else:
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
        print(f"Feature Statistics: n={self.n_samples}, mean_norm={self.mean.norm():.4f}, std_mean={self.std.mean():.4f}")

    def get_reference_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            raise ValueError("Statistics not finalized.")
        return self.mean.clone(), self.std.clone()


class WhiteningAdapter(nn.Module):
    """
    Whitening-based Task Adapter.

    Design:
    1. All tasks go through Whitening first (mean=0, std=1)
    2. Task-specific de-whitening with constrained parameters
    3. Task 0 stays close to identity (anchor point)
    """

    def __init__(self, channels: int, task_id: int = 0,
                 reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None,
                 gamma_range: tuple = (0.5, 2.0),
                 beta_max: float = 2.0):
        super().__init__()

        self.channels = channels
        self.task_id = task_id
        self.gamma_min, self.gamma_max = gamma_range
        self.beta_max = beta_max

        self.whiten = nn.LayerNorm(channels, elementwise_affine=False)

        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))

        if task_id == 0:
            init_gamma = -0.7 * torch.ones(1, 1, 1, channels)
            self.gamma_raw = nn.Parameter(init_gamma)
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
        else:
            self.gamma_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))
            self.beta_raw = nn.Parameter(torch.zeros(1, 1, 1, channels))

    @property
    def gamma(self):
        return self.gamma_min + (self.gamma_max - self.gamma_min) * torch.sigmoid(self.gamma_raw)

    @property
    def beta(self):
        return self.beta_max * torch.tanh(self.beta_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, D = x.shape
        x_white = self.whiten(x.reshape(-1, D)).reshape(B, H, W, D)
        return self.gamma * x_white + self.beta


class SpatialContextMixer(nn.Module):
    """
    Spatial Context Mixer for NF input preprocessing (depthwise_residual mode).

    Allows the Flow to detect:
    - "Is this patch abnormal COMPARED TO neighbors?" (local contrast)

    Uses 3x3 depthwise convolution with residual connection.
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Depthwise conv: each channel has its own kernel
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                                         stride=1, padding=self.padding, groups=channels, bias=True)
        nn.init.constant_(self.depthwise_conv.weight, 1.0 / (kernel_size * kernel_size))
        nn.init.zeros_(self.depthwise_conv.bias)

        # Learnable residual gate
        self.residual_gate = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply depthwise conv with residual connection."""
        B, H, W, D = x.shape
        x_conv = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        x_context = self.depthwise_conv(x_conv)
        gate = torch.sigmoid(self.residual_gate)
        x_mixed = (1 - gate) * x_conv + gate * x_context

        return x_mixed.permute(0, 2, 3, 1)  # (B, H, W, D)


# =============================================================================
# Routing (Task Prototype and Router)
# =============================================================================

class TaskPrototype:
    """Task Prototype for Distance-based Routing using Mahalanobis distance."""

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
            centered = features - self.mean.unsqueeze(0)
            self.covariance = (centered.T @ centered) / (features.shape[0] - 1)
            reg = 1e-5 * torch.eye(features.shape[1], device=features.device)
            self.covariance = self.covariance + reg
        else:
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
        """Compute precision matrix."""
        if self.covariance is not None:
            try:
                self.precision = torch.linalg.inv(self.covariance)
            except:
                self.precision = torch.linalg.pinv(self.covariance)

    def mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance from prototype."""
        if self.mean is None or self.precision is None:
            raise ValueError("Prototype not initialized")
        centered = features - self.mean.unsqueeze(0)
        return torch.sqrt(torch.sum(centered @ self.precision * centered, dim=1))


class PrototypeRouter:
    """Prototype-based Router for Task Selection using Mahalanobis distance."""

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.prototypes: Dict[int, TaskPrototype] = {}

    def add_prototype(self, task_id: int, prototype: TaskPrototype):
        self.prototypes[task_id] = prototype

    def route(self, features: torch.Tensor) -> torch.Tensor:
        """Route features to the best task based on Mahalanobis distance."""
        if not self.prototypes:
            return torch.zeros(features.shape[0], dtype=torch.long, device=features.device)

        all_distances = []
        task_ids = sorted(self.prototypes.keys())

        for task_id in task_ids:
            distances = self.prototypes[task_id].mahalanobis_distance(features)
            all_distances.append(distances)

        all_distances = torch.stack(all_distances, dim=0)
        min_indices = torch.argmin(all_distances, dim=0)
        return torch.tensor([task_ids[idx] for idx in min_indices], device=features.device)


# =============================================================================
# Main Model: MoLE-Flow Normalizing Flow
# =============================================================================

class MoLEFlowNF(nn.Module):
    """
    MoLE-Flow: Mixture of LoRA Experts for Normalizing Flow.

    Key Features:
    - Base NF weights shared across all tasks
    - Task-specific LoRA adapters for distribution shift
    - Task-specific input adapters for feature normalization
    - Deep Invertible Adapters (DIA) for nonlinear manifold adaptation
    - Feature statistics alignment for cross-task consistency

    Configuration follows Ultimate-Combo2 defaults.
    """

    def __init__(self, config: MoLEFlowConfig = None):
        super().__init__()

        if config is None:
            config = MoLEFlowConfig()

        self.config = config
        self.embed_dim = config.embed_dim
        self.coupling_layers = config.num_coupling_layers
        self.clamp_alpha = config.clamp_alpha
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.device = config.device

        # Flags from config
        self.use_lora = config.use_lora
        self.use_task_adapter = config.use_task_adapter
        self.use_task_bias = config.use_task_bias
        self.use_spatial_context = config.use_spatial_context
        self.use_scale_context = config.use_scale_context
        self.use_dia = config.use_dia
        self.dia_n_blocks = config.dia_n_blocks
        self.dia_hidden_ratio = config.dia_hidden_ratio

        # Track tasks
        self.num_tasks = 0
        self.current_task_id: Optional[int] = None

        # Build flow
        self.subnets: List[MoLEContextSubnet] = []
        self.flow = self._build_flow()

        # Reference statistics from Task 0
        self.reference_stats = FeatureStatistics(device=config.device)

        # Task-specific input adapters
        self.input_adapters = nn.ModuleDict()

        # DIA adapters per task
        self.dia_adapters = nn.ModuleDict()

        # Spatial Context Mixer (depthwise_residual)
        if self.use_spatial_context:
            self.spatial_mixer = SpatialContextMixer(
                channels=config.embed_dim,
                kernel_size=config.spatial_context_kernel
            )
        else:
            self.spatial_mixer = None

        self.to(config.device)

    def _build_flow(self) -> Ff.SequenceINN:
        """Build normalizing flow with MoLE coupling blocks."""

        def make_subnet(dims_in, dims_out):
            if self.use_scale_context:
                subnet = MoLEContextSubnet(
                    dims_in, dims_out,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    use_lora=self.use_lora,
                    use_task_bias=self.use_task_bias,
                    context_kernel=self.config.scale_context_kernel,
                    context_init_scale=self.config.scale_context_init_scale,
                    context_max_alpha=self.config.scale_context_max_alpha
                )
            else:
                # Fallback to simpler subnet (not implemented here for brevity)
                subnet = MoLEContextSubnet(dims_in, dims_out, rank=self.lora_rank,
                                           alpha=self.lora_alpha, use_lora=self.use_lora,
                                           use_task_bias=self.use_task_bias)
            self.subnets.append(subnet)
            return subnet

        coder = Ff.SequenceINN(self.embed_dim)
        print(f'MoLE-Flow: Embed={self.embed_dim}, LoRA_rank={self.lora_rank}, Layers={self.coupling_layers}')

        for _ in range(self.coupling_layers):
            coder.append(
                Fm.AllInOneBlock,
                subnet_constructor=make_subnet,
                affine_clamping=self.clamp_alpha,
                global_affine_type='SOFTPLUS',
                permute_soft=False
            )

        return coder

    def add_task(self, task_id: int):
        """Add LoRA adapters and input adapter for a new task."""
        task_key = str(task_id)

        if task_id == 0:
            # Task 0: Train base weights + LoRA adapter + InputAdapter
            for subnet in self.subnets:
                subnet.unfreeze_base()
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(task_id)

            if self.use_task_adapter:
                self.input_adapters[task_key] = WhiteningAdapter(
                    channels=self.embed_dim,
                    task_id=task_id,
                    reference_mean=None,
                    reference_std=None
                ).to(self.device)

            if self.use_dia:
                self.dia_adapters[task_key] = DeepInvertibleAdapter(
                    channels=self.embed_dim,
                    task_id=task_id,
                    n_blocks=self.dia_n_blocks,
                    hidden_ratio=self.dia_hidden_ratio,
                    clamp_alpha=self.clamp_alpha
                ).to(self.device)

            print(f"Task {task_id}: Base trainable + LoRA + WhiteningAdapter + DIA({self.dia_n_blocks})")

        else:
            # Task > 0: Freeze base, add LoRA adapters
            for subnet in self.subnets:
                subnet.freeze_base()
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(task_id)

            if self.use_task_adapter:
                if self.reference_stats.is_initialized:
                    ref_mean, ref_std = self.reference_stats.get_reference_params()
                else:
                    ref_mean, ref_std = None, None

                self.input_adapters[task_key] = WhiteningAdapter(
                    channels=self.embed_dim,
                    task_id=task_id,
                    reference_mean=ref_mean,
                    reference_std=ref_std
                ).to(self.device)

            if self.use_dia:
                self.dia_adapters[task_key] = DeepInvertibleAdapter(
                    channels=self.embed_dim,
                    task_id=task_id,
                    n_blocks=self.dia_n_blocks,
                    hidden_ratio=self.dia_hidden_ratio,
                    clamp_alpha=self.clamp_alpha
                ).to(self.device)

            print(f"Task {task_id}: Base frozen + LoRA + WhiteningAdapter + DIA({self.dia_n_blocks})")

        self.num_tasks = task_id + 1
        self.set_active_task(task_id)

    def update_reference_stats(self, features: torch.Tensor):
        """Update reference statistics (call during Task 0 training)."""
        self.reference_stats.update(features)

    def finalize_reference_stats(self):
        """Finalize reference statistics after Task 0 training."""
        self.reference_stats.finalize()

    def set_active_task(self, task_id: Optional[int]):
        """Set the currently active LoRA adapter."""
        self.current_task_id = task_id
        if task_id is None:
            for subnet in self.subnets:
                subnet.set_active_task(None)
        else:
            for subnet in self.subnets:
                subnet.set_active_task(task_id)

    def get_trainable_params(self, task_id: int) -> List[nn.Parameter]:
        """Get trainable parameters for a specific task."""
        params = []
        task_key = str(task_id)

        if task_id == 0:
            # Base + LoRA + context parameters
            for subnet in self.subnets:
                for layer in [subnet.s_layer1, subnet.s_layer2, subnet.t_layer1, subnet.t_layer2]:
                    params.extend(layer.base_linear.parameters())
                    if task_key in layer.lora_A:
                        params.append(layer.lora_A[task_key])
                        params.append(layer.lora_B[task_key])
                    if task_key in layer.task_biases:
                        params.append(layer.task_biases[task_key])
                params.extend(subnet.context_conv.parameters())
                if subnet.context_scale_param is not None:
                    params.append(subnet.context_scale_param)
        else:
            # Only LoRA + task biases
            for subnet in self.subnets:
                for layer in [subnet.s_layer1, subnet.s_layer2, subnet.t_layer1, subnet.t_layer2]:
                    if task_key in layer.lora_A:
                        params.append(layer.lora_A[task_key])
                        params.append(layer.lora_B[task_key])
                    if task_key in layer.task_biases:
                        params.append(layer.task_biases[task_key])

        # Input adapter
        if task_key in self.input_adapters:
            params.extend(self.input_adapters[task_key].parameters())

        # DIA
        if task_key in self.dia_adapters:
            params.extend(self.dia_adapters[task_key].parameters())

        # Spatial mixer (only for Task 0)
        if self.spatial_mixer is not None and task_id == 0:
            params.extend(self.spatial_mixer.parameters())

        return params

    def forward(self, patch_embeddings_with_pos: torch.Tensor,
                reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward or inverse transformation.

        Args:
            patch_embeddings_with_pos: (B, H, W, D) spatial patch embeddings
            reverse: If True, generate samples from latent

        Returns:
            z: (B, H, W, D) latent or reconstructed embeddings
            logdet_patch: (B, H, W) patch-wise log determinant of Jacobian
        """
        B, H, W, D = patch_embeddings_with_pos.shape
        x = patch_embeddings_with_pos

        # Apply task-specific input adapter
        if self.use_task_adapter and not reverse and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.input_adapters:
                x = self.input_adapters[task_key](x)

        # Apply spatial context mixing
        if not reverse and self.spatial_mixer is not None:
            x = self.spatial_mixer(x)

        # Flatten spatial dimensions
        x_flat = x.reshape(B * H * W, D)

        # Set spatial info for MoLEContextSubnet
        if self.use_scale_context:
            MoLEContextSubnet._spatial_info = (B, H, W)

        # Flow transformation
        if not reverse:
            z_flat, log_jac_det_flat = self.flow(x_flat)
        else:
            z_flat, log_jac_det_flat = self.flow(x_flat, rev=True)

        # Clear spatial info
        if self.use_scale_context:
            MoLEContextSubnet._spatial_info = None

        log_jac_det_flat = log_jac_det_flat.reshape(-1)
        logdet_patch = log_jac_det_flat.reshape(B, H, W)
        z = z_flat.reshape(B, H, W, D)

        # Apply DIA
        if self.use_dia and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.dia_adapters:
                z, dia_logdet = self.dia_adapters[task_key](z, reverse=reverse)
                logdet_patch = logdet_patch + dia_logdet

        return z, logdet_patch

    def log_prob(self, patch_embeddings_with_pos: torch.Tensor) -> torch.Tensor:
        """Compute log probability (image-level)."""
        B, H, W, D = patch_embeddings_with_pos.shape
        z, logdet_patch = self.forward(patch_embeddings_with_pos, reverse=False)

        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
        log_px_patch = log_pz_patch + logdet_patch
        log_px_image = log_px_patch.sum(dim=(1, 2))

        return log_px_image

    def log_prob_patch(self, patch_embeddings_with_pos: torch.Tensor) -> torch.Tensor:
        """Compute patch-wise log probability (for anomaly detection)."""
        B, H, W, D = patch_embeddings_with_pos.shape
        z, logdet_patch = self.forward(patch_embeddings_with_pos, reverse=False)

        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)
        log_px_patch = log_pz_patch + logdet_patch

        return log_px_patch


# =============================================================================
# Complete Integrated Model (Feature Extractor + NF + Router)
# =============================================================================

class MoLEFlowModel(nn.Module):
    """
    Complete MoLE-Flow Model integrating all components.

    Pipeline:
    Image -> CNN Extractor -> Positional Embedding -> MoLE-Flow NF -> Anomaly Score

    With Router for multi-task inference.
    """

    def __init__(self, config: MoLEFlowConfig = None):
        super().__init__()

        if config is None:
            config = MoLEFlowConfig()

        self.config = config
        self.device = config.device

        # Feature Extractor
        self.extractor = CNNPatchCoreExtractor(
            backbone_name=config.backbone_name,
            input_shape=(3, config.img_size, config.img_size),
            target_embed_dim=config.embed_dim,
            device=config.device
        )

        # Positional Embedding Generator
        self.pos_embed_gen = PositionalEmbeddingGenerator(device=config.device)

        # Normalizing Flow
        self.nf = MoLEFlowNF(config)

        # Router (Mahalanobis distance-based)
        self.router = PrototypeRouter(device=config.device)

        # Task classes
        self.task_classes: Dict[int, List[str]] = {}

    def add_task(self, task_id: int, task_classes: List[str]):
        """Add a new task."""
        self.task_classes[task_id] = task_classes
        self.nf.add_task(task_id)

    def extract_features(self, images: torch.Tensor) -> Tuple[torch.Tensor, tuple]:
        """Extract patch embeddings with positional encoding."""
        patch_embeddings, spatial_shape = self.extractor(images, return_spatial_shape=True)
        patch_embeddings_with_pos = self.pos_embed_gen(spatial_shape, patch_embeddings)
        return patch_embeddings_with_pos, spatial_shape

    def compute_anomaly_score(self, images: torch.Tensor,
                               task_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly scores for images.

        Args:
            images: (B, C, H, W) input images
            task_id: Task ID to use (if None, use router)

        Returns:
            image_scores: (B,) image-level anomaly scores
            pixel_scores: (B, H, W) pixel-level anomaly scores
        """
        # Extract features
        patch_embeddings_with_pos, spatial_shape = self.extract_features(images)

        # Route if task_id not provided
        if task_id is None and len(self.router.prototypes) > 0:
            image_features = self.extractor.get_image_level_features(images)
            predicted_tasks = self.router.route(image_features)

            # Process each sample with its predicted task
            all_image_scores = []
            all_pixel_scores = []

            for i, pred_task in enumerate(predicted_tasks):
                self.nf.set_active_task(pred_task.item())
                sample = patch_embeddings_with_pos[i:i+1]
                log_prob_patch = self.nf.log_prob_patch(sample)

                # Convert log prob to anomaly score (negative log prob)
                pixel_score = -log_prob_patch[0]
                image_score = pixel_score.mean()

                all_image_scores.append(image_score)
                all_pixel_scores.append(pixel_score)

            image_scores = torch.stack(all_image_scores)
            pixel_scores = torch.stack(all_pixel_scores)
        else:
            # Use specified task
            if task_id is not None:
                self.nf.set_active_task(task_id)

            log_prob_patch = self.nf.log_prob_patch(patch_embeddings_with_pos)
            pixel_scores = -log_prob_patch
            image_scores = pixel_scores.mean(dim=(1, 2))

        return image_scores, pixel_scores

    def forward(self, images: torch.Tensor,
                task_id: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - compute anomaly scores."""
        return self.compute_anomaly_score(images, task_id)


# =============================================================================
# Usage Example
# =============================================================================

def create_model(device: str = 'cuda') -> MoLEFlowModel:
    """Create a MoLE-Flow model with Ultimate-Combo2 configuration."""
    config = MoLEFlowConfig(device=device)
    model = MoLEFlowModel(config)
    return model


if __name__ == "__main__":
    # Example usage
    print("MoLE-Flow Integrated Model (Ultimate-Combo2 Configuration)")
    print("=" * 60)

    config = MoLEFlowConfig()
    print("\nDefault Configuration (Ultimate-Combo2):")
    print(f"  Backbone: {config.backbone_name}")
    print(f"  Embedding Dim: {config.embed_dim}")
    print(f"  LoRA Rank: {config.lora_rank}")
    print(f"  Coupling Layers: {config.num_coupling_layers}")
    print(f"  Spatial Context: {config.use_spatial_context} (depthwise_residual, k={config.spatial_context_kernel})")
    print(f"  Scale Context: {config.use_scale_context} (k={config.scale_context_kernel})")
    print(f"  DIA: {config.use_dia} ({config.dia_n_blocks} blocks)")
    print(f"  LogDet Reg: {config.lambda_logdet}")

    print("\n" + "=" * 60)
    print("To create the model:")
    print("  model = create_model(device='cuda')")
    print("  model.add_task(0, ['leather'])")
    print("  # Train Task 0...")
    print("  model.add_task(1, ['grid'])")
    print("  # Train Task 1...")
