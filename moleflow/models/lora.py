"""
LoRA (Low-Rank Adaptation) Module.

Implements LoRA-enhanced Linear Layer with Task-specific Bias.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LoRALinear(nn.Module):
    """
    LoRA-enhanced Linear Layer with Task-specific Bias.

    Output: h(x) = W_base @ x + scaling * (B @ A) @ x + (base_bias + task_bias)

    Key Design Principles:
    1. SMALL SCALING: Use smaller alpha/rank ratio for stable initial adaptation
    2. ZERO-INIT B: Ensures delta_W = 0 at start (pure identity mapping for LoRA part)
    3. XAVIER-INIT A: Better than Kaiming for symmetric distributions
    4. TASK BIAS: Handles distribution shift without modifying base model

    Ablation Support:
    - use_lora: If False, skip LoRA adaptation entirely
    - use_task_bias: If False, skip task-specific bias
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

        Respects ablation flags:
        - use_lora: If False, skip LoRA A/B matrices
        - use_task_bias: If False, skip task-specific bias
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

        Respects ablation flags:
        - use_lora: If False, skip LoRA contribution
        - use_task_bias: If False, use only base_bias
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


class MoLESubnet(nn.Module):
    """
    MoLE Subnet for NF Coupling Blocks.

    Architecture: Linear -> ReLU -> Linear
    With LoRA adapters on both linear layers.

    Ablation Support:
    - use_lora: If False, skip LoRA adaptation
    - use_task_bias: If False, skip task-specific bias
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


# =============================================================================
# V3 Improvements: Lightweight Multi-Scale Context
# =============================================================================

class LightweightMSContext(nn.Module):
    """
    Lightweight Multi-Scale Context Extractor (V3 Solution 2).

    Uses dilated depthwise convolutions to efficiently expand receptive field
    without parameter explosion.

    Key Design:
    - Shared depthwise conv weights across scales (parameter efficient)
    - Dilations [1, 2, 4] → effective RF: 3×3, 5×5, 9×9
    - Attention-weighted fusion of scales
    - Optional regional context for global awareness

    Compared to Inception-style:
    - Much fewer parameters (single kernel, multiple dilations)
    - Similar receptive field coverage
    - Learnable scale importance via attention
    """

    def __init__(self, channels: int,
                 dilations: tuple = (1, 2, 4),
                 use_regional: bool = True,
                 regional_grid: int = 4):
        super(LightweightMSContext, self).__init__()

        self.channels = channels
        self.dilations = dilations
        self.use_regional = use_regional
        self.regional_grid = regional_grid

        # Single depthwise conv kernel (shared across dilations)
        # Shape: (channels, 1, 3, 3) for depthwise
        self.dw_weight = nn.Parameter(torch.zeros(channels, 1, 3, 3))
        self.dw_bias = nn.Parameter(torch.zeros(channels))

        # Initialize to slight smoothing
        nn.init.constant_(self.dw_weight, 1.0 / 9.0)

        # Scale attention: learns which dilation to focus on
        n_scales = len(dilations) + (1 if use_regional else 0)
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, n_scales),
            nn.Softmax(dim=1)
        )

        # Regional projection (if enabled)
        if use_regional:
            self.regional_proj = nn.Linear(channels, channels)
            nn.init.zeros_(self.regional_proj.weight)
            nn.init.zeros_(self.regional_proj.bias)

        # Final fusion with residual gate
        self.fusion_gate = nn.Parameter(torch.tensor([0.3]))

    def _dilated_conv(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        """Apply depthwise conv with specified dilation."""
        B, C, H, W = x.shape
        pad = dilation  # Padding to maintain spatial size

        # Reflect padding for better edge handling
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='reflect')

        # Depthwise convolution with dilation
        out = F.conv2d(x_padded, self.dw_weight, self.dw_bias,
                       groups=C, dilation=dilation)

        # Crop to original size
        return out[:, :, :H, :W]

    def _regional_context(self, x: torch.Tensor) -> torch.Tensor:
        """Compute regional context (preserves some spatial info unlike GAP)."""
        B, C, H, W = x.shape

        # Pool to grid_size × grid_size
        rh = max(H // self.regional_grid, 1)
        rw = max(W // self.regional_grid, 1)

        # Reshape and average within regions
        x_regions = F.adaptive_avg_pool2d(x, (self.regional_grid, self.regional_grid))

        # Upsample back to original size (bilinear)
        x_global = F.interpolate(x_regions, size=(H, W), mode='bilinear', align_corners=False)

        # Project
        x_global = x_global.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_global = self.regional_proj(x_global)
        x_global = x_global.permute(0, 3, 1, 2)  # (B, C, H, W)

        return x_global

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract multi-scale context.

        Args:
            x: (B, H, W, D) input features

        Returns:
            Multi-scale context with same shape
        """
        B, H, W, D = x.shape
        identity = x

        # Convert to (B, D, H, W) for conv operations
        x_conv = x.permute(0, 3, 1, 2)

        # Compute multi-scale features
        ms_features = []
        for d in self.dilations:
            feat = self._dilated_conv(x_conv, d)
            ms_features.append(feat)

        # Add regional context if enabled
        if self.use_regional:
            regional = self._regional_context(x_conv)
            ms_features.append(regional)

        # Compute attention weights
        attn = self.scale_attention(x_conv)  # (B, n_scales)

        # Weighted sum of multi-scale features
        fused = torch.zeros_like(x_conv)
        for i, feat in enumerate(ms_features):
            weight = attn[:, i:i+1, None, None]  # (B, 1, 1, 1)
            fused = fused + weight * feat

        # Convert back to (B, H, W, D)
        fused = fused.permute(0, 2, 3, 1)

        # Residual connection with learnable gate
        gate = torch.sigmoid(self.fusion_gate)
        output = (1 - gate) * identity + gate * fused

        return output

    def get_attention_stats(self) -> dict:
        """Get attention statistics for logging."""
        with torch.no_grad():
            # Get the linear layer weights to understand scale preferences
            return {
                'fusion_gate': torch.sigmoid(self.fusion_gate).item()
            }


# =============================================================================
# V3 Improvement: Task-Conditioned Multi-Scale Context (Fundamental Solution)
# =============================================================================

class TaskConditionedMSContext(nn.Module):
    """
    Task-Conditioned Multi-Scale Context with LoRA adaptation.

    This is a fundamental redesign of LightweightMSContext that follows
    MoLE-Flow's core philosophy:
    - Shared base weights (frozen after Task 0)
    - Task-specific adaptation via LoRA

    Design Principles:
    1. Multi-scale dilated convs → SHARED (task-agnostic feature extraction)
    2. Scale attention → TASK-SPECIFIC (which scales matter depends on task)
    3. Regional projection → SHARED base + LoRA (large params need LoRA)
    4. Fusion gate → TASK-SPECIFIC (context blend ratio varies by task)

    Why this solves catastrophic forgetting:
    - LightweightMSContext has 601K shared params trained every task → forgetting
    - TaskConditionedMSContext freezes 598K shared params after Task 0
    - Only ~27K task-specific params are trained per task

    Parameter Breakdown:
    - Shared (frozen after Task 0):
        - dw_weight: 768×1×3×3 = 6,912
        - dw_bias: 768
        - regional_proj_base: 768×768 + 768 = 590,592
        - Total: ~598K
    - Per-Task:
        - scale_attention: ~3K
        - regional_lora_A: 768×16 = 12,288
        - regional_lora_B: 16×768 = 12,288
        - fusion_gate: 1
        - Total: ~27K
    """

    def __init__(self, channels: int,
                 dilations: tuple = (1, 2, 4),
                 use_regional: bool = True,
                 regional_grid: int = 4,
                 lora_rank: int = 16,
                 lora_alpha: float = 1.0):
        """
        Args:
            channels: Feature dimension (e.g., 768 for ViT-Base)
            dilations: Dilation rates for multi-scale convs
            use_regional: Whether to use regional context
            regional_grid: Grid size for regional pooling
            lora_rank: Rank for LoRA adaptation in regional_proj
            lora_alpha: Alpha scaling for LoRA
        """
        super(TaskConditionedMSContext, self).__init__()

        self.channels = channels
        self.dilations = dilations
        self.use_regional = use_regional
        self.regional_grid = regional_grid
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = lora_alpha / lora_rank

        n_scales = len(dilations) + (1 if use_regional else 0)
        self.n_scales = n_scales

        # =====================================================================
        # SHARED COMPONENTS (Frozen after Task 0)
        # These are task-agnostic feature extractors
        # =====================================================================

        # Multi-scale dilated convolution weights (shared)
        # Shape: (channels, 1, 3, 3) for depthwise conv
        self.dw_weight = nn.Parameter(torch.zeros(channels, 1, 3, 3))
        self.dw_bias = nn.Parameter(torch.zeros(channels))

        # Initialize to slight smoothing (averaging kernel)
        nn.init.constant_(self.dw_weight, 1.0 / 9.0)

        # Regional projection BASE weights (shared, frozen after Task 0)
        if use_regional:
            self.regional_proj_base = nn.Linear(channels, channels)
            # Initialize to zero for near-identity at start
            nn.init.zeros_(self.regional_proj_base.weight)
            nn.init.zeros_(self.regional_proj_base.bias)

        # =====================================================================
        # TASK-SPECIFIC COMPONENTS (trained per task)
        # These capture task-specific preferences
        # =====================================================================

        # Scale attention per task: learns which scales are important
        self.scale_attentions = nn.ModuleDict()

        # Regional LoRA adapters per task (for the large regional_proj)
        if use_regional:
            self.regional_lora_A = nn.ParameterDict()
            self.regional_lora_B = nn.ParameterDict()

        # Fusion gate per task: how much context to blend in
        self.fusion_gates = nn.ParameterDict()

        # Track tasks and current active task
        self.num_tasks = 0
        self.current_task_id: Optional[int] = None
        self.base_frozen = False

    def add_task(self, task_id: int):
        """
        Add task-specific components for a new task.

        This is called when training begins for a new task.
        Task 0: Also trains shared components
        Task > 0: Shared frozen, only task-specific components trained
        """
        task_key = str(task_id)
        device = self.dw_weight.device

        # 1. Task-specific scale attention
        # Learns which dilation scales are important for this task
        self.scale_attentions[task_key] = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.channels, self.n_scales),
            nn.Softmax(dim=1)
        ).to(device)

        # 2. Task-specific fusion gate
        # Controls how much context to blend with original features
        # Initialize to small value (0.1) so context influence grows gradually
        self.fusion_gates[task_key] = nn.Parameter(
            torch.tensor([0.1], device=device)
        )

        # 3. Task-specific LoRA for regional projection
        if self.use_regional:
            # A matrix: Xavier init for good gradient flow
            A = torch.empty(self.channels, self.lora_rank, device=device)
            nn.init.xavier_uniform_(A)
            self.regional_lora_A[task_key] = nn.Parameter(A)

            # B matrix: Zero init so LoRA starts as identity
            B = torch.zeros(self.lora_rank, self.channels, device=device)
            self.regional_lora_B[task_key] = nn.Parameter(B)

        # Freeze shared components after Task 0
        if task_id > 0 and not self.base_frozen:
            self.freeze_base()
            print(f"   🔒 [TaskConditionedMSContext] Shared components frozen after Task 0")

        self.num_tasks = task_id + 1
        self.current_task_id = task_id

        # Log task-specific param count
        task_params = sum(p.numel() for p in self.scale_attentions[task_key].parameters())
        task_params += self.fusion_gates[task_key].numel()
        if self.use_regional:
            task_params += self.regional_lora_A[task_key].numel()
            task_params += self.regional_lora_B[task_key].numel()
        print(f"   ✅ [TaskConditionedMSContext] Task {task_id} adapter added: {task_params:,} params")

    def freeze_base(self):
        """Freeze shared components after Task 0."""
        self.dw_weight.requires_grad = False
        self.dw_bias.requires_grad = False
        if self.use_regional:
            for param in self.regional_proj_base.parameters():
                param.requires_grad = False
        self.base_frozen = True

    def unfreeze_base(self):
        """Unfreeze for Task 0 training."""
        self.dw_weight.requires_grad = True
        self.dw_bias.requires_grad = True
        if self.use_regional:
            for param in self.regional_proj_base.parameters():
                param.requires_grad = True
        self.base_frozen = False

    def set_active_task(self, task_id: Optional[int]):
        """Set current active task for forward pass."""
        self.current_task_id = task_id

    def _dilated_conv(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        """Apply shared depthwise conv with specified dilation."""
        B, C, H, W = x.shape
        pad = dilation  # Padding to maintain spatial size

        # Reflect padding for better edge handling
        x_padded = F.pad(x, [pad, pad, pad, pad], mode='reflect')

        # Depthwise convolution with dilation
        out = F.conv2d(x_padded, self.dw_weight, self.dw_bias,
                       groups=C, dilation=dilation)

        # Crop to original size
        return out[:, :, :H, :W]

    def _regional_context(self, x: torch.Tensor, task_key: str) -> torch.Tensor:
        """
        Compute regional context with task-specific LoRA.

        Pipeline:
        1. Pool to regional grid (shared, no params)
        2. Upsample back to original size (shared, no params)
        3. Project via base + LoRA (base frozen, LoRA task-specific)
        """
        B, C, H, W = x.shape

        # Pool to grid (no params)
        x_regions = F.adaptive_avg_pool2d(x, (self.regional_grid, self.regional_grid))

        # Upsample back (no params)
        x_global = F.interpolate(x_regions, size=(H, W), mode='bilinear', align_corners=False)

        # Project with base + LoRA
        x_global = x_global.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Base projection (shared, frozen after Task 0)
        out = self.regional_proj_base(x_global)

        # Add LoRA contribution (task-specific)
        if task_key in self.regional_lora_A:
            lora_A = self.regional_lora_A[task_key]
            lora_B = self.regional_lora_B[task_key]
            # LoRA: out += scaling * (x @ A @ B)
            lora_out = x_global @ lora_A @ lora_B
            out = out + self.lora_scaling * lora_out

        out = out.permute(0, 3, 1, 2)  # (B, C, H, W)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with task-conditioned adaptation.

        Args:
            x: (B, H, W, D) input features

        Returns:
            (B, H, W, D) context-enhanced features
        """
        if self.current_task_id is None:
            return x  # No task set, return identity

        task_key = str(self.current_task_id)
        if task_key not in self.scale_attentions:
            return x  # Task not added yet

        B, H, W, D = x.shape
        identity = x

        # Convert to conv format: (B, H, W, D) -> (B, D, H, W)
        x_conv = x.permute(0, 3, 1, 2)

        # =====================================================================
        # Multi-scale feature extraction (SHARED convs)
        # =====================================================================
        ms_features = []
        for d in self.dilations:
            feat = self._dilated_conv(x_conv, d)
            ms_features.append(feat)

        # Regional context with task-specific LoRA
        if self.use_regional:
            regional = self._regional_context(x_conv, task_key)
            ms_features.append(regional)

        # =====================================================================
        # Task-specific scale attention
        # =====================================================================
        attn = self.scale_attentions[task_key](x_conv)  # (B, n_scales)

        # Weighted fusion of multi-scale features
        fused = torch.zeros_like(x_conv)
        for i, feat in enumerate(ms_features):
            weight = attn[:, i:i+1, None, None]  # (B, 1, 1, 1)
            fused = fused + weight * feat

        # Convert back: (B, D, H, W) -> (B, H, W, D)
        fused = fused.permute(0, 2, 3, 1)

        # =====================================================================
        # Task-specific fusion gate
        # =====================================================================
        gate = torch.sigmoid(self.fusion_gates[task_key])
        output = (1 - gate) * identity + gate * fused

        return output

    def get_trainable_params(self, task_id: int) -> list:
        """
        Get trainable parameters for a specific task.

        Task 0: Shared + task-specific params
        Task > 0: Only task-specific params
        """
        task_key = str(task_id)
        params = []

        if task_id == 0:
            # Task 0: Also train shared components
            params.append(self.dw_weight)
            params.append(self.dw_bias)
            if self.use_regional:
                params.extend(self.regional_proj_base.parameters())

        # All tasks: Train task-specific components
        if task_key in self.scale_attentions:
            params.extend(self.scale_attentions[task_key].parameters())
        if task_key in self.fusion_gates:
            params.append(self.fusion_gates[task_key])
        if self.use_regional and task_key in self.regional_lora_A:
            params.append(self.regional_lora_A[task_key])
            params.append(self.regional_lora_B[task_key])

        return params

    def get_attention_stats(self, task_id: Optional[int] = None) -> dict:
        """Get attention statistics for logging."""
        if task_id is None:
            task_id = self.current_task_id
        if task_id is None:
            return {}

        task_key = str(task_id)
        stats = {}

        with torch.no_grad():
            if task_key in self.fusion_gates:
                stats['fusion_gate'] = torch.sigmoid(self.fusion_gates[task_key]).item()

        return stats


# =============================================================================
# V3 Solution 1: Deep Invertible Adapter (DIA)
# =============================================================================

class DeepInvertibleAdapter(nn.Module):
    """
    Deep Invertible Adapter (DIA) - V3 Solution 1 (No Replay).

    Key Insight:
    Instead of linear LoRA (W + BA), we add a small task-specific Flow
    AFTER the base NF. This allows nonlinear manifold adaptation.

    Architecture:
    - Base NF: Frozen after Task 0 (extracts common features)
    - DIA: 1-2 lightweight coupling blocks per task (learns task-specific warping)

    The DIA is invertible, so it preserves the density estimation property.
    Each task has its own DIA, achieving complete parameter isolation.

    Mathematical Formulation:
    - Base: z_base = f_base(x)
    - DIA:  z_final = f_DIA_t(z_base)
    - log p(x) = log p(z_final) + log|det J_base| + log|det J_DIA|
    """

    def __init__(self,
                 channels: int,
                 task_id: int,
                 n_blocks: int = 2,
                 hidden_ratio: float = 0.5,
                 clamp_alpha: float = 1.9):
        """
        Args:
            channels: Feature dimension
            task_id: Task identifier
            n_blocks: Number of coupling blocks (1-2 recommended)
            hidden_ratio: Hidden dim = channels * hidden_ratio
            clamp_alpha: Clamping for affine coupling
        """
        super(DeepInvertibleAdapter, self).__init__()

        self.channels = channels
        self.task_id = task_id
        self.n_blocks = n_blocks
        self.clamp_alpha = clamp_alpha

        hidden_dim = int(channels * hidden_ratio)

        # Build mini-flow: sequence of coupling blocks
        self.coupling_blocks = nn.ModuleList()

        for i in range(n_blocks):
            # Alternate which half is transformed
            self.coupling_blocks.append(
                AffineCouplingBlock(
                    channels=channels,
                    hidden_dim=hidden_dim,
                    clamp_alpha=clamp_alpha,
                    reverse=(i % 2 == 1)
                )
            )

        # Initialize to near-identity for stable start
        self._initialize_near_identity()

    def _initialize_near_identity(self):
        """Initialize to near-identity transformation."""
        for block in self.coupling_blocks:
            # Initialize scale network output to 0 (exp(0) = 1)
            if hasattr(block.s_net, 'layers'):
                nn.init.zeros_(block.s_net.layers[-1].weight)
                nn.init.zeros_(block.s_net.layers[-1].bias)
            # Initialize translation network output to 0
            if hasattr(block.t_net, 'layers'):
                nn.init.zeros_(block.t_net.layers[-1].weight)
                nn.init.zeros_(block.t_net.layers[-1].bias)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        """
        Forward or inverse transformation.

        Args:
            x: (B, H, W, D) input
            reverse: If True, compute inverse transformation

        Returns:
            y: Transformed output
            log_det: Log determinant of Jacobian (B, H, W)
        """
        B, H, W, D = x.shape
        log_det = torch.zeros(B, H, W, device=x.device)

        if not reverse:
            # Forward: apply blocks in order
            for block in self.coupling_blocks:
                x, block_log_det = block(x, reverse=False)
                log_det = log_det + block_log_det
        else:
            # Inverse: apply blocks in reverse order
            for block in reversed(self.coupling_blocks):
                x, block_log_det = block(x, reverse=True)
                log_det = log_det + block_log_det

        return x, log_det


class AffineCouplingBlock(nn.Module):
    """
    Affine Coupling Block for DIA.

    Split input into two halves, transform one conditioned on the other:
    y1 = x1
    y2 = x2 * exp(s(x1)) + t(x1)

    This is invertible with known Jacobian determinant.
    """

    def __init__(self,
                 channels: int,
                 hidden_dim: int,
                 clamp_alpha: float = 1.9,
                 reverse: bool = False):
        super(AffineCouplingBlock, self).__init__()

        self.channels = channels
        self.clamp_alpha = clamp_alpha
        self.reverse_split = reverse

        # Split dimension
        self.split_dim = channels // 2

        # Scale network: x1 -> s
        self.s_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)

        # Translation network: x1 -> t
        self.t_net = SimpleSubnet(self.split_dim, self.split_dim, hidden_dim)

    def forward(self, x: torch.Tensor, reverse: bool = False):
        """
        Forward or inverse coupling transformation.

        Args:
            x: (B, H, W, D) input
            reverse: If True, compute inverse

        Returns:
            y: Output
            log_det: (B, H, W) log determinant
        """
        B, H, W, D = x.shape

        # Split
        if self.reverse_split:
            x2, x1 = x[..., :self.split_dim], x[..., self.split_dim:]
        else:
            x1, x2 = x[..., :self.split_dim], x[..., self.split_dim:]

        # Flatten for network
        x1_flat = x1.reshape(-1, self.split_dim)

        # Compute scale and translation
        s = self.s_net(x1_flat).reshape(B, H, W, self.split_dim)
        t = self.t_net(x1_flat).reshape(B, H, W, self.split_dim)

        # Clamp scale for numerical stability
        s = self.clamp_alpha * torch.tanh(s / self.clamp_alpha)

        if not reverse:
            # Forward: y2 = x2 * exp(s) + t
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)  # (B, H, W)
        else:
            # Inverse: x2 = (y2 - t) * exp(-s)
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)  # (B, H, W)

        # Reconstruct
        if self.reverse_split:
            y = torch.cat([y2, x1], dim=-1)
        else:
            y = torch.cat([x1, y2], dim=-1)

        return y, log_det


class SimpleSubnet(nn.Module):
    """Simple MLP subnet for DIA coupling blocks."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super(SimpleSubnet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

        # Initialize output layer to zero for identity start
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x)


class MoLEContextSubnet(nn.Module):
    """
    MoLE Subnet with Context-Aware Scale.

    Key Innovation:
    - s-network: concat(x, local_context) → anomaly-sensitive scale
    - t-network: x only → density-preserving shift

    This design ensures:
    - scale(s) can detect "patches different from neighbors" (anomaly-sensitive)
    - shift(t) preserves density estimation without noise from context

    Architecture:
    - 3×3 depthwise conv for local context extraction
    - Global alpha: alpha = alpha_max * sigmoid(alpha_param)
    - s_net: Linear(2D→H) → ReLU → Linear(H→D/2)
    - t_net: Linear(D→H) → ReLU → Linear(H→D/2)
    """

    # Class-level storage for spatial info (set by parent NF before forward)
    _spatial_info = None  # (batch_size, H, W)

    def __init__(self, dims_in: int, dims_out: int, rank: int = 4, alpha: float = 1.0,
                 use_lora: bool = True, use_task_bias: bool = True,
                 context_kernel: int = 3, context_init_scale: float = 0.1,
                 context_max_alpha: float = 0.2, **kwargs):
        super(MoLEContextSubnet, self).__init__()

        hidden_dim = 2 * dims_in
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.use_lora = use_lora
        self.use_task_bias = use_task_bias
        self.context_max_alpha = context_max_alpha

        # =====================================================================
        # Context extraction (3×3 depthwise conv)
        # =====================================================================
        self.context_conv = nn.Conv2d(
            dims_in, dims_in,
            kernel_size=context_kernel,
            padding=context_kernel // 2,
            groups=dims_in,  # depthwise: each channel independently
            bias=True
        )
        # Initialize to near-zero (minimal initial context influence)
        nn.init.zeros_(self.context_conv.weight)
        nn.init.zeros_(self.context_conv.bias)

        # =====================================================================
        # Global alpha with sigmoid upper bound
        # =====================================================================
        # alpha = alpha_max * sigmoid(alpha_param)
        # Initialize alpha_param so that sigmoid(alpha_param) * alpha_max = init_scale
        p = min(max(context_init_scale / context_max_alpha, 0.01), 0.99)
        init_param = torch.log(torch.tensor([p / (1 - p)]))  # Inverse sigmoid
        self.context_scale_param = nn.Parameter(init_param)

        # =====================================================================
        # s-network: context-aware (anomaly-sensitive)
        # Input: concat(x, context) = 2 * dims_in
        # Output: dims_out // 2 (first half of coupling output)
        # =====================================================================
        self.s_layer1 = LoRALinear(dims_in * 2, hidden_dim, rank=rank, alpha=alpha,
                                   use_lora=use_lora, use_task_bias=use_task_bias)
        self.s_layer2 = LoRALinear(hidden_dim, dims_out // 2, rank=rank, alpha=alpha,
                                   use_lora=use_lora, use_task_bias=use_task_bias)

        # =====================================================================
        # t-network: context-free (density-preserving)
        # Input: x = dims_in
        # Output: dims_out // 2 (second half of coupling output)
        # =====================================================================
        self.t_layer1 = LoRALinear(dims_in, hidden_dim, rank=rank, alpha=alpha,
                                   use_lora=use_lora, use_task_bias=use_task_bias)
        self.t_layer2 = LoRALinear(hidden_dim, dims_out // 2, rank=rank, alpha=alpha,
                                   use_lora=use_lora, use_task_bias=use_task_bias)

        self.relu = nn.ReLU()

    def add_task_adapter(self, task_id: int):
        """Add LoRA adapters for a new task to all 4 layers."""
        self.s_layer1.add_task_adapter(task_id)
        self.s_layer2.add_task_adapter(task_id)
        self.t_layer1.add_task_adapter(task_id)
        self.t_layer2.add_task_adapter(task_id)

    def freeze_base(self):
        """Freeze base weights of all layers."""
        self.s_layer1.freeze_base()
        self.s_layer2.freeze_base()
        self.t_layer1.freeze_base()
        self.t_layer2.freeze_base()

    def unfreeze_base(self):
        """Unfreeze base weights of all layers."""
        self.s_layer1.unfreeze_base()
        self.s_layer2.unfreeze_base()
        self.t_layer1.unfreeze_base()
        self.t_layer2.unfreeze_base()

    def set_active_task(self, task_id: Optional[int]):
        """Set active LoRA adapter for all layers."""
        self.s_layer1.set_active_task(task_id)
        self.s_layer2.set_active_task(task_id)
        self.t_layer1.set_active_task(task_id)
        self.t_layer2.set_active_task(task_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with context-aware scale.

        Args:
            x: (BHW, D) flattened patch features

        Returns:
            output: (BHW, dims_out) where first half is s, second half is t
        """
        BHW, D = x.shape

        # Get spatial info (set by parent NF model)
        if MoLEContextSubnet._spatial_info is not None:
            B, H, W = MoLEContextSubnet._spatial_info
        else:
            # Fallback: assume square spatial layout
            B = 1
            HW = BHW
            H = W = int(HW ** 0.5)
            if H * W != HW:
                # Non-square, find factors
                for h in range(int(HW ** 0.5), 0, -1):
                    if HW % h == 0:
                        H = h
                        W = HW // h
                        break
            B = BHW // (H * W)

        # =================================================================
        # Extract local context via 3×3 depthwise conv
        # =================================================================
        x_spatial = x.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        ctx = self.context_conv(x_spatial)  # (B, D, H, W)
        ctx = ctx.permute(0, 2, 3, 1).reshape(BHW, D)  # (BHW, D)

        # =================================================================
        # Apply global alpha scaling
        # =================================================================
        # alpha = alpha_max * sigmoid(alpha_param)
        alpha = self.context_max_alpha * torch.sigmoid(self.context_scale_param)
        ctx = alpha * ctx

        # =================================================================
        # s-network: context-aware (anomaly-sensitive)
        # =================================================================
        s_input = torch.cat([x, ctx], dim=-1)  # (BHW, 2D)
        s = self.s_layer2(self.relu(self.s_layer1(s_input)))  # (BHW, dims_out//2)

        # =================================================================
        # t-network: context-free (density-preserving)
        # =================================================================
        t = self.t_layer2(self.relu(self.t_layer1(x)))  # (BHW, dims_out//2)

        # Concatenate [s, t] for FrEIA coupling block
        return torch.cat([s, t], dim=-1)  # (BHW, dims_out)

    # =========================================================================
    # Logging/Debugging utilities
    # =========================================================================

    def get_context_alpha(self) -> float:
        """Get current global context alpha value."""
        if self.context_scale_param is not None:
            with torch.no_grad():
                return (
                    self.context_max_alpha
                    * torch.sigmoid(self.context_scale_param)
                ).item()
        return None
