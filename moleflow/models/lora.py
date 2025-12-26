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


class MoLEContextSubnet(nn.Module):
    """
    MoLE Subnet with Context-Aware Scale (Baseline 1.5 / 2.0).

    Key Innovation:
    - s-network: concat(x, local_context) → anomaly-sensitive scale
    - t-network: x only → density-preserving shift

    This design ensures:
    - scale(s) can detect "patches different from neighbors" (anomaly-sensitive)
    - shift(t) preserves density estimation without noise from context

    Architecture:
    - 3×3 depthwise conv for local context extraction
    - Context gating (two modes):
      1. Global alpha (legacy): alpha = alpha_max * sigmoid(alpha_param)
         - Single learnable scalar, same for all patches
      2. Patch-wise gate (v2.0): gate = sigmoid(MLP([x, ctx]))
         - Per-patch decision: "should I use context here?"
         - Normal patches → gate ≈ 0 (learn to ignore context)
         - Anomaly-like patches → gate ≈ 1 (context matters)
    - s_net: Linear(2D→H) → ReLU → Linear(H→D/2)
    - t_net: Linear(D→H) → ReLU → Linear(H→D/2)
    """

    # Class-level storage for spatial info (set by parent NF before forward)
    _spatial_info = None  # (batch_size, H, W)

    def __init__(self, dims_in: int, dims_out: int, rank: int = 4, alpha: float = 1.0,
                 use_lora: bool = True, use_task_bias: bool = True,
                 context_kernel: int = 3, context_init_scale: float = 0.1,
                 context_max_alpha: float = 0.2, use_context_gate: bool = False,
                 context_gate_hidden: int = 64):
        super(MoLEContextSubnet, self).__init__()

        hidden_dim = 2 * dims_in
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.use_lora = use_lora
        self.use_task_bias = use_task_bias
        self.context_max_alpha = context_max_alpha
        self.use_context_gate = use_context_gate

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
        # Context gating: Global alpha vs Patch-wise gate
        # =====================================================================
        if use_context_gate:
            # Patch-wise gate: sigmoid(MLP([x, ctx])) → (BHW, 1)
            # This learns "when to use context" per-patch
            # - Normal patches: gate → 0 (ignore context)
            # - Anomaly-like patches: gate → 1 (use context)
            self.context_gate_net = nn.Sequential(
                nn.Linear(dims_in * 2, context_gate_hidden),
                nn.ReLU(),
                nn.Linear(context_gate_hidden, 1)
                # No sigmoid here - we'll apply it in forward for numerical stability
            )
            # Initialize to output ~0 (gate starts near 0.5)
            # This means context is initially used moderately
            nn.init.zeros_(self.context_gate_net[0].weight)
            nn.init.zeros_(self.context_gate_net[0].bias)
            nn.init.zeros_(self.context_gate_net[2].weight)
            nn.init.zeros_(self.context_gate_net[2].bias)

            # No global alpha param needed
            self.context_scale_param = None
        else:
            # Legacy: Global alpha with sigmoid upper bound
            # alpha = alpha_max * sigmoid(alpha_param)
            # Initialize alpha_param so that sigmoid(alpha_param) * alpha_max = init_scale
            p = min(max(context_init_scale / context_max_alpha, 0.01), 0.99)
            init_param = torch.log(torch.tensor([p / (1 - p)]))  # Inverse sigmoid
            self.context_scale_param = nn.Parameter(init_param)
            self.context_gate_net = None

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
        # Apply context gating (patch-wise or global)
        # =================================================================
        if self.use_context_gate and self.context_gate_net is not None:
            # Patch-wise gate: sigmoid(MLP([x, ctx])) → (BHW, 1)
            # Each patch decides independently how much context to use
            gate_input = torch.cat([x, ctx], dim=-1)  # (BHW, 2D)
            gate_logit = self.context_gate_net(gate_input)  # (BHW, 1)
            gate = torch.sigmoid(gate_logit)  # (BHW, 1), in [0, 1]

            # Store gate for logging/debugging (detached)
            self._last_gate = gate.detach()

            # Scale context by patch-wise gate
            ctx = gate * ctx  # (BHW, D)
        else:
            # Legacy: Global alpha with sigmoid upper bound
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
        """Get current global context alpha value (legacy mode only)."""
        if self.context_scale_param is not None:
            with torch.no_grad():
                return (
                    self.context_max_alpha
                    * torch.sigmoid(self.context_scale_param)
                ).item()
        return None

    def get_last_gate_stats(self) -> dict:
        """
        Get statistics of the last gate values (patch-wise gate mode only).

        Returns:
            dict with 'mean', 'std', 'min', 'max' of gate values
            or None if not using patch-wise gate
        """
        if hasattr(self, '_last_gate') and self._last_gate is not None:
            gate = self._last_gate
            return {
                'mean': gate.mean().item(),
                'std': gate.std().item(),
                'min': gate.min().item(),
                'max': gate.max().item()
            }
        return None
