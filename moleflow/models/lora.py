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
