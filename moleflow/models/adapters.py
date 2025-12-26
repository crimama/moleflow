"""
Feature Statistics and Task Input Adapters.

Provides distribution alignment for cross-task consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


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
        """
        Compute final std and mark as initialized.
        """
        if self._M2 is not None and self.n_samples > 1:
            variance = self._M2 / (self.n_samples - 1)
            self.std = torch.sqrt(variance + 1e-6)
        else:
            self.std = torch.ones_like(self.mean)

        self.is_initialized = True
        print(f"📊 Feature Statistics Finalized:")
        print(f"   - n_samples: {self.n_samples}")
        print(f"   - mean_norm: {self.mean.norm():.4f}")
        print(f"   - std_mean: {self.std.mean():.4f}")
        print(f"   - std_min: {self.std.min():.4f}, std_max: {self.std.max():.4f}")

    def get_reference_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get reference mean and std for TaskInputAdapter initialization.

        Returns:
            (mean, std) tensors
        """
        if not self.is_initialized:
            raise ValueError("Statistics not finalized. Call finalize() first.")
        return self.mean.clone(), self.std.clone()


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


class SimpleTaskAdapter(nn.Module):
    """
    Simpler adapter using stored Task 0 statistics for alignment.

    Transforms new task features to have similar distribution to Task 0.
    """

    def __init__(self, channels: int, reference_stats: FeatureStatistics):
        super(SimpleTaskAdapter, self).__init__()

        self.channels = channels
        self.reference_stats = reference_stats

        # Learnable transformation to match reference distribution
        self.target_mean = nn.Parameter(torch.zeros(channels))
        self.target_std = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize to zero mean/unit std, then transform to target distribution.

        Args:
            x: (B, H, W, D) input features

        Returns:
            Aligned features
        """
        B, H, W, D = x.shape

        # Compute current batch statistics
        x_flat = x.reshape(-1, D)
        batch_mean = x_flat.mean(dim=0, keepdim=True)
        batch_std = x_flat.std(dim=0, keepdim=True) + 1e-6

        # Normalize to zero mean, unit std
        x_normalized = (x_flat - batch_mean) / batch_std

        # Transform to target distribution (learned)
        # Initialize target to reference stats if available
        if self.reference_stats.is_initialized:
            target_mean = self.reference_stats.mean + self.target_mean
            target_std = self.reference_stats.std * torch.exp(self.target_std - 1)
        else:
            target_mean = self.target_mean
            target_std = torch.exp(self.target_std)

        x_aligned = x_normalized * target_std + target_mean

        return x_aligned.reshape(B, H, W, D)


class RobustTaskInputAdapter(nn.Module):
    """
    Robust Task Input Adapter with anomaly-aware gating.

    Key Design (Option A):
    - Normalize based on normal patch statistics only
    - Reduce normalization strength for anomaly-like patches using gating
    - x' = (1-g)*x + g*Adapter(x)  where g ∈ [0,1]
    - Anomalies have higher x norms → lower g → less adapter effect → preserved anomaly signal

    Gate Types:
    - "norm": Use feature L2 norm (higher norm = more anomalous)
    - "mahal": Use Mahalanobis distance from reference distribution
    - "learned": Use learned MLP gate
    """

    def __init__(self, channels: int, reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None, gate_type: str = "norm"):
        super(RobustTaskInputAdapter, self).__init__()

        self.channels = channels
        self.eps = 1e-6
        self.gate_type = gate_type

        # Store reference statistics (from Task 0)
        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
            self.has_reference = True
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))
            self.has_reference = False

        # FiLM parameters for normalization
        self.film_gamma = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.film_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

        # Layer Norm for the adapter path
        self.layer_norm = nn.LayerNorm(channels)

        # MLP for feature transformation
        hidden_dim = max(channels // 2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

        # Gate parameters
        if gate_type == "learned":
            # Learned gate network
            self.gate_net = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.GELU(),
                nn.Linear(channels // 4, 1),
                nn.Sigmoid()
            )
        else:
            # Temperature and bias for norm/mahal-based gate
            self.gate_temperature = nn.Parameter(torch.tensor([1.0]))
            self.gate_bias = nn.Parameter(torch.tensor([0.0]))

        # Running statistics for gate normalization (norm-based)
        self.register_buffer('running_norm_mean', torch.tensor([0.0]))
        self.register_buffer('running_norm_std', torch.tensor([1.0]))
        self.register_buffer('norm_count', torch.tensor([0]))

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gate value g ∈ [0,1] for each patch.

        High g (close to 1) = normal patch → apply adapter fully
        Low g (close to 0) = anomalous patch → preserve original features

        Args:
            x: (B, H, W, D) input features

        Returns:
            gate: (B, H, W, 1) gate values
        """
        B, H, W, D = x.shape

        if self.gate_type == "learned":
            # Learned gate
            x_flat = x.reshape(-1, D)
            gate = self.gate_net(x_flat)  # (BHW, 1)
            return gate.reshape(B, H, W, 1)

        elif self.gate_type == "mahal":
            # Mahalanobis distance-based gate
            x_flat = x.reshape(-1, D)  # (BHW, D)
            diff = x_flat - self.reference_mean.unsqueeze(0)
            mahal_sq = (diff / (self.reference_std.unsqueeze(0) + self.eps)) ** 2
            distance = mahal_sq.sum(dim=-1)  # (BHW,)

            # Higher distance = more anomalous = lower gate
            # Use sigmoid with learnable temperature and bias
            gate = torch.sigmoid(self.gate_bias - self.gate_temperature * distance / D)
            return gate.reshape(B, H, W, 1)

        else:  # "norm"
            # L2 norm-based gate
            x_flat = x.reshape(-1, D)
            norms = x_flat.norm(dim=-1)  # (BHW,)

            # Update running statistics during training
            if self.training and self.norm_count < 10000:
                with torch.no_grad():
                    batch_mean = norms.mean()
                    batch_std = norms.std() + self.eps
                    momentum = 0.1
                    if self.norm_count == 0:
                        self.running_norm_mean.copy_(batch_mean)
                        self.running_norm_std.copy_(batch_std)
                    else:
                        self.running_norm_mean.mul_(1 - momentum).add_(batch_mean * momentum)
                        self.running_norm_std.mul_(1 - momentum).add_(batch_std * momentum)
                    self.norm_count += 1

            # Normalize norms
            norm_normalized = (norms - self.running_norm_mean) / (self.running_norm_std + self.eps)

            # Higher norm = more anomalous = lower gate
            gate = torch.sigmoid(self.gate_bias - self.gate_temperature * norm_normalized)
            return gate.reshape(B, H, W, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Robust feature modulation with anomaly-aware gating.

        Flow: x -> compute_gate -> x' = (1-g)*x + g*Adapter(x)

        Args:
            x: (B, H, W, D) input features

        Returns:
            Transformed features with anomaly signal preserved
        """
        B, H, W, D = x.shape
        identity = x

        # 1. Compute gate (how much to apply adapter)
        gate = self._compute_gate(x)  # (B, H, W, 1)

        # 2. Apply adapter transformation
        x_flat = x.reshape(-1, D)
        x_normed = self.layer_norm(x_flat)
        x_adapted = x_normed.reshape(B, H, W, D)

        # FiLM modulation
        x_adapted = self.film_gamma * x_adapted + self.film_beta

        # MLP
        x_flat = x_adapted.reshape(-1, D)
        mlp_out = self.mlp(x_flat).reshape(B, H, W, D)
        x_adapted = x_adapted + 0.5 * mlp_out

        # 3. Gated combination: x' = (1-g)*x + g*Adapter(x)
        # High gate (normal) → adapter applied
        # Low gate (anomaly) → original preserved
        x_out = (1 - gate) * identity + gate * x_adapted

        return x_out


class SoftLNTaskInputAdapter(nn.Module):
    """
    Task Input Adapter with Soft/Optional LayerNorm.

    Key Design (Baseline 1.4):
    - Task 0: LayerNorm ON (weak) - sets the base density anchor
    - Task > 0: LayerNorm OFF (blend=0, fixed) - let Flow's scale(s) + LoRA handle distribution

    This prevents LN from stealing Flow's role:
    - Flow already has mean shift (t(x)) and scale (s(x))
    - LN before Flow makes scale(s) meaningless → log_det noise
    - Task 0 LN provides the reference point, then Flow takes over

    Parameters:
    - task_id: 0 for base task (LN on), >0 for subsequent tasks (LN off)
    - soft_ln_init_scale: Initial blend for Task 0 (default 0.01 = very weak)
    """

    def __init__(self, channels: int, reference_mean: torch.Tensor = None,
                 reference_std: torch.Tensor = None, task_id: int = 0,
                 soft_ln_init_scale: float = 0.01):
        super(SoftLNTaskInputAdapter, self).__init__()

        self.channels = channels
        self.eps = 1e-6
        self.task_id = task_id
        self.soft_ln_init_scale = soft_ln_init_scale

        # Task 0: LN on (learnable blend starting at soft_ln_init_scale)
        # Task > 0: LN off (blend fixed at 0)
        self.use_ln = (task_id == 0)

        # Store reference statistics (from Task 0)
        if reference_mean is not None:
            self.register_buffer('reference_mean', reference_mean.clone())
            self.register_buffer('reference_std', reference_std.clone())
            self.has_reference = True
        else:
            self.register_buffer('reference_mean', torch.zeros(channels))
            self.register_buffer('reference_std', torch.ones(channels))
            self.has_reference = False

        # FiLM parameters
        self.film_gamma = nn.Parameter(torch.ones(1, 1, 1, channels))
        self.film_beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

        # MLP
        hidden_dim = max(channels // 2, 128)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels)
        )
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

        self.residual_gate = nn.Parameter(torch.tensor([0.5]))

        # LayerNorm (only used for Task 0)
        self.layer_norm = nn.LayerNorm(channels)

        # Soft LN blend factor
        # Task 0: learnable, starts at soft_ln_init_scale
        # Task > 0: fixed at 0 (LN completely off)
        if self.use_ln:
            self.ln_blend = nn.Parameter(torch.tensor([soft_ln_init_scale]))
        else:
            # Fixed at 0 for Task > 0
            self.register_buffer('ln_blend', torch.tensor([0.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FiLM-style feature modulation with soft LayerNorm.

        Task 0: Soft LN applied (weak, learnable blend)
        Task > 0: LN skipped (blend=0), only FiLM + MLP applied

        Args:
            x: (B, H, W, D) input features

        Returns:
            Transformed features
        """
        B, H, W, D = x.shape
        identity = x

        # 1. Soft LayerNorm: blend between normalized and original
        # Task 0: learnable blend (starts at soft_ln_init_scale)
        # Task > 0: blend=0 (LN effectively off, but still computes for stability)
        x_flat = x.reshape(-1, D)
        x_normed = self.layer_norm(x_flat).reshape(B, H, W, D)

        # Soft blending: scale is clamped to [0, 1]
        blend = torch.clamp(self.ln_blend, 0.0, 1.0)
        x = blend * x_normed + (1 - blend) * x

        # 2. FiLM modulation
        x = self.film_gamma * x + self.film_beta

        # 3. MLP residual with learnable gate
        gate = torch.sigmoid(self.residual_gate)
        x_flat = x.reshape(-1, D)
        mlp_out = self.mlp(x_flat).reshape(B, H, W, D)
        x = x + gate * mlp_out

        # 4. Blend with identity for stability (for self-adapting Task 0)
        if not self.has_reference:
            x = 0.9 * identity + 0.1 * x

        return x


def create_task_adapter(adapter_mode: str, channels: int,
                        reference_mean: torch.Tensor = None,
                        reference_std: torch.Tensor = None,
                        task_id: int = 0,
                        soft_ln_init_scale: float = 0.01,
                        robust_gate_type: str = "norm") -> nn.Module:
    """
    Factory function to create the appropriate adapter based on mode.

    Args:
        adapter_mode: "standard", "robust", "soft_ln", "no_ln_after_task0"
        channels: Feature dimension
        reference_mean: Reference mean from Task 0
        reference_std: Reference std from Task 0
        task_id: Current task ID
        soft_ln_init_scale: Initial scale for soft LN
        robust_gate_type: Gate type for robust adapter

    Returns:
        Appropriate adapter module
    """
    if adapter_mode == "robust":
        return RobustTaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            gate_type=robust_gate_type
        )
    elif adapter_mode == "soft_ln":
        # SoftLN: Task 0 has learnable LN blend, Task > 0 has blend=0 (LN off)
        return SoftLNTaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            task_id=task_id,  # Critical: controls LN behavior
            soft_ln_init_scale=soft_ln_init_scale
        )
    elif adapter_mode == "no_ln_after_task0":
        # Task 0: use LN, Task > 0: no LN (hard switch)
        use_norm = (task_id == 0)
        return TaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            use_norm=use_norm
        )
    else:  # "standard"
        return TaskInputAdapter(
            channels=channels,
            reference_mean=reference_mean,
            reference_std=reference_std,
            use_norm=True
        )


class SpatialContextMixer(nn.Module):
    """
    Spatial Context Mixer for NF input preprocessing.

    Key Design (Baseline 1.5):
    - Current problem: scale(s) only sees individual patch features
    - Solution: Add shallow spatial mixing so scale(s) can see local context

    This allows the Flow to detect:
    - "Is this patch abnormal COMPARED TO neighbors?" (local contrast)
    - Not just "Is this patch abnormal in isolation?"

    Implementation:
    - 3x3 depthwise conv (channel-wise, preserves spatial structure)
    - Optional: local statistics (mean, std) concatenation
    - Residual connection for stability

    Modes:
    - "depthwise": 3x3 depthwise conv only (preserves dim)
    - "depthwise_residual": depthwise conv + residual
    - "local_stats": concat local mean/std (doubles dim, needs projection)
    - "full": depthwise + local_stats + projection back to original dim
    """

    def __init__(self, channels: int, mode: str = "depthwise_residual",
                 kernel_size: int = 3, learnable: bool = True):
        super(SpatialContextMixer, self).__init__()

        self.channels = channels
        self.mode = mode
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 3x3 Depthwise conv: each channel has its own 3x3 kernel
        # groups=channels means each channel is convolved independently
        self.depthwise_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            groups=channels,  # Depthwise: each channel separately
            bias=True
        )

        # Initialize to identity-like (center = 1, others = small)
        if not learnable:
            # Fixed averaging kernel
            with torch.no_grad():
                self.depthwise_conv.weight.fill_(1.0 / (kernel_size * kernel_size))
                self.depthwise_conv.bias.zero_()
            for param in self.depthwise_conv.parameters():
                param.requires_grad = False
        else:
            # Learnable, initialized to slight smoothing
            nn.init.constant_(self.depthwise_conv.weight, 1.0 / (kernel_size * kernel_size))
            nn.init.zeros_(self.depthwise_conv.bias)

        # Residual gate (learnable blend between original and context)
        self.residual_gate = nn.Parameter(torch.tensor([0.5]))

        # For "local_stats" or "full" mode: projection to preserve dimensions
        if mode in ["local_stats", "full"]:
            # Local stats doubles the channels (concat mean, std)
            # Project back to original dimension
            self.stats_proj = nn.Sequential(
                nn.Linear(channels * 3 if mode == "full" else channels * 2, channels),
                nn.GELU(),
                nn.Linear(channels, channels)
            )
            nn.init.zeros_(self.stats_proj[-1].weight)
            nn.init.zeros_(self.stats_proj[-1].bias)

    def _compute_local_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute local mean and std using average pooling.

        Args:
            x: (B, C, H, W) input tensor

        Returns:
            local_mean: (B, C, H, W)
            local_std: (B, C, H, W)
        """
        # Use average pooling to get local mean
        local_mean = F.avg_pool2d(
            x, kernel_size=self.kernel_size, stride=1, padding=self.padding
        )

        # Compute local variance: E[X^2] - E[X]^2
        local_sq_mean = F.avg_pool2d(
            x ** 2, kernel_size=self.kernel_size, stride=1, padding=self.padding
        )
        local_var = local_sq_mean - local_mean ** 2
        local_std = torch.sqrt(local_var.clamp(min=1e-6))

        return local_mean, local_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial context mixing.

        Args:
            x: (B, H, W, D) input features (note: H, W, D order from ViT)

        Returns:
            Mixed features with same shape (B, H, W, D)
        """
        B, H, W, D = x.shape
        identity = x

        # Convert to (B, D, H, W) for conv2d
        x_conv = x.permute(0, 3, 1, 2)  # (B, D, H, W)

        if self.mode == "depthwise":
            # Simple depthwise conv
            x_mixed = self.depthwise_conv(x_conv)
            x_out = x_mixed.permute(0, 2, 3, 1)  # Back to (B, H, W, D)

        elif self.mode == "depthwise_residual":
            # Depthwise conv with residual
            x_context = self.depthwise_conv(x_conv)
            gate = torch.sigmoid(self.residual_gate)
            x_mixed = (1 - gate) * x_conv + gate * x_context
            x_out = x_mixed.permute(0, 2, 3, 1)  # Back to (B, H, W, D)

        elif self.mode == "local_stats":
            # Concat local mean and std, then project back
            local_mean, local_std = self._compute_local_stats(x_conv)
            # Concat: (B, 2D, H, W)
            x_concat = torch.cat([x_conv, local_std], dim=1)
            # Permute to (B, H, W, 2D) for linear
            x_concat = x_concat.permute(0, 2, 3, 1)
            # Project back to D
            x_proj = self.stats_proj(x_concat)
            # Residual
            gate = torch.sigmoid(self.residual_gate)
            x_out = (1 - gate) * identity + gate * x_proj

        elif self.mode == "full":
            # Depthwise conv + local stats
            x_context = self.depthwise_conv(x_conv)
            local_mean, local_std = self._compute_local_stats(x_conv)
            # Concat original, context, local_std: (B, 3D, H, W)
            x_concat = torch.cat([x_conv, x_context, local_std], dim=1)
            # Permute to (B, H, W, 3D)
            x_concat = x_concat.permute(0, 2, 3, 1)
            # Project back to D
            x_proj = self.stats_proj(x_concat)
            # Residual
            gate = torch.sigmoid(self.residual_gate)
            x_out = (1 - gate) * identity + gate * x_proj

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return x_out
