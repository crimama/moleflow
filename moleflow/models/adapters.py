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
