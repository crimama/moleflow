"""
Task Prototype and Router for Task Selection.

Uses Mahalanobis distance to select the best LoRA expert.

Ablation Support:
- use_mahalanobis: If False, use Euclidean distance instead
"""

import torch
from typing import Dict, List, Optional


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
        """
        Update prototype statistics with new features.

        Args:
            features: (N, D) image-level features
        """
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
            # Incremental update
            n_new = features.shape[0]
            n_total = self.n_samples + n_new

            new_mean = features.mean(dim=0)
            delta = new_mean - self.mean

            # Update mean
            self.mean = (self.n_samples * self.mean + n_new * new_mean) / n_total

            # Update covariance (Welford's online algorithm)
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
                # Fallback: use pseudo-inverse
                self.precision = torch.linalg.pinv(self.covariance)

    def mahalanobis_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Mahalanobis distance from prototype.

        Args:
            features: (N, D) features

        Returns:
            distances: (N,) Mahalanobis distances
        """
        if self.mean is None or self.precision is None:
            raise ValueError("Prototype not initialized. Call finalize() first.")

        centered = features - self.mean.unsqueeze(0)  # (N, D)
        # D_M = sqrt((x - mu)^T Sigma^{-1} (x - mu))
        # (N, D) @ (D, D) @ (D, N) -> diagonal gives (N,)
        distances = torch.sqrt(torch.sum(centered @ self.precision * centered, dim=1))

        return distances

    def euclidean_distance(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute Euclidean distance from prototype.

        Args:
            features: (N, D) features

        Returns:
            distances: (N,) Euclidean distances
        """
        if self.mean is None:
            raise ValueError("Prototype not initialized.")

        centered = features - self.mean.unsqueeze(0)  # (N, D)
        distances = torch.sqrt(torch.sum(centered ** 2, dim=1))

        return distances


class PrototypeRouter:
    """
    Prototype-based Router for Task Selection.

    Uses Mahalanobis or Euclidean distance to select the best LoRA expert.

    Ablation Support:
    - use_mahalanobis: If False, use Euclidean distance instead
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

        Args:
            features: (N, D) image-level features

        Returns:
            task_ids: (N,) predicted task IDs

        Uses Mahalanobis distance by default, or Euclidean if use_mahalanobis=False.
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

        # Stack: (num_tasks, N)
        all_distances = torch.stack(all_distances, dim=0)

        # Find minimum distance task
        min_indices = torch.argmin(all_distances, dim=0)
        predicted_tasks = torch.tensor([task_ids[idx] for idx in min_indices],
                                        device=features.device)

        return predicted_tasks
