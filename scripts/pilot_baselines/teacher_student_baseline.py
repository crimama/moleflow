#!/usr/bin/env python
"""
Teacher-Student Baseline for Continual Anomaly Detection.

Pilot Experiment: Demonstrates that parameter decomposition (Base frozen + Task-specific trainable)
works for Normalizing Flows but NOT for other architectures like Teacher-Student.

Architecture:
    Input: Patch embeddings from frozen WideResNet-50 backbone (768-dim)
    Teacher: 2-3 FC layers -> Teacher features
    Student: 2-3 FC layers (same architecture) -> Student features
    Loss: ||Teacher(x) - Student(x)||^2 (MSE)
    Anomaly Score: ||Teacher(x) - Student(x)||^2 at test time

Continual Learning Setup (Base + Task-specific decomposition):
    Task 0: Train Student to match Teacher (Teacher starts randomly initialized, jointly learned)
    Task 0 complete: Freeze Teacher (Base) - represents "what normal looks like"
    Task 1+: Train only Student (Task-specific) - adapts to new task's normal

Hypothesis:
    When Teacher is frozen and only Student learns, the Teacher-Student alignment breaks
    for new tasks because:
    1. Teacher's "normal" representation is Task 0 specific
    2. Student cannot properly learn new task's normal without Teacher adaptation

Usage:
    python teacher_student_baseline.py \
        --task_classes leather grid transistor \
        --num_epochs 40 \
        --hidden_dim 512

Author: MoLE-Flow Team
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.ndimage import gaussian_filter

# Add parent path to import moleflow data utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moleflow.data.mvtec import MVTEC, MVTEC_CLASS_NAMES
from moleflow.extractors.cnn_extractor import CNNPatchCoreExtractor


# =============================================================================
# Model Architecture
# =============================================================================

class TeacherNetwork(nn.Module):
    """
    Teacher Network for Knowledge Distillation-based Anomaly Detection.

    Architecture: 3 FC layers with BatchNorm and ReLU.
    Learns a representation of "normal" features during Task 0.

    Args:
        input_dim: Input feature dimension (from backbone)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            # Layer 3 (output)
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, D] or [B, H, W, D]

        Returns:
            Output features [B, output_dim] or [B, H, W, output_dim]
        """
        original_shape = x.shape

        if len(original_shape) == 4:
            # Spatial input: [B, H, W, D]
            B, H, W, D = original_shape
            x = x.reshape(B * H * W, D)
            out = self.network(x)
            out = out.reshape(B, H, W, -1)
        else:
            # Flat input: [B, D]
            out = self.network(x)

        return out


class StudentNetwork(nn.Module):
    """
    Student Network for Knowledge Distillation-based Anomaly Detection.

    Same architecture as Teacher, learns to mimic Teacher's output.

    Args:
        input_dim: Input feature dimension (from backbone)
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension (must match Teacher)
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            # Layer 3 (output)
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [B, D] or [B, H, W, D]

        Returns:
            Output features [B, output_dim] or [B, H, W, output_dim]
        """
        original_shape = x.shape

        if len(original_shape) == 4:
            # Spatial input: [B, H, W, D]
            B, H, W, D = original_shape
            x = x.reshape(B * H * W, D)
            out = self.network(x)
            out = out.reshape(B, H, W, -1)
        else:
            # Flat input: [B, D]
            out = self.network(x)

        return out


class TeacherStudentModel(nn.Module):
    """
    Teacher-Student Model for Continual Anomaly Detection.

    Combines Teacher and Student networks with distillation loss.

    Continual Learning Strategy:
        - Task 0: Train both Teacher and Student jointly
        - Task 0 complete: Freeze Teacher (becomes the "base" representation)
        - Task 1+: Train only Student (task-specific adaptation)

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        device: Device to use
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device

        # Networks
        self.teacher = TeacherNetwork(input_dim, hidden_dim, output_dim)
        self.student = StudentNetwork(input_dim, hidden_dim, output_dim)

        # State tracking
        self.teacher_frozen = False
        self.current_task = 0

        # Move to device
        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Teacher and Student.

        Args:
            x: Input features [B, H, W, D]

        Returns:
            teacher_out: Teacher features [B, H, W, output_dim]
            student_out: Student features [B, H, W, output_dim]
        """
        teacher_out = self.teacher(x)
        student_out = self.student(x)
        return teacher_out, student_out

    def compute_distillation_loss(
        self,
        x: torch.Tensor,
        loss_type: str = 'mse'
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute distillation loss between Teacher and Student.

        Args:
            x: Input features [B, H, W, D]
            loss_type: 'mse' or 'cosine'

        Returns:
            loss: Scalar loss value
            info: Dictionary with additional metrics
        """
        teacher_out, student_out = self.forward(x)

        if loss_type == 'mse':
            # MSE loss
            loss = F.mse_loss(student_out, teacher_out.detach() if self.teacher_frozen else teacher_out)
        elif loss_type == 'cosine':
            # Cosine similarity loss
            teacher_flat = teacher_out.reshape(-1, self.output_dim)
            student_flat = student_out.reshape(-1, self.output_dim)
            cosine_sim = F.cosine_similarity(student_flat, teacher_flat.detach() if self.teacher_frozen else teacher_flat, dim=1)
            loss = 1 - cosine_sim.mean()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Compute additional metrics
        with torch.no_grad():
            mse = F.mse_loss(student_out, teacher_out).item()
            teacher_norm = teacher_out.norm(dim=-1).mean().item()
            student_norm = student_out.norm(dim=-1).mean().item()

        info = {
            'loss': loss.item(),
            'mse': mse,
            'teacher_norm': teacher_norm,
            'student_norm': student_norm,
        }

        return loss, info

    def compute_anomaly_score(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute anomaly scores based on Teacher-Student discrepancy.

        Args:
            x: Input features [B, H, W, D]

        Returns:
            pixel_scores: Spatial anomaly map [B, H, W]
            image_scores: Image-level scores [B]
        """
        self.eval()
        with torch.no_grad():
            teacher_out, student_out = self.forward(x)

            # Pixel-wise anomaly score: L2 distance
            diff = (teacher_out - student_out) ** 2
            pixel_scores = diff.sum(dim=-1)  # [B, H, W]

            # Image-level score: max or top-k pooling
            flat_scores = pixel_scores.reshape(pixel_scores.shape[0], -1)
            k = max(1, int(flat_scores.shape[1] * 0.02))  # Top 2%
            top_k_scores, _ = torch.topk(flat_scores, k, dim=1)
            image_scores = top_k_scores.mean(dim=1)  # [B]

        return pixel_scores, image_scores

    def freeze_teacher(self):
        """
        Freeze Teacher network after Task 0.

        This is the key operation for the continual learning setup:
        After Task 0, Teacher represents the "base" normal representation
        and should not be updated.
        """
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        self.teacher_frozen = True
        print("  [TS] Teacher network FROZEN (base representation locked)")

    def unfreeze_teacher(self):
        """Unfreeze Teacher network (for ablation studies)."""
        for param in self.teacher.parameters():
            param.requires_grad = True
        self.teacher.train()
        self.teacher_frozen = False
        print("  [TS] Teacher network UNFROZEN")

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get trainable parameters based on current state."""
        params = list(self.student.parameters())
        if not self.teacher_frozen:
            params.extend(self.teacher.parameters())
        return params

    def set_task(self, task_id: int):
        """Set current task (for logging purposes)."""
        self.current_task = task_id


# =============================================================================
# Trainer
# =============================================================================

class TSContinualTrainer:
    """
    Continual Learning Trainer for Teacher-Student Baseline.

    Training Strategy:
        Task 0: Joint training of Teacher and Student
        Task 1+: Frozen Teacher, only Student trains

    This demonstrates that freezing the "base" (Teacher) breaks
    the model's ability to learn new tasks, unlike Normalizing Flows
    where base weights can be reused.

    Args:
        feature_extractor: Frozen backbone for feature extraction
        ts_model: TeacherStudentModel instance
        args: Configuration arguments
        device: Device to use
    """

    def __init__(
        self,
        feature_extractor: CNNPatchCoreExtractor,
        ts_model: TeacherStudentModel,
        args,
        device: str = 'cuda'
    ):
        self.feature_extractor = feature_extractor
        self.ts_model = ts_model
        self.args = args
        self.device = device

        # Task tracking
        self.task_classes: Dict[int, List[str]] = {}
        self.training_history: Dict[int, List[Dict]] = {}

    def train_task(
        self,
        task_id: int,
        task_classes: List[str],
        train_loader: DataLoader,
        num_epochs: int = 40,
        lr: float = 1e-4,
        log_interval: int = 10
    ):
        """
        Train on a single task.

        Args:
            task_id: Task identifier
            task_classes: List of class names in this task
            train_loader: Training data loader
            num_epochs: Number of training epochs
            lr: Learning rate
            log_interval: Logging interval
        """
        print(f"\n{'='*70}")
        print(f"Training Task {task_id}: {task_classes}")
        print(f"{'='*70}")

        self.task_classes[task_id] = task_classes
        self.ts_model.set_task(task_id)

        # Task 0: Train both Teacher and Student
        # Task 1+: Freeze Teacher, train only Student
        if task_id == 0:
            print(f"  [Task 0] Joint training: Teacher + Student")
            self._train_joint(task_id, train_loader, num_epochs, lr, log_interval)
            # Freeze Teacher after Task 0
            self.ts_model.freeze_teacher()
        else:
            print(f"  [Task {task_id}] Frozen Teacher, training Student only")
            self._train_student_only(task_id, train_loader, num_epochs, lr, log_interval)

        print(f"\n  Task {task_id} training completed!")

    def _train_joint(
        self,
        task_id: int,
        train_loader: DataLoader,
        num_epochs: int,
        lr: float,
        log_interval: int
    ):
        """Train both Teacher and Student jointly (Task 0)."""
        self.ts_model.train()
        self.feature_extractor.eval()

        # All parameters trainable
        params = self.ts_model.get_trainable_params()
        num_params = sum(p.numel() for p in params)
        print(f"  Trainable parameters: {num_params:,}")

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch[0].to(self.device)

                # Extract features from frozen backbone
                with torch.no_grad():
                    features = self.feature_extractor(images)  # [B, H, W, D]

                # Compute loss
                loss, info = self.ts_model.compute_distillation_loss(features, loss_type='mse')

                # Skip invalid batches
                if torch.isnan(loss) or loss.item() > 1e8:
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"    [JOINT] Epoch [{epoch+1}/{num_epochs}] "
                          f"Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            history.append({'epoch': epoch, 'loss': avg_epoch_loss})

            print(f"  [JOINT] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        self.training_history[task_id] = history

    def _train_student_only(
        self,
        task_id: int,
        train_loader: DataLoader,
        num_epochs: int,
        lr: float,
        log_interval: int
    ):
        """Train only Student with frozen Teacher (Task 1+)."""
        self.ts_model.student.train()
        self.ts_model.teacher.eval()
        self.feature_extractor.eval()

        # Only Student parameters trainable
        params = list(self.ts_model.student.parameters())
        num_params = sum(p.numel() for p in params)
        print(f"  Trainable parameters (Student only): {num_params:,}")

        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        history = []

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                images = batch[0].to(self.device)

                # Extract features from frozen backbone
                with torch.no_grad():
                    features = self.feature_extractor(images)  # [B, H, W, D]

                # Compute loss (Teacher is frozen, so detach in loss computation)
                loss, info = self.ts_model.compute_distillation_loss(features, loss_type='mse')

                # Skip invalid batches
                if torch.isnan(loss) or loss.item() > 1e8:
                    continue

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    print(f"    [STUDENT] Epoch [{epoch+1}/{num_epochs}] "
                          f"Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            history.append({'epoch': epoch, 'loss': avg_epoch_loss})

            print(f"  [STUDENT] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

        self.training_history[task_id] = history

    def inference(
        self,
        images: torch.Tensor,
        task_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run inference to compute anomaly scores.

        Args:
            images: Input images [B, C, H, W]
            task_id: Task ID (not used, kept for API compatibility)

        Returns:
            pixel_scores: Spatial anomaly map [B, H, W]
            image_scores: Image-level scores [B]
        """
        self.ts_model.eval()
        self.feature_extractor.eval()

        with torch.no_grad():
            images = images.to(self.device)

            # Extract features
            features = self.feature_extractor(images)  # [B, H, W, D]

            # Compute anomaly scores
            pixel_scores, image_scores = self.ts_model.compute_anomaly_score(features)

        return pixel_scores, image_scores


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_class(
    trainer: TSContinualTrainer,
    class_name: str,
    args,
    target_size: int = 224
) -> Dict:
    """
    Evaluate on a single class.

    Args:
        trainer: TSContinualTrainer instance
        class_name: Class name to evaluate
        args: Configuration
        target_size: Target size for pixel-level evaluation

    Returns:
        Dictionary with evaluation metrics
    """
    device = trainer.device

    # Create test dataset
    test_dataset = MVTEC(
        args.data_path, class_name=class_name, train=False,
        img_size=args.img_size, crp_size=args.img_size, msk_size=target_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=8, shuffle=False,
        drop_last=False, pin_memory=True, num_workers=4
    )

    gt_label_list = []
    gt_mask_list = []
    pixel_scores_list = []
    image_scores_list = []

    for batch in test_loader:
        images, labels, masks = batch[0], batch[1], batch[2]
        gt_label_list.extend(labels.cpu().numpy())
        gt_mask_list.extend(masks.cpu().numpy())

        # Inference
        pixel_scores, image_scores = trainer.inference(images.to(device))

        # Resize for pixel-level evaluation
        pixel_scores_resized = F.interpolate(
            pixel_scores.unsqueeze(1),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=True
        ).squeeze(1)

        pixel_scores_list.append(pixel_scores_resized.cpu())
        image_scores_list.append(image_scores.cpu())

    # Concatenate results
    pixel_scores_all = torch.cat(pixel_scores_list, dim=0).numpy()
    image_scores_all = torch.cat(image_scores_list, dim=0).numpy()

    gt_label = np.asarray(gt_label_list, dtype=bool)
    gt_mask = np.squeeze(np.asarray(gt_mask_list, dtype=bool), axis=1)

    # Apply Gaussian smoothing
    for i in range(pixel_scores_all.shape[0]):
        pixel_scores_all[i] = gaussian_filter(pixel_scores_all[i], sigma=4)

    # Sanitize scores
    image_scores_all = np.nan_to_num(image_scores_all, nan=0.0, posinf=1e6, neginf=0.0)
    pixel_scores_all = np.nan_to_num(pixel_scores_all, nan=0.0, posinf=1e6, neginf=0.0)

    # Compute metrics
    img_auc = roc_auc_score(gt_label, image_scores_all)
    pixel_auc = roc_auc_score(gt_mask.flatten(), pixel_scores_all.flatten())
    img_ap = average_precision_score(gt_label, image_scores_all)
    pixel_ap = average_precision_score(gt_mask.flatten(), pixel_scores_all.flatten())

    return {
        'img_auc': img_auc,
        'pixel_auc': pixel_auc,
        'img_ap': img_ap,
        'pixel_ap': pixel_ap,
        'n_samples': len(gt_label)
    }


def evaluate_all_tasks(
    trainer: TSContinualTrainer,
    args,
    target_size: int = 224
) -> Dict:
    """
    Evaluate all learned tasks.

    Args:
        trainer: TSContinualTrainer instance
        args: Configuration
        target_size: Target size for evaluation

    Returns:
        Dictionary with evaluation results
    """
    results = {
        'classes': [],
        'task_ids': [],
        'img_aucs': [],
        'pixel_aucs': [],
        'img_aps': [],
        'pixel_aps': [],
        'class_img_aucs': {},
        'class_pixel_aucs': {},
    }

    print("\n" + "="*70)
    print("Evaluating All Tasks (Teacher-Student Baseline)")
    print("="*70)

    for task_id, task_classes in trainer.task_classes.items():
        print(f"\nTask {task_id}: {task_classes}")

        for class_name in task_classes:
            print(f"  Evaluating {class_name}...")

            class_results = evaluate_class(trainer, class_name, args, target_size)

            img_auc = class_results['img_auc']
            pixel_auc = class_results['pixel_auc']

            results['classes'].append(class_name)
            results['task_ids'].append(task_id)
            results['img_aucs'].append(img_auc)
            results['pixel_aucs'].append(pixel_auc)
            results['img_aps'].append(class_results['img_ap'])
            results['pixel_aps'].append(class_results['pixel_ap'])
            results['class_img_aucs'][class_name] = img_auc
            results['class_pixel_aucs'][class_name] = pixel_auc

            print(f"    {class_name}: Image AUC={img_auc:.4f}, Pixel AUC={pixel_auc:.4f}")

    # Compute averages
    results['mean_img_auc'] = np.mean(results['img_aucs'])
    results['mean_pixel_auc'] = np.mean(results['pixel_aucs'])
    results['mean_img_ap'] = np.mean(results['img_aps'])
    results['mean_pixel_ap'] = np.mean(results['pixel_aps'])

    # Task-wise averages
    results['task_avg_img_aucs'] = {}
    results['task_avg_pixel_aucs'] = {}

    for task_id in trainer.task_classes.keys():
        task_indices = [i for i, t in enumerate(results['task_ids']) if t == task_id]
        results['task_avg_img_aucs'][task_id] = np.mean([results['img_aucs'][i] for i in task_indices])
        results['task_avg_pixel_aucs'][task_id] = np.mean([results['pixel_aucs'][i] for i in task_indices])

    # Print summary
    print("\n" + "="*70)
    print("Evaluation Summary (Teacher-Student Baseline)")
    print("="*70)

    for i, class_name in enumerate(results['classes']):
        print(f"  {class_name:15s} (Task {results['task_ids'][i]}): "
              f"I-AUC={results['img_aucs'][i]:.4f}, P-AUC={results['pixel_aucs'][i]:.4f}")

    print("-"*70)

    for task_id in trainer.task_classes.keys():
        print(f"  Task {task_id} Average: "
              f"I-AUC={results['task_avg_img_aucs'][task_id]:.4f}, "
              f"P-AUC={results['task_avg_pixel_aucs'][task_id]:.4f}")

    print("-"*70)
    print(f"  Overall Average: "
          f"I-AUC={results['mean_img_auc']:.4f}, P-AUC={results['mean_pixel_auc']:.4f}")
    print("="*70)

    return results


# =============================================================================
# Data Utilities
# =============================================================================

class TaskDataset(Dataset):
    """Dataset wrapper for continual learning tasks."""

    def __init__(self, data: List[Tuple], class_names: List[str], class_to_idx: Dict[str, int]):
        self.data = data
        self.class_names = class_names
        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]


def create_task_dataset(
    args,
    task_classes: List[str],
    global_class_to_idx: Dict[str, int],
    train: bool = True
) -> TaskDataset:
    """
    Create dataset for specific task classes.

    Args:
        args: Configuration with data_path, img_size, msk_size
        task_classes: List of class names for this task
        global_class_to_idx: Global class to index mapping
        train: Whether to load training or test data

    Returns:
        TaskDataset for the specified classes
    """
    filtered_data = []

    for class_name in task_classes:
        class_dataset = MVTEC(
            root=args.data_path,
            class_name=class_name,
            train=train,
            img_size=args.img_size,
            crp_size=args.img_size,
            msk_size=args.msk_size
        )

        for i in range(len(class_dataset)):
            image, label, mask, name, path = class_dataset[i]
            final_class_id = int(global_class_to_idx[class_name])
            filtered_data.append((image, final_class_id, mask, name, path))

    task_class_to_idx = {cls: global_class_to_idx[cls] for cls in task_classes}
    return TaskDataset(filtered_data, task_classes, task_class_to_idx)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Teacher-Student Baseline for Continual Anomaly Detection'
    )

    # Task configuration
    parser.add_argument('--task_classes', type=str, nargs='+',
                        default=['leather', 'grid', 'transistor'],
                        help='Classes to learn sequentially (each class = 1 task)')

    # Model configuration
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for Teacher/Student networks')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Output dimension for Teacher/Student networks')
    parser.add_argument('--backbone_name', type=str, default='wide_resnet50_2',
                        help='Backbone model name')

    # Training configuration
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of epochs per task')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')

    # Data configuration
    parser.add_argument('--data_path', type=str, default='/Data/MVTecAD',
                        help='Path to MVTec AD dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--msk_size', type=int, default=224,
                        help='Mask size for evaluation')

    # Output configuration
    parser.add_argument('--log_dir', type=str,
                        default='/Volume/MoLeFlow/logs/PilotExperiment/TS_baseline',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Setup experiment
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_str = "_".join(args.task_classes[:3])
        args.experiment_name = f"TS_baseline_{task_str}_{timestamp}"

    # Create log directory
    log_dir = Path(args.log_dir) / args.experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("Teacher-Student Baseline for Continual Anomaly Detection")
    print("="*70)
    print(f"   Task Classes: {args.task_classes}")
    print(f"   Hidden Dim: {args.hidden_dim}")
    print(f"   Output Dim: {args.output_dim}")
    print(f"   Epochs per Task: {args.num_epochs}")
    print(f"   Learning Rate: {args.lr}")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Backbone: {args.backbone_name}")
    print(f"   Log Directory: {log_dir}")
    print("="*70)

    # Global class mapping
    ALL_CLASSES = args.task_classes
    GLOBAL_CLASS_TO_IDX = {cls: i for i, cls in enumerate(ALL_CLASSES)}

    # Each class is one task (1-1 scenario)
    CONTINUAL_TASKS = [[cls] for cls in ALL_CLASSES]

    print("\nContinual Learning Setup (1-1 scenario):")
    for t_id, t_classes in enumerate(CONTINUAL_TASKS):
        freeze_status = "(Teacher FROZEN)" if t_id > 0 else "(Joint Training)"
        print(f"   Task {t_id}: {t_classes[0]} {freeze_status}")

    # Initialize feature extractor (frozen backbone)
    print("\nInitializing feature extractor...")
    embed_dim = 768  # Output dimension for wide_resnet50_2

    feature_extractor = CNNPatchCoreExtractor(
        backbone_name=args.backbone_name,
        input_shape=(3, args.img_size, args.img_size),
        target_embed_dimension=embed_dim,
        device=device
    )
    print(f"   Backbone: {args.backbone_name} (FROZEN)")
    print(f"   Embedding Dimension: {embed_dim}")

    # Initialize Teacher-Student model
    print("\nInitializing Teacher-Student model...")
    ts_model = TeacherStudentModel(
        input_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        device=device
    )

    teacher_params = sum(p.numel() for p in ts_model.teacher.parameters())
    student_params = sum(p.numel() for p in ts_model.student.parameters())
    print(f"   Teacher Parameters: {teacher_params:,}")
    print(f"   Student Parameters: {student_params:,}")

    # Initialize trainer
    trainer = TSContinualTrainer(
        feature_extractor=feature_extractor,
        ts_model=ts_model,
        args=args,
        device=device
    )

    # Save configuration
    config = {
        'experiment_name': args.experiment_name,
        'task_classes': args.task_classes,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.output_dim,
        'backbone_name': args.backbone_name,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'img_size': args.img_size,
        'seed': args.seed,
        'embed_dim': embed_dim,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'continual_strategy': 'Base + Task-specific decomposition',
        'base': 'Teacher (frozen after Task 0)',
        'task_specific': 'Student (trained each task)',
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(log_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    all_results = []

    for task_id, task_classes in enumerate(CONTINUAL_TASKS):
        print(f"\n{'#'*70}")
        print(f"# Task {task_id}: {task_classes}")
        print(f"{'#'*70}")

        # Create task dataset
        train_dataset = create_task_dataset(
            args, task_classes, GLOBAL_CLASS_TO_IDX, train=True
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True
        )

        print(f"\n   Training samples: {len(train_dataset)}")

        # Train on this task
        trainer.train_task(
            task_id=task_id,
            task_classes=task_classes,
            train_loader=train_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            log_interval=10
        )

        # Evaluate all tasks seen so far
        print(f"\n--- Evaluation after Task {task_id} ---")
        results = evaluate_all_tasks(trainer, args, target_size=args.msk_size)

        # Store results
        task_result = {
            'task_id': task_id,
            'task_classes': task_classes,
            'mean_img_auc': results['mean_img_auc'],
            'mean_pixel_auc': results['mean_pixel_auc'],
            'task_avg_img_aucs': dict(results['task_avg_img_aucs']),
            'task_avg_pixel_aucs': dict(results['task_avg_pixel_aucs']),
            'class_img_aucs': dict(results['class_img_aucs']),
            'class_pixel_aucs': dict(results['class_pixel_aucs']),
        }
        all_results.append(task_result)

        # Save intermediate results
        with open(log_dir / f'results_after_task_{task_id}.json', 'w') as f:
            json.dump(task_result, f, indent=2)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY - Teacher-Student Baseline")
    print("="*70)

    print("\n--- Hypothesis Test: Base + Task-specific Decomposition ---")
    print("Expected Behavior:")
    print("  - Task 0: Good performance (joint Teacher-Student training)")
    print("  - Task 1+: Performance should DEGRADE because:")
    print("      1. Teacher is frozen (Task 0 specific 'normal' representation)")
    print("      2. Student cannot learn new task's normal without Teacher adaptation")
    print("")

    # Print task-wise performance
    print("Observed Results:")
    for result in all_results:
        task_id = result['task_id']
        print(f"\n  After Task {task_id} ({result['task_classes'][0]}):")
        print(f"    Overall: I-AUC={result['mean_img_auc']:.4f}, P-AUC={result['mean_pixel_auc']:.4f}")

        for t_id in range(task_id + 1):
            t_classes = CONTINUAL_TASKS[t_id][0]
            img_auc = result['task_avg_img_aucs'].get(t_id, 0.0)
            pixel_auc = result['task_avg_pixel_aucs'].get(t_id, 0.0)
            marker = "*" if t_id == task_id else " "
            print(f"      Task {t_id} ({t_classes}): I-AUC={img_auc:.4f}, P-AUC={pixel_auc:.4f} {marker}")

    # Analysis of forgetting
    if len(all_results) > 1:
        print("\n--- Catastrophic Forgetting Analysis ---")
        task0_initial = all_results[0]['task_avg_img_aucs'][0]
        task0_final = all_results[-1]['task_avg_img_aucs'][0]
        forgetting = task0_initial - task0_final

        print(f"  Task 0 Image AUC:")
        print(f"    After Task 0: {task0_initial:.4f}")
        print(f"    After Task {len(all_results)-1}: {task0_final:.4f}")
        print(f"    Forgetting: {forgetting:+.4f} ({forgetting/task0_initial*100:+.1f}%)")

        # Check if later tasks learned well
        print(f"\n  New Task Performance (frozen Teacher):")
        for i in range(1, len(all_results)):
            task_auc = all_results[i]['task_avg_img_aucs'][i]
            print(f"    Task {i} ({CONTINUAL_TASKS[i][0]}): I-AUC={task_auc:.4f}")

    print("\n" + "="*70)
    print("Experiment completed!")
    print(f"Results saved to: {log_dir}")
    print("="*70)

    # Save final summary
    final_summary = {
        'all_results': all_results,
        'task_classes': args.task_classes,
        'config': config,
        'hypothesis': {
            'name': 'Base + Task-specific decomposition fails for Teacher-Student',
            'expected': 'Task 1+ should degrade due to frozen Teacher',
            'reason': 'Teacher encodes Task 0 specific normal representation'
        }
    }

    with open(log_dir / 'final_summary.json', 'w') as f:
        json.dump(final_summary, f, indent=2)

    # Save CSV for easy analysis
    csv_path = log_dir / 'results.csv'
    with open(csv_path, 'w') as f:
        f.write("eval_after_task,class_name,img_auc,pixel_auc\n")
        for result in all_results:
            task_id = result['task_id']
            for cls_name in result['class_img_aucs'].keys():
                img_auc = result['class_img_aucs'][cls_name]
                pixel_auc = result['class_pixel_aucs'][cls_name]
                f.write(f"{task_id},{cls_name},{img_auc:.4f},{pixel_auc:.4f}\n")

    print(f"\nCSV results saved to: {csv_path}")

    return trainer, all_results


if __name__ == '__main__':
    main()
