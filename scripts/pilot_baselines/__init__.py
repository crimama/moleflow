"""
Pilot Baselines for MoLE-Flow Paper

This package contains baseline implementations for the pilot experiment
comparing different anomaly detection methods under continual learning.

Key Hypothesis:
    Parameter decomposition (Base frozen + Task-specific trainable) works for
    Normalizing Flows but NOT for reconstruction-based or distillation-based methods.

Available Baselines:
    - ae_baseline.py: AutoEncoder baseline with frozen encoder after Task 0
    - vae_baseline.py: Variational AutoEncoder baseline with frozen encoder after Task 0
    - teacher_student_baseline.py: Teacher-Student (Knowledge Distillation) baseline
    - memory_bank_baseline.py: PatchCore-style Memory Bank baseline
    - run_pilot_experiment.py: Unified experiment runner for all baselines

Shared Utilities:
    - shared_utils.py: Feature extraction, evaluation metrics, logging

Teacher-Student Hypothesis:
    When Teacher is frozen and only Student learns, the Teacher-Student alignment
    breaks for new tasks because:
    1. Teacher's "normal" representation is Task 0 specific
    2. Student cannot properly learn new task's normal without Teacher adaptation

Memory Bank Key Insight:
    Memory Bank methods have NO learnable parameters for "Base + Task-specific"
    decomposition. The only way to handle new tasks is to store their features,
    which is equivalent to data replay.

    - Task-Separated Mode: Stores features per task (implicit replay via storage)
    - Accumulated Mode: Stores ALL features (explicit replay)

Expected Results:
    - MoLE-Flow: Good performance on all tasks (LoRA adapts density estimation)
    - AE/VAE Baseline: Poor performance on Task 1+ (frozen encoder cannot adapt)
    - Teacher-Student: Poor performance on Task 1+ (frozen Teacher cannot represent new normal)
    - Memory Bank: Requires data storage (replay) for continual learning

Usage:
    # Run unified experiment with all baselines
    python -m scripts.pilot_baselines.run_pilot_experiment --models ae vae ts memory

    # Run individual baselines
    python -m scripts.pilot_baselines.ae_baseline --task_classes leather grid transistor
    python -m scripts.pilot_baselines.vae_baseline --task_classes leather grid transistor
    python -m scripts.pilot_baselines.teacher_student_baseline --task_classes leather grid transistor
    python -m scripts.pilot_baselines.memory_bank_baseline --task_classes leather grid transistor
"""

from .ae_baseline import AEModel, AEContinualTrainer
from .vae_baseline import VAEModel, VAEContinualTrainer
from .teacher_student_baseline import (
    TeacherNetwork,
    StudentNetwork,
    TeacherStudentModel,
    TSContinualTrainer,
)
from .memory_bank_baseline import (
    MemoryBank,
    MultiTaskMemoryBank,
    MemoryBankModel,
    MBContinualTrainer,
)

# Shared utilities for unified experiment runner
from .shared_utils import (
    ExperimentConfig,
    TaskMetrics,
    TrainingState,
    FeatureExtractorWrapper,
    ExperimentLogger,
    BaselineModel,
    create_dataloader,
    create_task_dataloaders,
    compute_image_auc,
    compute_pixel_metrics,
    compute_forgetting_measure,
    evaluate_model,
    evaluate_all_tasks,
    set_seed,
    generate_comparison_table,
    generate_forgetting_report,
)

__all__ = [
    # Baseline models
    'AEModel', 'AEContinualTrainer',
    'VAEModel', 'VAEContinualTrainer',
    'TeacherNetwork', 'StudentNetwork', 'TeacherStudentModel', 'TSContinualTrainer',
    'MemoryBank', 'MultiTaskMemoryBank', 'MemoryBankModel', 'MBContinualTrainer',
    # Shared utilities
    'ExperimentConfig',
    'TaskMetrics',
    'TrainingState',
    'FeatureExtractorWrapper',
    'ExperimentLogger',
    'BaselineModel',
    'create_dataloader',
    'create_task_dataloaders',
    'compute_image_auc',
    'compute_pixel_metrics',
    'compute_forgetting_measure',
    'evaluate_model',
    'evaluate_all_tasks',
    'set_seed',
    'generate_comparison_table',
    'generate_forgetting_report',
]
