"""
MoLE-Flow: Mixture of LoRA Experts for Continual Anomaly Detection

This package provides modular components for continual learning based
anomaly detection using Normalizing Flows with LoRA adapters.

Ablation Support:
- AblationConfig: Configuration class for ablation studies
- ABLATION_PRESETS: Predefined ablation configurations
- get_ablation_config: Factory function for ablation configs
"""

from moleflow.models.mole_nf import MoLESpatialAwareNF
from moleflow.trainer.continual_trainer import MoLEContinualTrainer
from moleflow.extractors.vit_extractor import ViTPatchCoreExtractor
from moleflow.extractors.cnn_extractor import CNNPatchCoreExtractor, PatchCoreExtractor
from moleflow.extractors import (
    create_feature_extractor,
    get_backbone_type,
    is_vit_backbone,
    is_cnn_backbone,
)
from moleflow.models.position_embedding import PositionalEmbeddingGenerator
from moleflow.utils.logger import TrainingLogger, setup_training_logger
from moleflow.utils.helpers import init_seeds, setting_lr_parameters
from moleflow.utils.config import get_config, get_default_config
from moleflow.evaluation.evaluator import (
    evaluate_class,
    evaluate_all_tasks,
    evaluate_routing_performance
)
from moleflow.config.ablation import (
    AblationConfig,
    ABLATION_PRESETS,
    get_ablation_config,
    add_ablation_args,
    parse_ablation_args,
)
from moleflow.data import (
    MVTEC, MVTEC_CLASS_NAMES,
    VISA, VISA_CLASS_NAMES,
    MPDD, MPDD_CLASS_NAMES,
    create_task_dataset, TaskDataset,
    get_dataset_class, get_class_names, DATASET_REGISTRY,
)
from moleflow.utils.diagnostics import FlowDiagnostics, run_diagnostics_on_model

__version__ = "0.1.0"
__all__ = [
    # Core components
    "MoLESpatialAwareNF",
    "MoLEContinualTrainer",
    # Feature extractors
    "ViTPatchCoreExtractor",
    "CNNPatchCoreExtractor",
    "PatchCoreExtractor",
    "create_feature_extractor",
    "get_backbone_type",
    "is_vit_backbone",
    "is_cnn_backbone",
    # Position embedding
    "PositionalEmbeddingGenerator",
    # Logging
    "TrainingLogger",
    "setup_training_logger",
    # Utilities
    "init_seeds",
    "setting_lr_parameters",
    "get_config",
    "get_default_config",
    # Evaluation
    "evaluate_class",
    "evaluate_all_tasks",
    "evaluate_routing_performance",
    # Diagnostics
    "FlowDiagnostics",
    "run_diagnostics_on_model",
    # Ablation
    "AblationConfig",
    "ABLATION_PRESETS",
    "get_ablation_config",
    "add_ablation_args",
    "parse_ablation_args",
    # Data
    "MVTEC",
    "MVTEC_CLASS_NAMES",
    "VISA",
    "VISA_CLASS_NAMES",
    "MPDD",
    "MPDD_CLASS_NAMES",
    "create_task_dataset",
    "TaskDataset",
    "get_dataset_class",
    "get_class_names",
    "DATASET_REGISTRY",
]
