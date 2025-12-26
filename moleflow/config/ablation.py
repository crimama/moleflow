"""
Ablation Configuration for MoLE-Flow.

This module provides configuration for ablation studies,
allowing individual components to be enabled/disabled.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AblationConfig:
    """
    Configuration for ablation studies.

    Each flag controls whether a specific component is enabled.
    Set to False to disable the component for ablation study.

    Components:
        use_lora: LoRA (Low-Rank Adaptation) for task-specific adaptation
        use_router: Prototype-based router for task selection
        use_task_adapter: Task Input Adapter (Instance Normalization)
        use_pos_embedding: Positional embedding for spatial awareness
        use_slow_stage: Slow-Fast two-stage training
        use_task_bias: Task-specific bias in LoRA layers
        use_mahalanobis: Mahalanobis distance in router (vs Euclidean)
    """
    # Core components
    use_lora: bool = True
    use_router: bool = True
    use_task_adapter: bool = True
    use_pos_embedding: bool = True

    # Training strategies
    use_slow_stage: bool = False

    # Fine-grained controls
    use_task_bias: bool = True
    use_mahalanobis: bool = True

    # LoRA configuration (only used if use_lora=True)
    lora_rank: int = 32
    lora_alpha: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not self.use_lora and self.use_task_bias:
            # Task bias requires LoRA
            self.use_task_bias = False

    def get_active_components(self) -> list:
        """Return list of active component names."""
        components = []
        if self.use_lora:
            components.append("LoRA")
        if self.use_router:
            components.append("Router")
        if self.use_task_adapter:
            components.append("TaskAdapter")
        if self.use_pos_embedding:
            components.append("PosEmbed")
        if self.use_slow_stage:
            components.append("SlowStage")
        if self.use_task_bias:
            components.append("TaskBias")
        if self.use_mahalanobis:
            components.append("Mahalanobis")
        return components

    def get_disabled_components(self) -> list:
        """Return list of disabled component names."""
        all_components = {
            "LoRA": self.use_lora,
            "Router": self.use_router,
            "TaskAdapter": self.use_task_adapter,
            "PosEmbed": self.use_pos_embedding,
            "SlowStage": self.use_slow_stage,
            "TaskBias": self.use_task_bias,
            "Mahalanobis": self.use_mahalanobis,
        }
        return [name for name, enabled in all_components.items() if not enabled]

    def get_experiment_name(self) -> str:
        """Generate experiment name based on disabled components."""
        disabled = self.get_disabled_components()
        if not disabled:
            return "full_model"
        return "wo_" + "_".join([c.lower() for c in disabled])

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["AblationConfig:"]
        lines.append(f"  Active: {', '.join(self.get_active_components())}")
        disabled = self.get_disabled_components()
        if disabled:
            lines.append(f"  Disabled: {', '.join(disabled)}")
        return "\n".join(lines)


# Predefined ablation configurations for common experiments
ABLATION_PRESETS = {
    "full": AblationConfig(),

    # Single component ablations
    "wo_lora": AblationConfig(use_lora=False),
    "wo_router": AblationConfig(use_router=False),
    "wo_adapter": AblationConfig(use_task_adapter=False),
    "wo_pos_embed": AblationConfig(use_pos_embedding=False),

    # Combined ablations
    "wo_router_adapter": AblationConfig(use_router=False, use_task_adapter=False),

    # Minimal baselines
    "baseline_nf": AblationConfig(
        use_lora=False,
        use_router=False,
        use_task_adapter=False,
        use_pos_embedding=False,
    ),
    "lora_only": AblationConfig(
        use_router=False,
        use_task_adapter=False,
    ),
}


def get_ablation_config(preset_name: str = None, **kwargs) -> AblationConfig:
    """
    Get ablation configuration.

    Args:
        preset_name: Name of preset configuration (see ABLATION_PRESETS)
        **kwargs: Override specific settings

    Returns:
        AblationConfig instance
    """
    if preset_name and preset_name in ABLATION_PRESETS:
        config = ABLATION_PRESETS[preset_name]
        # Apply overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    return AblationConfig(**kwargs)


def add_ablation_args(parser):
    """
    Add ablation arguments to argument parser.

    Args:
        parser: argparse.ArgumentParser instance

    Returns:
        parser with added arguments
    """
    ablation_group = parser.add_argument_group('Ablation Settings')

    ablation_group.add_argument(
        '--ablation_preset', type=str, default=None,
        choices=list(ABLATION_PRESETS.keys()),
        help='Use a predefined ablation configuration'
    )

    # Core components
    ablation_group.add_argument(
        '--no_lora', action='store_true',
        help='Disable LoRA adaptation (use shared NF weights)'
    )
    ablation_group.add_argument(
        '--no_router', action='store_true',
        help='Disable prototype router (use oracle task_id)'
    )
    ablation_group.add_argument(
        '--no_task_adapter', action='store_true',
        help='Disable task input adapter (instance normalization)'
    )
    ablation_group.add_argument(
        '--no_pos_embedding', action='store_true',
        help='Disable positional embedding'
    )

    # Fine-grained controls
    ablation_group.add_argument(
        '--no_task_bias', action='store_true',
        help='Disable task-specific bias in LoRA layers'
    )
    ablation_group.add_argument(
        '--no_mahalanobis', action='store_true',
        help='Use Euclidean distance instead of Mahalanobis in router'
    )

    return parser


def parse_ablation_args(parsed_args) -> AblationConfig:
    """
    Create AblationConfig from parsed arguments.

    Args:
        parsed_args: Parsed argparse namespace

    Returns:
        AblationConfig instance
    """
    # Check for preset first
    if hasattr(parsed_args, 'ablation_preset') and parsed_args.ablation_preset:
        config = get_ablation_config(parsed_args.ablation_preset)
    else:
        config = AblationConfig()

    # Apply individual overrides
    if hasattr(parsed_args, 'no_lora') and parsed_args.no_lora:
        config.use_lora = False
    if hasattr(parsed_args, 'no_router') and parsed_args.no_router:
        config.use_router = False
    if hasattr(parsed_args, 'no_task_adapter') and parsed_args.no_task_adapter:
        config.use_task_adapter = False
    if hasattr(parsed_args, 'no_pos_embedding') and parsed_args.no_pos_embedding:
        config.use_pos_embedding = False
    if hasattr(parsed_args, 'no_task_bias') and parsed_args.no_task_bias:
        config.use_task_bias = False
    if hasattr(parsed_args, 'no_mahalanobis') and parsed_args.no_mahalanobis:
        config.use_mahalanobis = False

    # Apply slow stage setting
    if hasattr(parsed_args, 'enable_slow_stage'):
        config.use_slow_stage = parsed_args.enable_slow_stage

    # Apply LoRA settings
    if hasattr(parsed_args, 'lora_rank'):
        config.lora_rank = parsed_args.lora_rank

    return config
