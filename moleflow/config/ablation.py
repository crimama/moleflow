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

    Adapter Modes (adapter_mode):
        "standard": Original FiLM-style TaskInputAdapter (v3)
        "robust": RobustTaskInputAdapter with gating - reduces adapter effect on anomalies
        "soft_ln": SoftLNTaskInputAdapter - optional/weak LayerNorm for Task > 0
        "no_ln_after_task0": Disable LN for Task > 0 (quick ablation)
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

    # Adapter mode configuration
    adapter_mode: str = "standard"  # "standard", "robust", "soft_ln", "no_ln_after_task0"
    soft_ln_init_scale: float = 0.01  # Initial scale for soft LN (used when adapter_mode="soft_ln")
    robust_gate_type: str = "norm"  # "norm", "mahal", "learned" (used when adapter_mode="robust")

    # Flow scale(s) regularization
    # Regularizes log_det to reduce variance and stabilize image-level scores
    # loss = nll_loss + lambda_logdet * (logdet_patch ** 2).mean()
    lambda_logdet: float = 3e-5  # Recommended: 3e-5 (stabilizes logdet without hurting performance)

    # Spatial Context Mixing (Legacy - superseded by scale_context)
    # Adds local context to features before NF, so scale(s) can detect local contrast
    # NOTE: Deprecated in favor of use_scale_context (s-only injection)
    use_spatial_context: bool = False
    spatial_context_mode: str = "depthwise_residual"  # "depthwise", "depthwise_residual", "local_stats", "full"
    spatial_context_kernel: int = 3  # Kernel size for context mixing

    # ==========================================================================
    # Scale-specific Context Injection (Baseline 1.5 - RECOMMENDED DEFAULT)
    # ==========================================================================
    # Injects 3x3 local context into s-network ONLY, keeping t-network context-free
    # This is the optimal structure for unsupervised NF-based anomaly detection:
    # - scale(s) sees local contrast → anomaly-sensitive
    # - shift(t) sees x only → density-preserving
    #
    # Key Insight: "판단은 gate가 아니라 Flow scale(s)가 해야 한다"
    # Context를 가리지 말고 주면, Flow가 알아서 anomaly를 구분한다.
    use_scale_context: bool = True   # ENABLED by default (Baseline 1.5)
    scale_context_kernel: int = 3    # 3x3 depthwise conv for local context
    scale_context_init_scale: float = 0.1  # Initial alpha (learnable)
    scale_context_max_alpha: float = 0.2   # Upper bound via sigmoid

    # ==========================================================================
    # Patch-wise Context Gate (Baseline 2.0 - DISABLED by default)
    # ==========================================================================
    # WARNING: Patch-wise gate degrades performance in unsupervised setting!
    #
    # Reason: Gate learns to be "noise suppressor" instead of "anomaly switch"
    # because it only sees normal data during training. This blocks context
    # information and reduces scale(s) sensitivity.
    #
    # Only enable if using semi-supervised learning or auxiliary loss.
    use_context_gate: bool = False   # DISABLED (gate hurts performance)
    context_gate_hidden: int = 64    # Hidden dim for gate MLP

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
        """Generate experiment name based on disabled components and adapter mode."""
        disabled = self.get_disabled_components()
        base_name = "full_model" if not disabled else "wo_" + "_".join([c.lower() for c in disabled])

        # Add adapter mode suffix if not standard
        if self.adapter_mode != "standard":
            base_name += f"_{self.adapter_mode}"
            if self.adapter_mode == "robust":
                base_name += f"_{self.robust_gate_type}"
            elif self.adapter_mode == "soft_ln":
                base_name += f"_s{self.soft_ln_init_scale}"

        # Add logdet regularization suffix if enabled
        if self.lambda_logdet > 0:
            base_name += f"_ldet{self.lambda_logdet:.0e}".replace("-0", "-").replace("+0", "")

        # Add spatial context suffix if enabled (legacy mode)
        if self.use_spatial_context:
            base_name += f"_sctx_{self.spatial_context_mode}"

        # Add scale context suffix if enabled (new improved mode)
        if self.use_scale_context:
            base_name += f"_scalectx_k{self.scale_context_kernel}_a{self.scale_context_init_scale}_max{self.scale_context_max_alpha}"
            # Add gate suffix if using patch-wise gate
            if self.use_context_gate:
                base_name += f"_gate_h{self.context_gate_hidden}"

        return base_name

    def __str__(self) -> str:
        """Human-readable string representation."""
        lines = ["AblationConfig:"]
        lines.append(f"  Active: {', '.join(self.get_active_components())}")
        disabled = self.get_disabled_components()
        if disabled:
            lines.append(f"  Disabled: {', '.join(disabled)}")
        if self.use_task_adapter:
            lines.append(f"  Adapter Mode: {self.adapter_mode}")
            if self.adapter_mode == "soft_ln":
                lines.append(f"    SoftLN Init Scale: {self.soft_ln_init_scale}")
            elif self.adapter_mode == "robust":
                lines.append(f"    Robust Gate Type: {self.robust_gate_type}")
        if self.lambda_logdet > 0:
            lines.append(f"  Logdet Regularization: {self.lambda_logdet:.2e}")
        if self.use_spatial_context:
            lines.append(f"  Spatial Context: {self.spatial_context_mode} (kernel={self.spatial_context_kernel})")
        if self.use_scale_context:
            gate_info = f", gate=patch-wise(h={self.context_gate_hidden})" if self.use_context_gate else ", gate=global_alpha"
            lines.append(f"  Scale Context: s-net only (kernel={self.scale_context_kernel}, init_alpha={self.scale_context_init_scale}, max_alpha={self.scale_context_max_alpha}{gate_info})")
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

    # Adapter mode options
    ablation_group.add_argument(
        '--adapter_mode', type=str, default='standard',
        choices=['standard', 'robust', 'soft_ln', 'no_ln_after_task0'],
        help='Adapter mode: standard (FiLM), robust (gated), soft_ln (weak LN), no_ln_after_task0'
    )
    ablation_group.add_argument(
        '--soft_ln_init_scale', type=float, default=0.01,
        help='Initial scale for soft LN (used with --adapter_mode soft_ln)'
    )
    ablation_group.add_argument(
        '--robust_gate_type', type=str, default='norm',
        choices=['norm', 'mahal', 'learned'],
        help='Gate type for robust adapter: norm, mahal, or learned'
    )

    # Flow regularization
    ablation_group.add_argument(
        '--lambda_logdet', type=float, default=1e-5,
        help='Logdet L2 regularization weight (0=disabled, recommended: 1e-5 ~ 1e-4)'
    )

    # Spatial context mixing
    ablation_group.add_argument(
        '--use_spatial_context', action='store_true',
        help='Enable spatial context mixing before NF (3x3 depthwise conv)'
    )
    ablation_group.add_argument(
        '--spatial_context_mode', type=str, default='depthwise_residual',
        choices=['depthwise', 'depthwise_residual', 'local_stats', 'full'],
        help='Spatial context mixing mode'
    )
    ablation_group.add_argument(
        '--spatial_context_kernel', type=int, default=3,
        help='Kernel size for spatial context mixing (default: 3)'
    )

    # Scale-specific context injection (Baseline 1.5 - ENABLED by default)
    ablation_group.add_argument(
        '--use_scale_context', action='store_true', default=False,
        help='Enable context injection for s-network only (NOTE: enabled by default in config)'
    )
    ablation_group.add_argument(
        '--no_scale_context', action='store_true',
        help='Disable scale context injection (overrides default)'
    )
    ablation_group.add_argument(
        '--scale_context_kernel', type=int, default=3,
        help='Kernel size for scale context extraction (default: 3)'
    )
    ablation_group.add_argument(
        '--scale_context_init_scale', type=float, default=0.1,
        help='Initial alpha for context scaling (learnable, default: 0.1)'
    )
    ablation_group.add_argument(
        '--scale_context_max_alpha', type=float, default=0.2,
        help='Maximum alpha for context scaling (sigmoid upper bound, default: 0.2)'
    )

    # Patch-wise context gate (Baseline 2.0)
    ablation_group.add_argument(
        '--use_context_gate', action='store_true',
        help='Use patch-wise context gate instead of global alpha (Baseline 2.0)'
    )
    ablation_group.add_argument(
        '--context_gate_hidden', type=int, default=64,
        help='Hidden dimension for context gate MLP (default: 64)'
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

    # Apply adapter mode settings
    if hasattr(parsed_args, 'adapter_mode'):
        config.adapter_mode = parsed_args.adapter_mode
    if hasattr(parsed_args, 'soft_ln_init_scale'):
        config.soft_ln_init_scale = parsed_args.soft_ln_init_scale
    if hasattr(parsed_args, 'robust_gate_type'):
        config.robust_gate_type = parsed_args.robust_gate_type

    # Apply logdet regularization
    if hasattr(parsed_args, 'lambda_logdet'):
        config.lambda_logdet = parsed_args.lambda_logdet

    # Apply spatial context settings (legacy)
    if hasattr(parsed_args, 'use_spatial_context') and parsed_args.use_spatial_context:
        config.use_spatial_context = True
    if hasattr(parsed_args, 'spatial_context_mode'):
        config.spatial_context_mode = parsed_args.spatial_context_mode
    if hasattr(parsed_args, 'spatial_context_kernel'):
        config.spatial_context_kernel = parsed_args.spatial_context_kernel

    # Apply scale context settings (Baseline 1.5 - enabled by default)
    # Note: config.use_scale_context is True by default
    if hasattr(parsed_args, 'no_scale_context') and parsed_args.no_scale_context:
        config.use_scale_context = False  # Explicitly disable
    elif hasattr(parsed_args, 'use_scale_context') and parsed_args.use_scale_context:
        config.use_scale_context = True   # Explicitly enable
    # If neither flag is set, keep default (True)
    if hasattr(parsed_args, 'scale_context_kernel'):
        config.scale_context_kernel = parsed_args.scale_context_kernel
    if hasattr(parsed_args, 'scale_context_init_scale'):
        config.scale_context_init_scale = parsed_args.scale_context_init_scale
    if hasattr(parsed_args, 'scale_context_max_alpha'):
        config.scale_context_max_alpha = parsed_args.scale_context_max_alpha

    # Apply patch-wise context gate settings (Baseline 2.0)
    if hasattr(parsed_args, 'use_context_gate') and parsed_args.use_context_gate:
        config.use_context_gate = True
    if hasattr(parsed_args, 'context_gate_hidden'):
        config.context_gate_hidden = parsed_args.context_gate_hidden

    return config
