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
        "soft_ln": SoftLNTaskInputAdapter (default) - optional/weak LayerNorm for Task > 0
        "standard": Original FiLM-style TaskInputAdapter
        "no_ln_after_task0": Disable LN for Task > 0 (quick ablation)
        "whitening": V3 WhiteningAdapter - Whitening + constrained de-whitening

    V3 Improvements (--use_* flags):
        use_whitening_adapter: Use WhiteningAdapter instead of SoftLN
        use_ms_context: Use LightweightMSContext (multi-scale dilated context)
        use_feature_bank: Use FeatureBank for replay-based continual learning
        use_distillation: Use knowledge distillation from teacher model
        use_ewc: Use Elastic Weight Consolidation
        use_hybrid_routing: Use TwoStageHybridRouter with likelihood
        use_adaptive_unfreeze: Unfreeze later blocks for Task > 0
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
    adapter_mode: str = "soft_ln"  # "soft_ln", "standard", "no_ln_after_task0"
    soft_ln_init_scale: float = 0.01  # Initial scale for soft LN (used when adapter_mode="soft_ln")

    # Flow scale(s) regularization
    # Regularizes log_det to reduce variance and stabilize image-level scores
    # loss = nll_loss + lambda_logdet * (logdet_patch ** 2).mean()
    lambda_logdet: float = 3e-5  # Recommended: 3e-5 (stabilizes logdet without hurting performance)

    # Spatial Context Mixing
    # Adds local context to features before NF, so scale(s) can detect local contrast
    use_spatial_context: bool = True
    spatial_context_mode: str = "depthwise_residual"  # "depthwise", "depthwise_residual", "local_stats", "full"
    spatial_context_kernel: int = 3  # Kernel size for context mixing

    # ==========================================================================
    # Scale-specific Context Injection
    # ==========================================================================
    # Injects 3x3 local context into s-network ONLY, keeping t-network context-free
    # - scale(s) sees local contrast → anomaly-sensitive
    # - shift(t) sees x only → density-preserving
    use_scale_context: bool = True
    scale_context_kernel: int = 3    # 3x3 depthwise conv for local context
    scale_context_init_scale: float = 0.1  # Initial alpha (learnable)
    scale_context_max_alpha: float = 0.2   # Upper bound via sigmoid

    # ==========================================================================
    # V3 Improvements: New Modules (all disabled by default for backward compat)
    # ==========================================================================

    # Solution 3: Whitening-based Distribution Adapter
    # Replaces SoftLN with Whitening + constrained de-whitening
    use_whitening_adapter: bool = False

    # Solution 2: Multi-Scale Context (Lightweight dilated convolutions)
    # Expands receptive field from 3×3 to 9×9 with attention-weighted fusion
    use_ms_context: bool = False
    ms_context_dilations: tuple = (1, 2, 4)  # Dilation rates
    ms_context_use_regional: bool = True     # Include regional pooling

    # Solution 1: Feature Bank for Replay
    # Stores representative features from previous tasks
    use_feature_bank: bool = False
    feature_bank_size: int = 500             # Samples per task
    feature_bank_selection: str = 'random'   # 'random', 'kmeans', 'herding'

    # Solution 1: Knowledge Distillation
    # Distill knowledge from teacher (previous task model)
    use_distillation: bool = False
    distillation_mode: str = 'logprob'       # 'logprob', 'latent', 'combined'
    distillation_temperature: float = 2.0
    distillation_weight: float = 1.0         # λ_distill

    # Solution 1: Elastic Weight Consolidation
    # Protects important parameters from previous tasks
    use_ewc: bool = False
    ewc_lambda: float = 1000.0               # EWC regularization strength

    # Solution 1: Adaptive Unfreezing
    # Partially unfreeze base NF for Task > 0 (with EWC protection)
    use_adaptive_unfreeze: bool = False
    adaptive_unfreeze_ratio: float = 0.4     # Ratio of later blocks to unfreeze
    adaptive_unfreeze_lr_ratio: float = 0.1  # LR ratio for unfrozen base params

    # Solution 4: Two-Stage Hybrid Routing
    # Combines prototype and likelihood-based routing
    use_hybrid_routing: bool = False
    hybrid_routing_top_k: int = 2            # Top-K candidates for stage 2
    hybrid_routing_proto_weight: float = 0.6 # Weight for prototype vs likelihood

    # Solution 4: Regional Prototypes
    # Captures spatial structure for better discrimination
    use_regional_prototype: bool = False
    regional_prototype_grid: int = 4         # Grid size (4×4 = 16 regions)

    # ==========================================================================
    # V5 Score Aggregation (Patch → Image)
    # ==========================================================================
    # Controls how patch-level anomaly scores are aggregated to image-level score
    #
    # Modes:
    #   "percentile": Use p-th percentile (default, p=0.99)
    #   "top_k": Average of top-K highest scoring patches
    #   "top_k_percent": Average of top K% highest scoring patches
    #   "max": Maximum patch score
    #   "mean": Mean of all patch scores
    #   "adaptive": Per-class optimal percentile (requires validation set)
    score_aggregation_mode: str = "percentile"
    score_aggregation_percentile: float = 0.99    # For "percentile" mode
    score_aggregation_top_k: int = 10             # For "top_k" mode
    score_aggregation_top_k_percent: float = 0.05 # For "top_k_percent" mode (top 5%)

    # ==========================================================================
    # V3 No-Replay Solutions: DIA + OGP
    # ==========================================================================

    # Deep Invertible Adapter (DIA)
    # Adds a small task-specific flow after base NF for nonlinear manifold adaptation
    use_dia: bool = False
    dia_n_blocks: int = 2                    # Number of coupling blocks per DIA
    dia_hidden_ratio: float = 0.5            # Hidden dim = channels * hidden_ratio

    # Orthogonal Gradient Projection (OGP)
    # Projects gradients to null space of important subspaces from previous tasks
    use_ogp: bool = False
    ogp_threshold: float = 0.99              # Cumulative variance threshold for SVD
    ogp_max_rank: int = 50                   # Maximum rank per task per parameter
    ogp_n_samples: int = 300                 # Samples for gradient collection

    # ==========================================================================
    # V3 Task-Conditioned Multi-Scale Context (Fundamental Solution)
    # ==========================================================================
    # Replaces LightweightMSContext with proper task-specific adaptation
    # - Shared base (dilated convs, regional_proj base) frozen after Task 0
    # - Task-specific components: scale_attention, regional LoRA, fusion_gate
    # This solves the catastrophic forgetting issue of the original MSContext
    use_task_conditioned_ms_context: bool = False
    tc_ms_context_dilations: tuple = (1, 2, 4)  # Dilation rates for multi-scale
    tc_ms_context_use_regional: bool = True     # Use regional context pooling
    tc_ms_context_lora_rank: int = 16           # LoRA rank for regional_proj

    def __post_init__(self):
        """Validate configuration."""
        if not self.use_lora and self.use_task_bias:
            # Task bias requires LoRA
            self.use_task_bias = False

        # V3: If whitening adapter is enabled, update adapter_mode
        if self.use_whitening_adapter:
            self.adapter_mode = "whitening"

        # V3: WhiteningAdapter + MS-Context 조합 충돌 방지
        # 두 모듈 모두 입력 분포를 변환하여 함께 사용시 학습 불안정 발생
        if self.use_whitening_adapter and self.use_ms_context:
            print("⚠️  Warning: use_whitening_adapter + use_ms_context 조합은 학습 불안정을 유발할 수 있습니다.")
            print("   → MS-Context 비활성화 (scale_context가 이미 s-network에 local context 제공)")
            self.use_ms_context = False

        # V3: Adaptive unfreeze requires EWC or distillation for stability
        if self.use_adaptive_unfreeze and not (self.use_ewc or self.use_distillation or self.use_feature_bank):
            print("⚠️  Warning: use_adaptive_unfreeze without EWC/distillation/feature_bank may cause forgetting")

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

        # V3 components
        if self.use_whitening_adapter:
            components.append("WhiteningAdapter")
        if self.use_ms_context:
            components.append("MSContext")
        if self.use_feature_bank:
            components.append("FeatureBank")
        if self.use_distillation:
            components.append("Distillation")
        if self.use_ewc:
            components.append("EWC")
        if self.use_adaptive_unfreeze:
            components.append("AdaptiveUnfreeze")
        if self.use_hybrid_routing:
            components.append("HybridRouting")
        if self.use_regional_prototype:
            components.append("RegionalProto")
        if self.use_dia:
            components.append("DIA")
        if self.use_ogp:
            components.append("OGP")
        if self.use_task_conditioned_ms_context:
            components.append("TCMSContext")

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

        # V3 components - add as suffixes when enabled
        v3_parts = []
        if self.use_whitening_adapter:
            v3_parts.append("whitening")
        if self.use_ms_context:
            v3_parts.append("msctx")
        if self.use_feature_bank:
            v3_parts.append("fbank")
        if self.use_distillation:
            v3_parts.append("distill")
        if self.use_ewc:
            v3_parts.append("ewc")
        if self.use_adaptive_unfreeze:
            v3_parts.append("unfreeze")
        if self.use_hybrid_routing:
            v3_parts.append("hybrid")
        if self.use_regional_prototype:
            v3_parts.append("regional")
        if self.use_dia:
            v3_parts.append("dia")
        if self.use_ogp:
            v3_parts.append("ogp")
        if self.use_task_conditioned_ms_context:
            v3_parts.append("tcmsctx")

        if v3_parts:
            base_name += "_v3_" + "_".join(v3_parts)
        else:
            # Add adapter mode suffix if not soft_ln (default) and no V3 components
            if self.adapter_mode != "soft_ln":
                base_name += f"_{self.adapter_mode}"
            else:
                base_name += f"_soft_ln_s{self.soft_ln_init_scale}"

        # Add logdet regularization suffix if enabled
        if self.lambda_logdet > 0:
            base_name += f"_ldet{self.lambda_logdet:.0e}".replace("-0", "-").replace("+0", "")

        # Add spatial context suffix if enabled (and not using ms_context which supersedes it)
        if self.use_spatial_context and not self.use_ms_context:
            base_name += f"_sctx_{self.spatial_context_mode}"

        # Add scale context suffix if enabled
        if self.use_scale_context:
            base_name += f"_scalectx_k{self.scale_context_kernel}"

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
        if self.lambda_logdet > 0:
            lines.append(f"  Logdet Regularization: {self.lambda_logdet:.2e}")
        if self.use_spatial_context:
            lines.append(f"  Spatial Context: {self.spatial_context_mode} (kernel={self.spatial_context_kernel})")
        if self.use_scale_context:
            lines.append(f"  Scale Context: s-net only (kernel={self.scale_context_kernel}, init_alpha={self.scale_context_init_scale}, max_alpha={self.scale_context_max_alpha})")
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
        '--adapter_mode', type=str, default='soft_ln',
        choices=['soft_ln', 'standard', 'no_ln_after_task0'],
        help='Adapter mode: soft_ln (default, weak LN), standard (FiLM), no_ln_after_task0'
    )
    ablation_group.add_argument(
        '--soft_ln_init_scale', type=float, default=0.01,
        help='Initial scale for soft LN (used with --adapter_mode soft_ln)'
    )

    # Flow regularization
    ablation_group.add_argument(
        '--lambda_logdet', type=float, default=1e-5,
        help='Logdet L2 regularization weight (0=disabled, recommended: 1e-5 ~ 1e-4)'
    )

    # Spatial context mixing (enabled by default)
    ablation_group.add_argument(
        '--no_spatial_context', action='store_true',
        help='Disable spatial context mixing before NF'
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

    # Scale-specific context injection (enabled by default)
    ablation_group.add_argument(
        '--no_scale_context', action='store_true',
        help='Disable scale context injection'
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

    # =========================================================================
    # V3 Improvements (--use_* flags for ablation study)
    # =========================================================================
    v3_group = parser.add_argument_group('V3 Improvements (Ablation Study)')

    # Solution 3: Whitening Adapter
    v3_group.add_argument(
        '--use_whitening_adapter', action='store_true',
        help='[V3] Use WhiteningAdapter instead of SoftLN (Solution 3)'
    )

    # Solution 2: Multi-Scale Context
    v3_group.add_argument(
        '--use_ms_context', action='store_true',
        help='[V3] Use LightweightMSContext with dilated convolutions (Solution 2)'
    )
    v3_group.add_argument(
        '--ms_context_dilations', type=str, default='1,2,4',
        help='Dilation rates for MS context (comma-separated, default: 1,2,4)'
    )

    # Solution 1: Feature Bank
    v3_group.add_argument(
        '--use_feature_bank', action='store_true',
        help='[V3] Use FeatureBank for replay-based continual learning (Solution 1)'
    )
    v3_group.add_argument(
        '--feature_bank_size', type=int, default=500,
        help='Number of features to store per task (default: 500)'
    )
    v3_group.add_argument(
        '--feature_bank_selection', type=str, default='random',
        choices=['random', 'kmeans', 'herding'],
        help='Feature selection method for bank (default: random)'
    )

    # Solution 1: Distillation
    v3_group.add_argument(
        '--use_distillation', action='store_true',
        help='[V3] Use knowledge distillation from teacher model (Solution 1)'
    )
    v3_group.add_argument(
        '--distillation_mode', type=str, default='logprob',
        choices=['logprob', 'latent', 'combined'],
        help='Distillation mode (default: logprob)'
    )
    v3_group.add_argument(
        '--distillation_weight', type=float, default=1.0,
        help='Weight for distillation loss (default: 1.0)'
    )

    # Solution 1: EWC
    v3_group.add_argument(
        '--use_ewc', action='store_true',
        help='[V3] Use Elastic Weight Consolidation (Solution 1)'
    )
    v3_group.add_argument(
        '--ewc_lambda', type=float, default=1000.0,
        help='EWC regularization strength (default: 1000)'
    )

    # Solution 1: Adaptive Unfreezing
    v3_group.add_argument(
        '--use_adaptive_unfreeze', action='store_true',
        help='[V3] Partially unfreeze base NF for Task > 0 (Solution 1)'
    )
    v3_group.add_argument(
        '--adaptive_unfreeze_ratio', type=float, default=0.4,
        help='Ratio of later blocks to unfreeze (default: 0.4)'
    )

    # Solution 4: Hybrid Routing
    v3_group.add_argument(
        '--use_hybrid_routing', action='store_true',
        help='[V3] Use Two-Stage Hybrid Router with likelihood (Solution 4)'
    )
    v3_group.add_argument(
        '--hybrid_routing_top_k', type=int, default=2,
        help='Top-K candidates for stage 2 routing (default: 2)'
    )

    # Solution 4: Regional Prototype
    v3_group.add_argument(
        '--use_regional_prototype', action='store_true',
        help='[V3] Use regional prototypes for spatial structure (Solution 4)'
    )
    v3_group.add_argument(
        '--regional_prototype_grid', type=int, default=4,
        help='Grid size for regional prototypes (default: 4 → 16 regions)'
    )

    # =========================================================================
    # V3 No-Replay Solutions: DIA + OGP
    # =========================================================================
    noreplay_group = parser.add_argument_group('V3 No-Replay Solutions (DIA/OGP)')

    # Deep Invertible Adapter (DIA)
    noreplay_group.add_argument(
        '--use_dia', action='store_true',
        help='[V3] Use Deep Invertible Adapter for nonlinear manifold adaptation'
    )
    noreplay_group.add_argument(
        '--dia_n_blocks', type=int, default=2,
        help='Number of coupling blocks per DIA (default: 2)'
    )
    noreplay_group.add_argument(
        '--dia_hidden_ratio', type=float, default=0.5,
        help='Hidden dim ratio for DIA (default: 0.5)'
    )

    # Orthogonal Gradient Projection (OGP)
    noreplay_group.add_argument(
        '--use_ogp', action='store_true',
        help='[V3] Use Orthogonal Gradient Projection for gradient-based protection'
    )
    noreplay_group.add_argument(
        '--ogp_threshold', type=float, default=0.99,
        help='Cumulative variance threshold for OGP SVD (default: 0.99)'
    )
    noreplay_group.add_argument(
        '--ogp_max_rank', type=int, default=50,
        help='Maximum rank per task per parameter (default: 50)'
    )
    noreplay_group.add_argument(
        '--ogp_n_samples', type=int, default=300,
        help='Number of samples for OGP gradient collection (default: 300)'
    )

    # =========================================================================
    # V3 Task-Conditioned Multi-Scale Context (Fundamental Solution)
    # =========================================================================
    tcms_group = parser.add_argument_group('V3 Task-Conditioned MS Context')

    tcms_group.add_argument(
        '--use_task_conditioned_ms_context', action='store_true',
        help='[V3] Use TaskConditionedMSContext (fixes MSContext forgetting issue)'
    )
    tcms_group.add_argument(
        '--tc_ms_context_dilations', type=str, default='1,2,4',
        help='Dilation rates for TC-MSContext (comma-separated, default: 1,2,4)'
    )
    tcms_group.add_argument(
        '--tc_ms_context_use_regional', action='store_true', default=True,
        help='Use regional context in TC-MSContext (default: True)'
    )
    tcms_group.add_argument(
        '--tc_ms_context_no_regional', action='store_true',
        help='Disable regional context in TC-MSContext'
    )
    tcms_group.add_argument(
        '--tc_ms_context_lora_rank', type=int, default=16,
        help='LoRA rank for regional_proj in TC-MSContext (default: 16)'
    )

    # =========================================================================
    # V5 Score Aggregation (Patch → Image)
    # =========================================================================
    score_group = parser.add_argument_group('V5 Score Aggregation')

    score_group.add_argument(
        '--score_aggregation_mode', type=str, default='percentile',
        choices=['percentile', 'top_k', 'top_k_percent', 'max', 'mean'],
        help='[V5] Score aggregation mode (default: percentile)'
    )
    score_group.add_argument(
        '--score_aggregation_percentile', type=float, default=0.99,
        help='Percentile for percentile mode (default: 0.99)'
    )
    score_group.add_argument(
        '--score_aggregation_top_k', type=int, default=10,
        help='K value for top_k mode (default: 10)'
    )
    score_group.add_argument(
        '--score_aggregation_top_k_percent', type=float, default=0.05,
        help='Percentage for top_k_percent mode (default: 0.05 = top 5%%)'
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

    # Apply logdet regularization
    if hasattr(parsed_args, 'lambda_logdet'):
        config.lambda_logdet = parsed_args.lambda_logdet

    # Apply spatial context settings (enabled by default)
    if hasattr(parsed_args, 'no_spatial_context') and parsed_args.no_spatial_context:
        config.use_spatial_context = False
    if hasattr(parsed_args, 'spatial_context_mode'):
        config.spatial_context_mode = parsed_args.spatial_context_mode
    if hasattr(parsed_args, 'spatial_context_kernel'):
        config.spatial_context_kernel = parsed_args.spatial_context_kernel

    # Apply scale context settings (enabled by default)
    if hasattr(parsed_args, 'no_scale_context') and parsed_args.no_scale_context:
        config.use_scale_context = False
    if hasattr(parsed_args, 'scale_context_kernel'):
        config.scale_context_kernel = parsed_args.scale_context_kernel
    if hasattr(parsed_args, 'scale_context_init_scale'):
        config.scale_context_init_scale = parsed_args.scale_context_init_scale
    if hasattr(parsed_args, 'scale_context_max_alpha'):
        config.scale_context_max_alpha = parsed_args.scale_context_max_alpha

    # =========================================================================
    # V3 Improvements
    # =========================================================================

    # Solution 3: Whitening Adapter
    # NOTE: Must also set adapter_mode because __post_init__ was already called during config creation
    if hasattr(parsed_args, 'use_whitening_adapter') and parsed_args.use_whitening_adapter:
        config.use_whitening_adapter = True
        config.adapter_mode = "whitening"  # Critical: __post_init__ won't run again

    # Solution 2: Multi-Scale Context
    if hasattr(parsed_args, 'use_ms_context') and parsed_args.use_ms_context:
        config.use_ms_context = True
    if hasattr(parsed_args, 'ms_context_dilations') and parsed_args.ms_context_dilations:
        # Parse comma-separated string to tuple
        dilations = tuple(int(d.strip()) for d in parsed_args.ms_context_dilations.split(','))
        config.ms_context_dilations = dilations

    # Solution 1: Feature Bank
    if hasattr(parsed_args, 'use_feature_bank') and parsed_args.use_feature_bank:
        config.use_feature_bank = True
    if hasattr(parsed_args, 'feature_bank_size'):
        config.feature_bank_size = parsed_args.feature_bank_size
    if hasattr(parsed_args, 'feature_bank_selection'):
        config.feature_bank_selection = parsed_args.feature_bank_selection

    # Solution 1: Distillation
    if hasattr(parsed_args, 'use_distillation') and parsed_args.use_distillation:
        config.use_distillation = True
    if hasattr(parsed_args, 'distillation_mode'):
        config.distillation_mode = parsed_args.distillation_mode
    if hasattr(parsed_args, 'distillation_weight'):
        config.distillation_weight = parsed_args.distillation_weight

    # Solution 1: EWC
    if hasattr(parsed_args, 'use_ewc') and parsed_args.use_ewc:
        config.use_ewc = True
    if hasattr(parsed_args, 'ewc_lambda'):
        config.ewc_lambda = parsed_args.ewc_lambda

    # Solution 1: Adaptive Unfreezing
    if hasattr(parsed_args, 'use_adaptive_unfreeze') and parsed_args.use_adaptive_unfreeze:
        config.use_adaptive_unfreeze = True
    if hasattr(parsed_args, 'adaptive_unfreeze_ratio'):
        config.adaptive_unfreeze_ratio = parsed_args.adaptive_unfreeze_ratio

    # Solution 4: Hybrid Routing
    if hasattr(parsed_args, 'use_hybrid_routing') and parsed_args.use_hybrid_routing:
        config.use_hybrid_routing = True
    if hasattr(parsed_args, 'hybrid_routing_top_k'):
        config.hybrid_routing_top_k = parsed_args.hybrid_routing_top_k

    # Solution 4: Regional Prototype
    if hasattr(parsed_args, 'use_regional_prototype') and parsed_args.use_regional_prototype:
        config.use_regional_prototype = True
    if hasattr(parsed_args, 'regional_prototype_grid'):
        config.regional_prototype_grid = parsed_args.regional_prototype_grid

    # =========================================================================
    # V3 No-Replay Solutions: DIA + OGP
    # =========================================================================

    # Deep Invertible Adapter (DIA)
    if hasattr(parsed_args, 'use_dia') and parsed_args.use_dia:
        config.use_dia = True
    if hasattr(parsed_args, 'dia_n_blocks'):
        config.dia_n_blocks = parsed_args.dia_n_blocks
    if hasattr(parsed_args, 'dia_hidden_ratio'):
        config.dia_hidden_ratio = parsed_args.dia_hidden_ratio

    # Orthogonal Gradient Projection (OGP)
    if hasattr(parsed_args, 'use_ogp') and parsed_args.use_ogp:
        config.use_ogp = True
    if hasattr(parsed_args, 'ogp_threshold'):
        config.ogp_threshold = parsed_args.ogp_threshold
    if hasattr(parsed_args, 'ogp_max_rank'):
        config.ogp_max_rank = parsed_args.ogp_max_rank
    if hasattr(parsed_args, 'ogp_n_samples'):
        config.ogp_n_samples = parsed_args.ogp_n_samples

    # =========================================================================
    # V3 Task-Conditioned Multi-Scale Context
    # =========================================================================
    if hasattr(parsed_args, 'use_task_conditioned_ms_context') and parsed_args.use_task_conditioned_ms_context:
        config.use_task_conditioned_ms_context = True
    if hasattr(parsed_args, 'tc_ms_context_dilations') and parsed_args.tc_ms_context_dilations:
        # Parse comma-separated string to tuple
        dilations = tuple(int(d.strip()) for d in parsed_args.tc_ms_context_dilations.split(','))
        config.tc_ms_context_dilations = dilations
    if hasattr(parsed_args, 'tc_ms_context_no_regional') and parsed_args.tc_ms_context_no_regional:
        config.tc_ms_context_use_regional = False
    if hasattr(parsed_args, 'tc_ms_context_lora_rank'):
        config.tc_ms_context_lora_rank = parsed_args.tc_ms_context_lora_rank

    # =========================================================================
    # V5 Score Aggregation
    # =========================================================================
    if hasattr(parsed_args, 'score_aggregation_mode'):
        config.score_aggregation_mode = parsed_args.score_aggregation_mode
    if hasattr(parsed_args, 'score_aggregation_percentile'):
        config.score_aggregation_percentile = parsed_args.score_aggregation_percentile
    if hasattr(parsed_args, 'score_aggregation_top_k'):
        config.score_aggregation_top_k = parsed_args.score_aggregation_top_k
    if hasattr(parsed_args, 'score_aggregation_top_k_percent'):
        config.score_aggregation_top_k_percent = parsed_args.score_aggregation_top_k_percent

    # Re-run __post_init__ to apply all validation/conflict resolution logic
    # This is necessary because __post_init__ was called when config was created (with default values)
    # but we've now modified the config with parsed arguments
    config.__post_init__()

    return config
