"""
MoLE-Flow: Spatial-Aware Normalizing Flow with LoRA Experts.

Mixture of LoRA Experts for Normalizing Flow with task-specific adaptation.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, TYPE_CHECKING

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from moleflow.models.lora import MoLESubnet, MoLEContextSubnet, LightweightMSContext, TaskConditionedMSContext, DeepInvertibleAdapter
from moleflow.models.adapters import FeatureStatistics, TaskInputAdapter, create_task_adapter, SpatialContextMixer, WhiteningAdapter

if TYPE_CHECKING:
    from moleflow.config.ablation import AblationConfig


class MoLESpatialAwareNF(nn.Module):
    """
    MoLE-Flow: Mixture of LoRA Experts for Normalizing Flow.

    Key Features:
    - Base NF weights shared across all tasks
    - Task-specific LoRA adapters for distribution shift
    - Task-specific biases for mean shift adaptation
    - Task-specific input adapters for feature normalization
    - Feature statistics alignment for cross-task consistency
    - Zero-initialization for stable adaptation

    Ablation Support:
    - use_lora: Enable/disable LoRA adaptation
    - use_task_adapter: Enable/disable task input adapters
    - use_task_bias: Enable/disable task-specific biases
    """

    def __init__(self,
                 embed_dim: int = 512,
                 coupling_layers: int = 8,
                 clamp_alpha: float = 1.9,
                 lora_rank: int = 32,
                 lora_alpha: float = 1.0,
                 device: str = 'cuda',
                 ablation_config: 'AblationConfig' = None):
        super(MoLESpatialAwareNF, self).__init__()

        self.embed_dim = embed_dim
        self.coupling_layers = coupling_layers
        self.clamp_alpha = clamp_alpha
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.device = device

        # Ablation configuration
        self.ablation_config = ablation_config
        if ablation_config is None:
            # Default: all components enabled
            self.use_lora = True
            self.use_task_adapter = True
            self.use_task_bias = True
            self.adapter_mode = "soft_ln"
            self.soft_ln_init_scale = 0.01
            self.use_spatial_context = True
            self.spatial_context_mode = "depthwise_residual"
            self.spatial_context_kernel = 3
            # Scale-specific context
            self.use_scale_context = True
            self.scale_context_kernel = 3
            self.scale_context_init_scale = 0.1
            self.scale_context_max_alpha = 0.2
            # V3 defaults (all disabled for backward compatibility)
            self.use_ms_context = False
            self.ms_context_dilations = (1, 2, 4)
            self.ms_context_use_regional = True
            self.use_whitening_adapter = False
            self.use_adaptive_unfreeze = False
            self.adaptive_unfreeze_ratio = 0.4
            # V3 DIA defaults
            self.use_dia = False
            self.dia_n_blocks = 2
            self.dia_hidden_ratio = 0.5
            # V3 Task-Conditioned MS Context defaults
            self.use_task_conditioned_ms_context = False
            self.tc_ms_context_dilations = (1, 2, 4)
            self.tc_ms_context_use_regional = True
            self.tc_ms_context_lora_rank = 16
        else:
            self.use_lora = ablation_config.use_lora
            self.use_task_adapter = ablation_config.use_task_adapter
            self.use_task_bias = ablation_config.use_task_bias
            self.adapter_mode = ablation_config.adapter_mode
            self.soft_ln_init_scale = ablation_config.soft_ln_init_scale
            self.use_spatial_context = ablation_config.use_spatial_context
            self.spatial_context_mode = ablation_config.spatial_context_mode
            self.spatial_context_kernel = ablation_config.spatial_context_kernel
            # Scale-specific context
            self.use_scale_context = ablation_config.use_scale_context
            self.scale_context_kernel = ablation_config.scale_context_kernel
            self.scale_context_init_scale = ablation_config.scale_context_init_scale
            self.scale_context_max_alpha = ablation_config.scale_context_max_alpha
            # V3 settings
            self.use_ms_context = ablation_config.use_ms_context
            self.ms_context_dilations = ablation_config.ms_context_dilations
            self.ms_context_use_regional = ablation_config.ms_context_use_regional
            self.use_whitening_adapter = ablation_config.use_whitening_adapter
            self.use_adaptive_unfreeze = ablation_config.use_adaptive_unfreeze
            self.adaptive_unfreeze_ratio = ablation_config.adaptive_unfreeze_ratio
            # V3 DIA settings
            self.use_dia = ablation_config.use_dia
            self.dia_n_blocks = ablation_config.dia_n_blocks
            self.dia_hidden_ratio = ablation_config.dia_hidden_ratio
            # V3 Task-Conditioned MS Context settings
            self.use_task_conditioned_ms_context = ablation_config.use_task_conditioned_ms_context
            self.tc_ms_context_dilations = ablation_config.tc_ms_context_dilations
            self.tc_ms_context_use_regional = ablation_config.tc_ms_context_use_regional
            self.tc_ms_context_lora_rank = ablation_config.tc_ms_context_lora_rank

        # Track tasks
        self.num_tasks = 0
        self.current_task_id: Optional[int] = None

        # Build flow
        self.subnets: List[MoLESubnet] = []
        self.flow = self._build_flow()

        # Feature statistics from Task 0 (reference distribution)
        self.reference_stats = FeatureStatistics(device=device)

        # Task-specific input adapters for pre-conditioning
        self.input_adapters = nn.ModuleDict()

        # V3: Deep Invertible Adapters (DIA) per task
        # Applied AFTER base NF for nonlinear manifold adaptation
        self.dia_adapters = nn.ModuleDict()

        # Spatial Context Mixer (Baseline 1.5) or MS-Context (V3)
        # Priority: TaskConditionedMSContext > LightweightMSContext > SpatialContextMixer
        if self.use_task_conditioned_ms_context:
            # V3 Improved: Task-Conditioned MS Context (fundamental solution)
            # - Shared base frozen after Task 0
            # - Task-specific adaptation via LoRA
            self.spatial_mixer = TaskConditionedMSContext(
                channels=embed_dim,
                dilations=self.tc_ms_context_dilations,
                use_regional=self.tc_ms_context_use_regional,
                regional_grid=4,
                lora_rank=self.tc_ms_context_lora_rank,
                lora_alpha=1.0
            )
            print(f"✅ [V3] TaskConditionedMSContext enabled: dilations={self.tc_ms_context_dilations}, "
                  f"regional={self.tc_ms_context_use_regional}, lora_rank={self.tc_ms_context_lora_rank}")
        elif self.use_ms_context:
            # V3: Lightweight Multi-Scale Context (supersedes SpatialContextMixer)
            # WARNING: This has catastrophic forgetting issues - use TaskConditionedMSContext instead
            self.spatial_mixer = LightweightMSContext(
                channels=embed_dim,
                dilations=self.ms_context_dilations,
                use_regional=self.ms_context_use_regional,
                regional_grid=4
            )
            print(f"⚠️  [V3] LightweightMSContext enabled: dilations={self.ms_context_dilations}, "
                  f"regional={self.ms_context_use_regional} (WARNING: May cause forgetting)")
        elif self.use_spatial_context:
            self.spatial_mixer = SpatialContextMixer(
                channels=embed_dim,
                mode=self.spatial_context_mode,
                kernel_size=self.spatial_context_kernel,
                learnable=True
            )
            print(f"✅ SpatialContextMixer enabled: mode={self.spatial_context_mode}, kernel={self.spatial_context_kernel}")
        else:
            self.spatial_mixer = None

        self.to(device)

    def _build_flow(self) -> Ff.SequenceINN:
        """Build normalizing flow with MoLE coupling blocks."""

        def make_subnet(dims_in, dims_out):
            # Use MoLEContextSubnet if scale context is enabled
            # This gives s-network local context while keeping t-network context-free
            if self.use_scale_context:
                subnet = MoLEContextSubnet(
                    dims_in, dims_out,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    use_lora=self.use_lora,
                    use_task_bias=self.use_task_bias,
                    context_kernel=self.scale_context_kernel,
                    context_init_scale=self.scale_context_init_scale,
                    context_max_alpha=self.scale_context_max_alpha
                )
            else:
                subnet = MoLESubnet(
                    dims_in, dims_out,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    use_lora=self.use_lora,
                    use_task_bias=self.use_task_bias
                )
            self.subnets.append(subnet)
            return subnet

        coder = Ff.SequenceINN(self.embed_dim)
        ablation_info = []
        if not self.use_lora:
            ablation_info.append("LoRA=OFF")
        if not self.use_task_adapter:
            ablation_info.append("TaskAdapter=OFF")
        if not self.use_task_bias:
            ablation_info.append("TaskBias=OFF")
        if self.use_scale_context:
            ablation_info.append(f"ScaleContext=ON(k={self.scale_context_kernel})")
        ablation_str = f" [{', '.join(ablation_info)}]" if ablation_info else ""
        print(f'MoLE-Flow => Embed Dim: {self.embed_dim}, LoRA Rank: {self.lora_rank}{ablation_str}')

        for k in range(self.coupling_layers):
            coder.append(
                Fm.AllInOneBlock,
                subnet_constructor=make_subnet,
                affine_clamping=self.clamp_alpha,
                global_affine_type='SOFTPLUS',
                permute_soft=True
            )

        return coder

    def add_task(self, task_id: int):
        """
        Add LoRA adapters and input adapter for a new task.

        NEW Design (Task 0 also uses LoRA):
        - Task 0: Train base weights + LoRA adapter (equal treatment)
        - Task > 0: Freeze base, add LoRA + task bias + input adapter

        Key Benefits:
        - Task 0 is no longer "special" - all tasks use LoRA for adaptation
        - Base NF learns general feature transformation
        - LoRA handles task-specific adaptation uniformly

        Respects ablation flags:
        - use_lora: If False, skip LoRA adapters
        - use_task_adapter: If False, skip input adapters
        - use_task_bias: If False, skip task-specific biases
        """
        task_key = str(task_id)

        if task_id == 0:
            # Task 0: Train base weights + LoRA adapter + InputAdapter (self-adaptation)
            for subnet in self.subnets:
                subnet.unfreeze_base()
                # NEW: Also add LoRA adapter for Task 0
                if self.use_lora or self.use_task_bias:
                    subnet.add_task_adapter(task_id)

            # Add InputAdapter for Task 0 as well (self-adaptation)
            # This helps Task 0 pixel-level performance by learning input feature adjustment
            if self.use_task_adapter:
                self.input_adapters[task_key] = create_task_adapter(
                    adapter_mode=self.adapter_mode,
                    channels=self.embed_dim,
                    reference_mean=None,  # No reference for Task 0 (it IS the reference)
                    reference_std=None,
                    task_id=task_id,
                    soft_ln_init_scale=self.soft_ln_init_scale
                ).to(self.device)

            # V3: Add DIA for Task 0 (all tasks get DIA for uniform treatment)
            if self.use_dia:
                self.dia_adapters[task_key] = DeepInvertibleAdapter(
                    channels=self.embed_dim,
                    task_id=task_id,
                    n_blocks=self.dia_n_blocks,
                    hidden_ratio=self.dia_hidden_ratio,
                    clamp_alpha=self.clamp_alpha
                ).to(self.device)

            components = []
            if self.use_lora:
                components.append(f"LoRA (rank={self.lora_rank})")
            if self.use_task_bias:
                components.append("Task Biases")
            if self.use_task_adapter:
                adapter_info = f"Input Adapter ({self.adapter_mode})"
                if self.adapter_mode == "soft_ln":
                    adapter_info += f" [init_scale={self.soft_ln_init_scale}]"
                components.append(adapter_info)
            if self.use_dia:
                components.append(f"DIA ({self.dia_n_blocks} blocks)")

            print(f"✅ Task {task_id}: Base weights trainable + {' + '.join(components) if components else 'no adapters'}")
            print(f"   📊 Feature statistics will be collected during training")
        else:
            # Task > 0: Freeze or partially unfreeze base, add LoRA adapters + task biases
            if self.use_adaptive_unfreeze:
                # V3: Adaptive unfreezing - unfreeze later blocks
                n_blocks = self.coupling_layers
                n_frozen = int(n_blocks * (1 - self.adaptive_unfreeze_ratio))

                for i, subnet in enumerate(self.subnets):
                    block_idx = i // 2  # Each block has ~2 subnets
                    if block_idx < n_frozen:
                        subnet.freeze_base()
                    else:
                        subnet.unfreeze_base()  # Allow later blocks to adapt

                    if self.use_lora or self.use_task_bias:
                        subnet.add_task_adapter(task_id)

                print(f"   🔓 [V3] Adaptive Unfreeze: {n_frozen}/{n_blocks} blocks frozen, {n_blocks - n_frozen} unfrozen")
            else:
                # Standard: Freeze all base weights
                for subnet in self.subnets:
                    subnet.freeze_base()
                    if self.use_lora or self.use_task_bias:
                        subnet.add_task_adapter(task_id)

            # Create input adapter with reference statistics
            if self.use_task_adapter:
                if self.reference_stats.is_initialized:
                    ref_mean, ref_std = self.reference_stats.get_reference_params()
                    print(f"   📊 Reference stats: mean_norm={ref_mean.norm():.4f}, std_mean={ref_std.mean():.4f}")
                else:
                    ref_mean, ref_std = None, None
                    print(f"   ⚠️  No reference stats available, using default initialization")

                self.input_adapters[task_key] = create_task_adapter(
                    adapter_mode=self.adapter_mode,
                    channels=self.embed_dim,
                    reference_mean=ref_mean,
                    reference_std=ref_std,
                    task_id=task_id,
                    soft_ln_init_scale=self.soft_ln_init_scale
                ).to(self.device)

            # V3: Add DIA for Task > 0 (nonlinear manifold adaptation)
            if self.use_dia:
                self.dia_adapters[task_key] = DeepInvertibleAdapter(
                    channels=self.embed_dim,
                    task_id=task_id,
                    n_blocks=self.dia_n_blocks,
                    hidden_ratio=self.dia_hidden_ratio,
                    clamp_alpha=self.clamp_alpha
                ).to(self.device)
                print(f"   🔄 [V3] DIA added: {self.dia_n_blocks} coupling blocks")

            # Print status
            components = []
            if self.use_lora:
                components.append(f"LoRA (rank={self.lora_rank})")
            if self.use_task_bias:
                components.append("Task Biases")
            if self.use_task_adapter:
                adapter_info = f"Input Adapter ({self.adapter_mode})"
                if self.adapter_mode == "soft_ln":
                    adapter_info += f" [init_scale={self.soft_ln_init_scale}]"
                elif self.adapter_mode == "no_ln_after_task0":
                    adapter_info += " [LN=OFF]"
                components.append(adapter_info)
            if self.use_dia:
                components.append(f"DIA ({self.dia_n_blocks} blocks)")

            if components:
                print(f"✅ Task {task_id}: Base frozen, {' + '.join(components)} added")
            else:
                print(f"✅ Task {task_id}: Base frozen (no task-specific components - ablation mode)")

        # V3: Add task to TaskConditionedMSContext (handles its own freeze/unfreeze)
        if self.use_task_conditioned_ms_context and self.spatial_mixer is not None:
            self.spatial_mixer.add_task(task_id)

        self.num_tasks = task_id + 1
        self.set_active_task(task_id)

    def update_reference_stats(self, features: torch.Tensor):
        """Update reference statistics (call during Task 0 training)."""
        self.reference_stats.update(features)

    def finalize_reference_stats(self):
        """Finalize reference statistics after Task 0 training."""
        self.reference_stats.finalize()

    def set_active_task(self, task_id: Optional[int]):
        """Set the currently active LoRA adapter."""
        self.current_task_id = task_id

        # Set active task for TaskConditionedMSContext
        if self.use_task_conditioned_ms_context and self.spatial_mixer is not None:
            self.spatial_mixer.set_active_task(task_id)

        if task_id is None:
            for subnet in self.subnets:
                subnet.set_active_task(None)
        else:
            # All tasks (including Task 0) now use LoRA
            for subnet in self.subnets:
                subnet.set_active_task(task_id)

    def _get_subnet_layers(self, subnet):
        """
        Get all LoRALinear layers from a subnet.

        Handles both MoLESubnet (layer1, layer2) and
        MoLEContextSubnet (s_layer1, s_layer2, t_layer1, t_layer2).
        """
        if hasattr(subnet, 's_layer1'):
            # MoLEContextSubnet: 4 layers (s_layer1, s_layer2, t_layer1, t_layer2)
            return [subnet.s_layer1, subnet.s_layer2, subnet.t_layer1, subnet.t_layer2]
        else:
            # MoLESubnet: 2 layers (layer1, layer2)
            return [subnet.layer1, subnet.layer2]

    def get_trainable_params(self, task_id: int):
        """Get trainable parameters for a specific task."""
        params = []
        task_key = str(task_id)

        if task_id == 0:
            # Task 0: Base parameters + LoRA parameters + InputAdapter
            for subnet in self.subnets:
                layers = self._get_subnet_layers(subnet)

                # Base parameters
                for layer in layers:
                    params.extend(layer.base_linear.parameters())

                # LoRA parameters for Task 0
                for layer in layers:
                    if task_key in layer.lora_A:
                        params.append(layer.lora_A[task_key])
                        params.append(layer.lora_B[task_key])

                # Task-specific biases for Task 0
                for layer in layers:
                    if task_key in layer.task_biases:
                        params.append(layer.task_biases[task_key])

                # MoLEContextSubnet: context_conv and context scaling parameters
                if hasattr(subnet, 'context_conv'):
                    params.extend(subnet.context_conv.parameters())
                if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                    params.append(subnet.context_scale_param)

            # Include InputAdapter parameters for Task 0
            if task_key in self.input_adapters:
                params.extend(self.input_adapters[task_key].parameters())

            # V3: DIA parameters for Task 0
            if task_key in self.dia_adapters:
                params.extend(self.dia_adapters[task_key].parameters())
        else:
            # Task > 0: LoRA parameters + Task Biases + Input Adapter
            for subnet in self.subnets:
                layers = self._get_subnet_layers(subnet)

                # LoRA A, B matrices
                for layer in layers:
                    if task_key in layer.lora_A:
                        params.append(layer.lora_A[task_key])
                        params.append(layer.lora_B[task_key])

                # Task-specific biases
                for layer in layers:
                    if task_key in layer.task_biases:
                        params.append(layer.task_biases[task_key])

                # V4.1: MoLEContextSubnet context parameters are frozen for Task > 0
                # They are only trained in Task 0 (see the task_id == 0 block above)

            # Input adapter parameters
            if task_key in self.input_adapters:
                params.extend(self.input_adapters[task_key].parameters())

            # V3: DIA parameters for Task > 0
            if task_key in self.dia_adapters:
                params.extend(self.dia_adapters[task_key].parameters())

        # Spatial mixer parameters
        if self.spatial_mixer is not None:
            if self.use_task_conditioned_ms_context:
                # TaskConditionedMSContext: Use its own get_trainable_params
                # This properly handles shared vs task-specific params
                params.extend(self.spatial_mixer.get_trainable_params(task_id))
            else:
                # Other spatial mixers (SpatialContextMixer, LightweightMSContext):
                # Train all params with each task (may cause forgetting for LightweightMSContext)
                params.extend(self.spatial_mixer.parameters())

        return params

    def forward(self, patch_embeddings_with_pos: torch.Tensor, reverse: bool = False):
        """
        Forward or inverse transformation with task-specific input pre-conditioning.

        Key Design:
        - Task 0: No pre-conditioning (base model)
        - Task 1+: TaskInputAdapter handles all distribution alignment (if enabled)

        Respects ablation flags:
        - use_task_adapter: If False, skip input adapter

        Args:
            patch_embeddings_with_pos: (B, H, W, D) spatial patch embeddings
            reverse: If True, generate samples from latent

        Returns:
            z: (B, H, W, D) latent or reconstructed embeddings
            logdet_patch: (B, H, W) patch-wise log determinant of Jacobian
        """
        B, H, W, D = patch_embeddings_with_pos.shape
        x = patch_embeddings_with_pos

        # Apply task-specific input adapter for ALL tasks (if enabled)
        # v3: Task 0 also uses InputAdapter for self-adaptation
        if self.use_task_adapter and not reverse and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.input_adapters:
                x = self.input_adapters[task_key](x)

        # Apply spatial context mixing (Baseline 1.5 - Legacy)
        # This allows scale(s) to see local context: "abnormal compared to neighbors?"
        if self.spatial_mixer is not None and not reverse:
            x = self.spatial_mixer(x)

        # Flatten spatial dimensions
        x_flat = x.reshape(B * H * W, D)

        # Set spatial info for MoLEContextSubnet (Baseline 1.5 Improved)
        # This allows the context-aware s-network to reshape features for 3x3 conv
        if self.use_scale_context:
            MoLEContextSubnet._spatial_info = (B, H, W)

        # Flow transformation
        if not reverse:
            z_flat, log_jac_det_flat = self.flow(x_flat)
        else:
            z_flat, log_jac_det_flat = self.flow(x_flat, rev=True)

        # Clear spatial info after forward pass
        if self.use_scale_context:
            MoLEContextSubnet._spatial_info = None

        # FrEIA returns (N,) or (N,1) - ensure flat shape
        log_jac_det_flat = log_jac_det_flat.reshape(-1)

        # Keep patch-wise log_det: (BHW,) -> (B, H, W)
        logdet_patch = log_jac_det_flat.reshape(B, H, W)

        # Reshape back to spatial
        z = z_flat.reshape(B, H, W, D)

        # V3: Apply DIA (Deep Invertible Adapter) if enabled
        # DIA is applied AFTER base NF for nonlinear manifold adaptation
        if self.use_dia and self.current_task_id is not None:
            task_key = str(self.current_task_id)
            if task_key in self.dia_adapters:
                dia = self.dia_adapters[task_key]
                z, dia_logdet = dia(z, reverse=reverse)
                logdet_patch = logdet_patch + dia_logdet

        return z, logdet_patch

    def log_prob(self, patch_embeddings_with_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of patch embeddings (image-level).

        Args:
            patch_embeddings_with_pos: (B, H, W, D)

        Returns:
            log_prob: (B,) log probability for each image
        """
        B, H, W, D = patch_embeddings_with_pos.shape

        # Forward transformation - now returns patch-wise logdet
        z, logdet_patch = self.forward(patch_embeddings_with_pos, reverse=False)

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch

        # Image-level log-likelihood: sum over all patches
        log_px_image = log_px_patch.sum(dim=(1, 2))  # (B,)

        return log_px_image

    def log_prob_patch(self, patch_embeddings_with_pos: torch.Tensor) -> torch.Tensor:
        """
        Compute patch-wise log probability (for anomaly detection).

        Args:
            patch_embeddings_with_pos: (B, H, W, D)

        Returns:
            log_prob_patch: (B, H, W) log probability for each patch
        """
        B, H, W, D = patch_embeddings_with_pos.shape

        # Forward transformation - returns patch-wise logdet
        z, logdet_patch = self.forward(patch_embeddings_with_pos, reverse=False)

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch

        return log_px_patch

    # =========================================================================
    # Slow-Fast Stage Methods for Stage 2 (SLOW Consolidation)
    # =========================================================================

    def unfreeze_last_k_blocks(self, k: int = 2):
        """
        Unfreeze only the last K coupling blocks for slow consolidation.

        This allows gradual generalization of the base NF while protecting
        earlier blocks that encode fundamental transformations.

        Args:
            k: Number of last coupling blocks to unfreeze (default: 2)
        """
        n_subnets = len(self.subnets)
        n_blocks = self.coupling_layers

        # Each coupling block has 2 subnets (s and t networks)
        subnets_per_block = n_subnets // n_blocks
        last_k_subnets = k * subnets_per_block

        # Freeze all first
        for subnet in self.subnets:
            subnet.freeze_base()

        # Unfreeze last k blocks
        for i in range(n_subnets - last_k_subnets, n_subnets):
            self.subnets[i].unfreeze_base()

        unfrozen_count = sum(
            1 for subnet in self.subnets
            if not self._get_subnet_layers(subnet)[0].base_frozen
        )
        print(f"   Unfroze last {k} blocks ({unfrozen_count} subnets)")

    def freeze_all_base(self):
        """Freeze all base parameters (for Stage 1 FAST)."""
        for subnet in self.subnets:
            subnet.freeze_base()

    def get_base_params_for_slow_update(self, k: int = 2) -> List:
        """
        Get base parameters from last K blocks for slow update.

        Args:
            k: Number of last coupling blocks

        Returns:
            List of base parameters to update
        """
        params = []
        n_subnets = len(self.subnets)
        n_blocks = self.coupling_layers
        subnets_per_block = n_subnets // n_blocks
        last_k_subnets = k * subnets_per_block

        for i in range(n_subnets - last_k_subnets, n_subnets):
            subnet = self.subnets[i]
            layers = self._get_subnet_layers(subnet)
            for layer in layers:
                params.extend(layer.base_linear.parameters())

        return params

    def get_fast_params(self, task_id: int) -> List:
        """
        Get FAST adaptation parameters (LoRA + InputAdapter).

        These are trained in Stage 1 (FAST Adaptation).
        """
        params = []
        task_key = str(task_id)

        for subnet in self.subnets:
            layers = self._get_subnet_layers(subnet)

            # LoRA A, B matrices
            for layer in layers:
                if task_key in layer.lora_A:
                    params.append(layer.lora_A[task_key])
                    params.append(layer.lora_B[task_key])

            # Task-specific biases
            for layer in layers:
                if task_key in layer.task_biases:
                    params.append(layer.task_biases[task_key])

            # V4.1: MoLEContextSubnet context parameters - only trained in Task 0
            # context_conv and context_scale_param are shared across tasks
            # Freezing after Task 0 prevents representation drift
            if task_id == 0:
                if hasattr(subnet, 'context_conv'):
                    params.extend(subnet.context_conv.parameters())
                if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                    params.append(subnet.context_scale_param)

        # Input adapter parameters
        if task_key in self.input_adapters:
            params.extend(self.input_adapters[task_key].parameters())

        # V3: DIA parameters (part of FAST stage)
        if task_key in self.dia_adapters:
            params.extend(self.dia_adapters[task_key].parameters())

        # V4 Complete Separation: Spatial mixer only trained in Task 0
        # After Task 0, SpatialMixer is frozen to prevent representation drift
        # This fixes the catastrophic forgetting issue where shared SpatialMixer
        # would drift during subsequent task training, corrupting Task 0 representations
        if self.spatial_mixer is not None and task_id == 0:
            params.extend(self.spatial_mixer.parameters())

        return params

    def freeze_fast_params(self, task_id: int):
        """Freeze FAST parameters (for Stage 2)."""
        task_key = str(task_id)

        for subnet in self.subnets:
            layers = self._get_subnet_layers(subnet)

            for layer in layers:
                if task_key in layer.lora_A:
                    layer.lora_A[task_key].requires_grad = False
                    layer.lora_B[task_key].requires_grad = False
                if task_key in layer.task_biases:
                    layer.task_biases[task_key].requires_grad = False

            # V4.1: MoLEContextSubnet context parameters - only freeze for Task 0
            if task_id == 0:
                if hasattr(subnet, 'context_conv'):
                    for param in subnet.context_conv.parameters():
                        param.requires_grad = False
                if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                    subnet.context_scale_param.requires_grad = False

        if task_key in self.input_adapters:
            for param in self.input_adapters[task_key].parameters():
                param.requires_grad = False

        # V3: Freeze DIA parameters
        if task_key in self.dia_adapters:
            for param in self.dia_adapters[task_key].parameters():
                param.requires_grad = False

        # V4: Freeze spatial mixer only for Task 0 (it's already frozen for Task 1+)
        if self.spatial_mixer is not None and task_id == 0:
            for param in self.spatial_mixer.parameters():
                param.requires_grad = False

    def unfreeze_fast_params(self, task_id: int):
        """Unfreeze FAST parameters (after Stage 2, if needed)."""
        task_key = str(task_id)

        for subnet in self.subnets:
            layers = self._get_subnet_layers(subnet)

            for layer in layers:
                if task_key in layer.lora_A:
                    layer.lora_A[task_key].requires_grad = True
                    layer.lora_B[task_key].requires_grad = True
                if task_key in layer.task_biases:
                    layer.task_biases[task_key].requires_grad = True

            # V4.1: MoLEContextSubnet context parameters - only unfreeze for Task 0
            if task_id == 0:
                if hasattr(subnet, 'context_conv'):
                    for param in subnet.context_conv.parameters():
                        param.requires_grad = True
                if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                    subnet.context_scale_param.requires_grad = True

        if task_key in self.input_adapters:
            for param in self.input_adapters[task_key].parameters():
                param.requires_grad = True

        # V3: Unfreeze DIA parameters
        if task_key in self.dia_adapters:
            for param in self.dia_adapters[task_key].parameters():
                param.requires_grad = True

        # V4: Unfreeze spatial mixer only for Task 0 (it stays frozen for Task 1+)
        if self.spatial_mixer is not None and task_id == 0:
            for param in self.spatial_mixer.parameters():
                param.requires_grad = True

    # =========================================================================
    # Context Alpha Logging Methods
    # =========================================================================

    def get_context_info(self) -> dict:
        """
        Get context alpha information for logging.

        Returns:
            dict with 'alpha_mean', 'alpha_min', 'alpha_max' values
            or empty dict if not using scale context
        """
        if not self.use_scale_context:
            return {}

        # Collect alpha from all subnets
        alphas = []
        for subnet in self.subnets:
            if hasattr(subnet, 'get_context_alpha'):
                alpha = subnet.get_context_alpha()
                if alpha is not None:
                    alphas.append(alpha)

        if alphas:
            return {
                'alpha_mean': sum(alphas) / len(alphas),
                'alpha_min': min(alphas),
                'alpha_max': max(alphas)
            }

        return {}
