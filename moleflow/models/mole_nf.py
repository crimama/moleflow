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

from moleflow.models.lora import MoLESubnet, MoLEContextSubnet
from moleflow.models.adapters import FeatureStatistics, TaskInputAdapter, create_task_adapter, SpatialContextMixer

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
            self.adapter_mode = "standard"
            self.soft_ln_init_scale = 0.01
            self.robust_gate_type = "norm"
            self.use_spatial_context = False
            self.spatial_context_mode = "depthwise_residual"
            self.spatial_context_kernel = 3
            # Scale-specific context (Baseline 1.5 - RECOMMENDED DEFAULT)
            self.use_scale_context = True   # ENABLED by default
            self.scale_context_kernel = 3
            self.scale_context_init_scale = 0.1
            self.scale_context_max_alpha = 0.2
            # Patch-wise context gate (Baseline 2.0 - DISABLED)
            self.use_context_gate = False   # Gate hurts performance in unsupervised setting
            self.context_gate_hidden = 64
        else:
            self.use_lora = ablation_config.use_lora
            self.use_task_adapter = ablation_config.use_task_adapter
            self.use_task_bias = ablation_config.use_task_bias
            self.adapter_mode = ablation_config.adapter_mode
            self.soft_ln_init_scale = ablation_config.soft_ln_init_scale
            self.robust_gate_type = ablation_config.robust_gate_type
            self.use_spatial_context = ablation_config.use_spatial_context
            self.spatial_context_mode = ablation_config.spatial_context_mode
            self.spatial_context_kernel = ablation_config.spatial_context_kernel
            # Scale-specific context (Baseline 1.5 Improved)
            self.use_scale_context = ablation_config.use_scale_context
            self.scale_context_kernel = ablation_config.scale_context_kernel
            self.scale_context_init_scale = ablation_config.scale_context_init_scale
            self.scale_context_max_alpha = ablation_config.scale_context_max_alpha
            # Patch-wise context gate (Baseline 2.0)
            self.use_context_gate = ablation_config.use_context_gate
            self.context_gate_hidden = ablation_config.context_gate_hidden

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

        # Spatial Context Mixer (Baseline 1.5)
        # Shared across all tasks - provides local context for scale(s)
        if self.use_spatial_context:
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
                    context_max_alpha=self.scale_context_max_alpha,
                    use_context_gate=self.use_context_gate,
                    context_gate_hidden=self.context_gate_hidden
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
            gate_str = f",gate=patch(h={self.context_gate_hidden})" if self.use_context_gate else ",gate=global"
            ablation_info.append(f"ScaleContext=ON(k={self.scale_context_kernel},α_init={self.scale_context_init_scale},α_max={self.scale_context_max_alpha}{gate_str})")
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

            # NEW v3: Add InputAdapter for Task 0 as well (self-adaptation)
            # This helps Task 0 pixel-level performance by learning input feature adjustment
            if self.use_task_adapter:
                self.input_adapters[task_key] = create_task_adapter(
                    adapter_mode=self.adapter_mode,
                    channels=self.embed_dim,
                    reference_mean=None,  # No reference for Task 0 (it IS the reference)
                    reference_std=None,
                    task_id=task_id,
                    soft_ln_init_scale=self.soft_ln_init_scale,
                    robust_gate_type=self.robust_gate_type
                ).to(self.device)

            components = []
            if self.use_lora:
                components.append(f"LoRA (rank={self.lora_rank})")
            if self.use_task_bias:
                components.append("Task Biases")
            if self.use_task_adapter:
                adapter_info = f"Input Adapter ({self.adapter_mode})"
                if self.adapter_mode == "robust":
                    adapter_info += f" [gate={self.robust_gate_type}]"
                elif self.adapter_mode == "soft_ln":
                    adapter_info += f" [init_scale={self.soft_ln_init_scale}]"
                components.append(adapter_info)

            print(f"✅ Task {task_id}: Base weights trainable + {' + '.join(components) if components else 'no adapters'}")
            print(f"   📊 Feature statistics will be collected during training")
        else:
            # Task > 0: Freeze base, add LoRA adapters + task biases
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
                    soft_ln_init_scale=self.soft_ln_init_scale,
                    robust_gate_type=self.robust_gate_type
                ).to(self.device)

            # Print status
            components = []
            if self.use_lora:
                components.append(f"LoRA (rank={self.lora_rank})")
            if self.use_task_bias:
                components.append("Task Biases")
            if self.use_task_adapter:
                adapter_info = f"Input Adapter ({self.adapter_mode})"
                if self.adapter_mode == "robust":
                    adapter_info += f" [gate={self.robust_gate_type}]"
                elif self.adapter_mode == "soft_ln":
                    adapter_info += f" [init_scale={self.soft_ln_init_scale}]"
                elif self.adapter_mode == "no_ln_after_task0":
                    adapter_info += " [LN=OFF]"
                components.append(adapter_info)

            if components:
                print(f"✅ Task {task_id}: Base frozen, {' + '.join(components)} added")
            else:
                print(f"✅ Task {task_id}: Base frozen (no task-specific components - ablation mode)")

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

                # MoLEContextSubnet: context_conv and context gating parameters
                if hasattr(subnet, 'context_conv'):
                    params.extend(subnet.context_conv.parameters())
                if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                    params.append(subnet.context_scale_param)
                if hasattr(subnet, 'context_gate_net') and subnet.context_gate_net is not None:
                    params.extend(subnet.context_gate_net.parameters())

            # Include InputAdapter parameters for Task 0
            if task_key in self.input_adapters:
                params.extend(self.input_adapters[task_key].parameters())
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

                # MoLEContextSubnet: context_conv and context gating parameters
                # (shared across tasks but still trainable)
                if hasattr(subnet, 'context_conv'):
                    params.extend(subnet.context_conv.parameters())
                if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                    params.append(subnet.context_scale_param)
                if hasattr(subnet, 'context_gate_net') and subnet.context_gate_net is not None:
                    params.extend(subnet.context_gate_net.parameters())

            # Input adapter parameters
            if task_key in self.input_adapters:
                params.extend(self.input_adapters[task_key].parameters())

        # Spatial mixer parameters (shared, trained with each task)
        if self.spatial_mixer is not None:
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

            # MoLEContextSubnet: context_conv and context gating parameters
            if hasattr(subnet, 'context_conv'):
                params.extend(subnet.context_conv.parameters())
            if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                params.append(subnet.context_scale_param)
            if hasattr(subnet, 'context_gate_net') and subnet.context_gate_net is not None:
                params.extend(subnet.context_gate_net.parameters())

        # Input adapter parameters
        if task_key in self.input_adapters:
            params.extend(self.input_adapters[task_key].parameters())

        # Spatial mixer parameters (shared, trained with FAST stage)
        if self.spatial_mixer is not None:
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

            # MoLEContextSubnet: freeze context gating parameters
            if hasattr(subnet, 'context_conv'):
                for param in subnet.context_conv.parameters():
                    param.requires_grad = False
            if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                subnet.context_scale_param.requires_grad = False
            if hasattr(subnet, 'context_gate_net') and subnet.context_gate_net is not None:
                for param in subnet.context_gate_net.parameters():
                    param.requires_grad = False

        if task_key in self.input_adapters:
            for param in self.input_adapters[task_key].parameters():
                param.requires_grad = False

        # Freeze spatial mixer
        if self.spatial_mixer is not None:
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

            # MoLEContextSubnet: unfreeze context gating parameters
            if hasattr(subnet, 'context_conv'):
                for param in subnet.context_conv.parameters():
                    param.requires_grad = True
            if hasattr(subnet, 'context_scale_param') and subnet.context_scale_param is not None:
                subnet.context_scale_param.requires_grad = True
            if hasattr(subnet, 'context_gate_net') and subnet.context_gate_net is not None:
                for param in subnet.context_gate_net.parameters():
                    param.requires_grad = True

        if task_key in self.input_adapters:
            for param in self.input_adapters[task_key].parameters():
                param.requires_grad = True

        # Unfreeze spatial mixer
        if self.spatial_mixer is not None:
            for param in self.spatial_mixer.parameters():
                param.requires_grad = True

    # =========================================================================
    # Context Gate/Alpha Logging Methods
    # =========================================================================

    def get_context_info(self) -> dict:
        """
        Get context gate/alpha information for logging.

        Returns:
            dict with context statistics:
            - If using patch-wise gate: 'gate_mean', 'gate_std', 'gate_min', 'gate_max'
            - If using global alpha: 'alpha' value for each subnet
            - Empty dict if not using scale context
        """
        if not self.use_scale_context:
            return {}

        info = {}

        if self.use_context_gate:
            # Patch-wise gate mode: aggregate stats from all subnets
            gate_stats = []
            for i, subnet in enumerate(self.subnets):
                if hasattr(subnet, 'get_last_gate_stats'):
                    stats = subnet.get_last_gate_stats()
                    if stats is not None:
                        gate_stats.append(stats)

            if gate_stats:
                # Average across all subnets
                info['gate_mean'] = sum(s['mean'] for s in gate_stats) / len(gate_stats)
                info['gate_std'] = sum(s['std'] for s in gate_stats) / len(gate_stats)
                info['gate_min'] = min(s['min'] for s in gate_stats)
                info['gate_max'] = max(s['max'] for s in gate_stats)
        else:
            # Global alpha mode: collect alpha from all subnets
            alphas = []
            for i, subnet in enumerate(self.subnets):
                if hasattr(subnet, 'get_context_alpha'):
                    alpha = subnet.get_context_alpha()
                    if alpha is not None:
                        alphas.append(alpha)

            if alphas:
                info['alpha_mean'] = sum(alphas) / len(alphas)
                info['alpha_min'] = min(alphas)
                info['alpha_max'] = max(alphas)

        return info
