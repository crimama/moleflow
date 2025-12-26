"""
MoLE-Flow Continual Learning Trainer.

Implements Stage-Separated Training:
- Stage 1 (FAST): LoRA + InputAdapter adaptation (base frozen)
- Stage 2 (SLOW): Base NF consolidation (optional)

Ablation Support:
- use_lora: Enable/disable LoRA adaptation
- use_router: Enable/disable prototype router
- use_pos_embedding: Enable/disable positional embedding
- use_task_adapter: Enable/disable task input adapters

Optimizer: AdamP (Adaptive Momentum with Decoupled Weight Decay)
- Better generalization than AdamW through projection-based gradient correction
- Hyperparameters tuned for normalizing flow training
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, TYPE_CHECKING

try:
    from adamp import AdamP
except ImportError:
    # Fallback to AdamW if adamp is not installed
    AdamP = None
    print("Warning: adamp not installed. Install with: pip install adamp")

from moleflow.models.mole_nf import MoLESpatialAwareNF
from moleflow.models.routing import TaskPrototype, PrototypeRouter
from moleflow.utils.logger import TrainingLogger

if TYPE_CHECKING:
    from moleflow.config.ablation import AblationConfig


# AdamP hyperparameters for MoLE-Flow
ADAMP_CONFIG = {
    'betas': (0.9, 0.999),      # Momentum coefficients
    'weight_decay': 0.01,        # Decoupled weight decay (higher than AdamW default)
    'delta': 0.1,                # Threshold for gradient projection
    'wd_ratio': 0.1,             # Weight decay ratio for decoupled WD
    'nesterov': False,           # Nesterov momentum
}


def create_optimizer(params, lr: float, weight_decay_scale: float = 1.0):
    """
    Create optimizer for training.

    Uses AdamP if available, otherwise falls back to AdamW.

    Args:
        params: Model parameters to optimize
        lr: Learning rate
        weight_decay_scale: Scale factor for weight decay (default 1.0)

    Returns:
        Optimizer instance
    """
    if AdamP is not None:
        return AdamP(
            params,
            lr=lr,
            betas=ADAMP_CONFIG['betas'],
            weight_decay=ADAMP_CONFIG['weight_decay'] * weight_decay_scale,
            delta=ADAMP_CONFIG['delta'],
            wd_ratio=ADAMP_CONFIG['wd_ratio'],
            nesterov=ADAMP_CONFIG['nesterov']
        )
    else:
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4 * weight_decay_scale)


class MoLEContinualTrainer:
    """
    Slow-Fast MoLE-Flow Continual Learning Trainer.

    Implements Stage-Separated Training:
    - Stage 1 (FAST): LoRA + InputAdapter adaptation (base frozen)
    - Stage 2 (SLOW): Base NF consolidation (optional)

    Key Design Philosophy:
    - NF base = Slow learner (global normality manifold)
    - LoRA + InputAdapter = Fast learner (task-specific adaptation)

    Ablation Support:
    - use_lora: If False, skip LoRA (train shared NF)
    - use_router: If False, use oracle task_id
    - use_pos_embedding: If False, skip positional embedding
    - use_mahalanobis: If False, use Euclidean distance in router
    """

    def __init__(self,
                 vit_extractor,
                 pos_embed_generator,
                 nf_model: MoLESpatialAwareNF,
                 args,
                 device: str = 'cuda',
                 slow_lr_ratio: float = 0.2,
                 slow_blocks_k: int = 2,
                 ablation_config: 'AblationConfig' = None,
                 logger: Optional[TrainingLogger] = None):

        self.vit_extractor = vit_extractor
        self.pos_embed_generator = pos_embed_generator
        self.nf_model = nf_model
        self.args = args
        self.device = device

        # Ablation configuration
        self.ablation_config = ablation_config
        if ablation_config is None:
            # Default: all components enabled
            self.use_lora = True
            self.use_router = True
            self.use_pos_embedding = True
            self.use_mahalanobis = True
            self.lambda_logdet = 0.0
        else:
            self.use_lora = ablation_config.use_lora
            self.use_router = ablation_config.use_router
            self.use_pos_embedding = ablation_config.use_pos_embedding
            self.use_mahalanobis = ablation_config.use_mahalanobis
            self.lambda_logdet = ablation_config.lambda_logdet

        # Slow-Fast hyperparameters
        self.slow_lr_ratio = slow_lr_ratio
        self.slow_blocks_k = slow_blocks_k
        self.enable_slow_stage = getattr(args, 'enable_slow_stage', False)

        # Router for inference (only if enabled)
        if self.use_router:
            self.router = PrototypeRouter(device=device, use_mahalanobis=self.use_mahalanobis)
        else:
            self.router = None

        # Task information
        self.task_classes: Dict[int, List[str]] = {}

        # Store train loaders
        self.task_loaders: Dict[int, DataLoader] = {}

        # Training logger
        self.logger: Optional[TrainingLogger] = logger

        # Log ablation status
        self._log_ablation_status()

    def set_logger(self, logger: TrainingLogger):
        """Set training logger."""
        self.logger = logger

    def _log_ablation_status(self):
        """Log which components are enabled/disabled for ablation."""
        disabled = []
        if not self.use_lora:
            disabled.append("LoRA")
        if not self.use_router:
            disabled.append("Router")
        if not self.use_pos_embedding:
            disabled.append("PosEmbed")
        if not self.use_mahalanobis:
            disabled.append("Mahalanobis")

        if disabled:
            print(f"[Ablation] Disabled components: {', '.join(disabled)}")
        else:
            print("[Ablation] Full model (all components enabled)")

        # Log logdet regularization
        if self.lambda_logdet > 0:
            print(f"[Regularization] Logdet L2: lambda={self.lambda_logdet:.2e}")

    def _log(self, message: str, level: str = 'info'):
        """Log message using logger if available, otherwise print."""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
        else:
            print(message)

    def _compute_nll_loss(self, z: torch.Tensor, logdet_patch: torch.Tensor,
                          return_logdet_reg: bool = False):
        """
        Compute NLL loss using patch-wise log-likelihood.

        Standard NLL: L = -log p(x) = -log p(z) - log |det J|
        Now computed properly at patch level before aggregation.

        Optional: Logdet L2 regularization to stabilize image-level scores
        L_reg = lambda * (logdet_patch ** 2).mean()

        Args:
            z: Latent tensor [B, H, W, D]
            logdet_patch: Patch-wise log Jacobian determinant [B, H, W]
            return_logdet_reg: If True, also return logdet regularization term

        Returns:
            nll_loss (scalar), or (nll_loss, logdet_reg) if return_logdet_reg=True
        """
        B, H, W, D = z.shape

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        log_px_patch = log_pz_patch + logdet_patch

        # Image-level log-likelihood: sum over all patches
        log_px_image = log_px_patch.sum(dim=(1, 2))  # (B,)

        # NLL = -log p(x), averaged over batch
        nll_loss = -log_px_image.mean()

        if return_logdet_reg:
            # Logdet L2 regularization: encourages log|det J| to be small
            # This reduces variance in log_det across patches, stabilizing image scores
            logdet_reg = (logdet_patch ** 2).mean()
            return nll_loss, logdet_reg

        return nll_loss

    def train_task(self,
                   task_id: int,
                   task_classes: List[str],
                   train_loader: DataLoader,
                   num_epochs: int = 10,
                   lr: float = 1e-4,
                   log_interval: int = 10):
        """
        Stage-Separated Slow-Fast Training.

        Task 0: Base NF training + feature statistics collection
        Task > 0:
            Stage 1 (FAST): LoRA + InputAdapter adaptation (base frozen)
            Stage 2 (SLOW): Base NF consolidation with EWC protection (optional)
        """
        # Log task start
        if self.logger:
            self.logger.log_task_start(task_id, task_classes, num_epochs, lr)
        else:
            print(f"\n{'='*70}")
            print(f"🚀 Training Task {task_id}: {task_classes}")
            print(f"{'='*70}")

        self.task_classes[task_id] = task_classes
        self.task_loaders[task_id] = train_loader

        # Add task to NF model
        self.nf_model.add_task(task_id)

        if task_id == 0:
            # Task 0: Base NF Training
            self._train_base_task(task_id, task_classes, train_loader,
                                  num_epochs, lr, log_interval)

        else:
            # Task > 0: Two-Stage Slow-Fast Training
            if self.enable_slow_stage:
                fast_epochs = int(num_epochs * 0.85)
                slow_epochs = num_epochs - fast_epochs
            else:
                fast_epochs = num_epochs
                slow_epochs = 0

            # Stage 1: FAST Adaptation
            if self.logger:
                self.logger.log_stage_start(
                    "Stage 1: FAST Adaptation",
                    fast_epochs,
                    {"Base NF": "FROZEN", "LoRA + InputAdapter": "TRAINING"}
                )
            else:
                print(f"\n{'─'*70}")
                print(f"📌 Stage 1: FAST Adaptation ({fast_epochs} epochs)")
                print(f"{'─'*70}")

            self._train_fast_stage(task_id, task_classes, train_loader,
                                   fast_epochs, lr, log_interval)

            # Stage 2: SLOW Consolidation (optional)
            if self.enable_slow_stage and slow_epochs > 0:
                if self.logger:
                    self.logger.log_stage_start(
                        "Stage 2: SLOW Consolidation",
                        slow_epochs,
                        {
                            f"Last {self.slow_blocks_k} blocks": f"TRAINING (LR={lr * self.slow_lr_ratio:.2e})",
                            "LoRA + InputAdapter": "FROZEN"
                        }
                    )

                self._train_slow_stage(task_id, task_classes, train_loader,
                                       slow_epochs, lr * self.slow_lr_ratio, log_interval)

                self.nf_model.unfreeze_fast_params(task_id)

        # Build prototype for routing (only if router is enabled)
        if self.use_router:
            self._build_prototype(task_id, task_classes, train_loader)

        # Log task completion
        if self.logger:
            self.logger.log_task_complete(task_id)
        else:
            print(f"\n✅ Task {task_id} training completed!")

    def _train_base_task(self, task_id: int, task_classes: List[str],
                         train_loader: DataLoader, num_epochs: int,
                         lr: float, log_interval: int):
        """Train base NF + LoRA for Task 0."""
        trainable_params = self.nf_model.get_trainable_params(task_id)
        num_params = sum(p.numel() for p in trainable_params)

        # Check if LoRA is included (handles both MoLESubnet and MoLEContextSubnet)
        def _get_first_layer(subnet):
            return subnet.s_layer1 if hasattr(subnet, 's_layer1') else subnet.layer1

        has_lora = self.use_lora and any(
            str(task_id) in _get_first_layer(subnet).lora_A for subnet in self.nf_model.subnets
        )
        phase_name = "Base + LoRA Training" if has_lora else "Base Training"

        if self.logger:
            self.logger.log_model_info(
                f"{phase_name} + Feature Statistics Collection",
                num_params,
                {"Classes": task_classes, "Epochs": num_epochs, "Learning Rate": f"{lr:.2e}"}
            )
        else:
            print(f"   - Phase: {phase_name} + Feature Statistics Collection")
            print(f"   - Trainable Parameters: {num_params:,}")

        # Create optimizer (AdamP if available, AdamW fallback)
        optimizer = create_optimizer(list(trainable_params), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        self.nf_model.train()
        self.vit_extractor.eval()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
                images = images.to(self.device)

                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.vit_extractor(
                        images, return_spatial_shape=True
                    )
                    # Apply positional embedding only if enabled
                    if self.use_pos_embedding:
                        patch_embeddings_with_pos = self.pos_embed_generator(
                            spatial_shape, patch_embeddings
                        )
                    else:
                        # Reshape to (B, H, W, D) without positional embedding
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)
                    # Collect feature statistics
                    self.nf_model.update_reference_stats(patch_embeddings_with_pos)

                # Forward through NF to get latent z and patch-wise logdet
                z, logdet_patch = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

                # Compute NLL loss with optional logdet regularization
                if self.lambda_logdet > 0:
                    nll_loss, logdet_reg = self._compute_nll_loss(z, logdet_patch, return_logdet_reg=True)
                    total_loss = nll_loss + self.lambda_logdet * logdet_reg
                else:
                    nll_loss = self._compute_nll_loss(z, logdet_patch)
                    total_loss = nll_loss
                    logdet_reg = None

                if torch.isnan(total_loss) or total_loss.item() > 1e8:
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(trainable_params), max_norm=0.5)
                optimizer.step()

                epoch_loss += nll_loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    extra_info = {}
                    if logdet_reg is not None:
                        extra_info["logdet_reg"] = logdet_reg.item()
                    if self.logger:
                        self.logger.log_batch(
                            task_id, epoch, num_epochs, batch_idx+1, len(train_loader),
                            nll_loss.item(), avg_loss, stage="BASE", extra_info=extra_info if extra_info else None
                        )
                    else:
                        reg_str = f" | LogdetReg: {logdet_reg.item():.4f}" if logdet_reg is not None else ""
                        print(f"  [BASE] Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {nll_loss.item():.4f} | Avg: {avg_loss:.4f}{reg_str}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr

            # Get context gate/alpha info for logging
            extra_info = {"LR": current_lr}
            context_info = self.nf_model.get_context_info()
            if context_info:
                extra_info.update(context_info)

            if self.logger:
                self.logger.log_epoch(
                    task_id, epoch, num_epochs, avg_epoch_loss,
                    stage="BASE", extra_info=extra_info
                )
            else:
                ctx_str = ""
                if 'gate_mean' in context_info:
                    ctx_str = f" | Gate: {context_info['gate_mean']:.4f}±{context_info['gate_std']:.4f}"
                elif 'alpha_mean' in context_info:
                    ctx_str = f" | Alpha: {context_info['alpha_mean']:.4f}"
                print(f"  📊 [BASE] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}{ctx_str}")

        # Finalize feature statistics
        self.nf_model.finalize_reference_stats()

        # Log feature statistics
        if self.logger and hasattr(self.nf_model.reference_stats, 'mean'):
            stats = self.nf_model.reference_stats
            if stats.is_initialized:
                self.logger.log_feature_stats(
                    stats.n_samples,
                    stats.mean.norm().item(),
                    stats.std.mean().item(),
                    stats.std.min().item(),
                    stats.std.max().item()
                )

    def _train_fast_stage(self, task_id: int, task_classes: List[str],
                          train_loader: DataLoader, num_epochs: int,
                          lr: float, log_interval: int):
        """Stage 1: FAST Adaptation (LoRA + InputAdapter, base frozen)."""
        self.nf_model.freeze_all_base()

        fast_params = self.nf_model.get_fast_params(task_id)
        num_params = sum(p.numel() for p in fast_params)

        if self.logger:
            self.logger.log_model_info(
                "FAST Adaptation",
                num_params,
                {"Learning Rate": f"{lr:.2e}"}
            )
        else:
            print(f"   - Trainable Parameters: {num_params:,}")

        # Create optimizer (AdamP if available, AdamW fallback)
        optimizer = create_optimizer(list(fast_params), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        self.nf_model.train()
        self.vit_extractor.eval()

        warmup_epochs = 2

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Warmup
            if epoch < warmup_epochs:
                warmup_factor = (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr * warmup_factor

            for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
                images = images.to(self.device)

                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.vit_extractor(
                        images, return_spatial_shape=True
                    )
                    # Apply positional embedding only if enabled
                    if self.use_pos_embedding:
                        patch_embeddings_with_pos = self.pos_embed_generator(
                            spatial_shape, patch_embeddings
                        )
                    else:
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

                # Forward through NF to get latent z and patch-wise logdet
                z, logdet_patch = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

                # Compute NLL loss with optional logdet regularization
                if self.lambda_logdet > 0:
                    nll_loss, logdet_reg = self._compute_nll_loss(z, logdet_patch, return_logdet_reg=True)
                    total_loss = nll_loss + self.lambda_logdet * logdet_reg
                else:
                    nll_loss = self._compute_nll_loss(z, logdet_patch)
                    total_loss = nll_loss
                    logdet_reg = None

                if torch.isnan(total_loss) or total_loss.item() > 1e8:
                    continue

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(fast_params), max_norm=0.5)
                optimizer.step()

                epoch_loss += nll_loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = optimizer.param_groups[0]['lr']
                    extra_info = {"LR": current_lr}
                    if epoch < warmup_epochs:
                        extra_info["Warmup"] = True
                    if logdet_reg is not None:
                        extra_info["logdet_reg"] = logdet_reg.item()

                    if self.logger:
                        self.logger.log_batch(
                            task_id, epoch, num_epochs, batch_idx+1, len(train_loader),
                            nll_loss.item(), avg_loss, stage="FAST", extra_info=extra_info
                        )
                    else:
                        warmup_str = " (warmup)" if epoch < warmup_epochs else ""
                        reg_str = f" | LogdetReg: {logdet_reg.item():.4f}" if logdet_reg is not None else ""
                        print(f"  [FAST] Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {nll_loss.item():.4f} | Avg: {avg_loss:.4f}{warmup_str}{reg_str}")

            if epoch >= warmup_epochs:
                scheduler.step()

            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']

            # Get context gate/alpha info for logging
            extra_info = {"LR": current_lr}
            context_info = self.nf_model.get_context_info()
            if context_info:
                extra_info.update(context_info)

            if self.logger:
                self.logger.log_epoch(
                    task_id, epoch, num_epochs, avg_epoch_loss,
                    stage="FAST", extra_info=extra_info
                )
            else:
                ctx_str = ""
                if 'gate_mean' in context_info:
                    ctx_str = f" | Gate: {context_info['gate_mean']:.4f}±{context_info['gate_std']:.4f}"
                elif 'alpha_mean' in context_info:
                    ctx_str = f" | Alpha: {context_info['alpha_mean']:.4f}"
                print(f"  📊 [FAST] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}{ctx_str}")

    def _train_slow_stage(self, task_id: int, task_classes: List[str],
                          train_loader: DataLoader, num_epochs: int,
                          lr: float, log_interval: int):
        """Stage 2: SLOW Consolidation (last K blocks of base NF)."""
        self.nf_model.freeze_fast_params(task_id)
        self.nf_model.unfreeze_last_k_blocks(self.slow_blocks_k)

        slow_params = self.nf_model.get_base_params_for_slow_update(self.slow_blocks_k)
        num_params = sum(p.numel() for p in slow_params)

        if self.logger:
            self.logger.log_model_info(
                "SLOW Consolidation",
                num_params,
                {"Learning Rate": f"{lr:.2e}"}
            )
        else:
            print(f"   - Trainable Parameters: {num_params:,}")

        # Create optimizer (lower weight decay for slow consolidation)
        optimizer = create_optimizer(slow_params, lr=lr, weight_decay_scale=0.5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr * 0.01
        )

        self.nf_model.train()
        self.vit_extractor.eval()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (images, labels, _, _, _) in enumerate(train_loader):
                images = images.to(self.device)

                with torch.no_grad():
                    patch_embeddings, spatial_shape = self.vit_extractor(
                        images, return_spatial_shape=True
                    )
                    # Apply positional embedding only if enabled
                    if self.use_pos_embedding:
                        patch_embeddings_with_pos = self.pos_embed_generator(
                            spatial_shape, patch_embeddings
                        )
                    else:
                        B = patch_embeddings.shape[0]
                        H, W = spatial_shape
                        patch_embeddings_with_pos = patch_embeddings.reshape(B, H, W, -1)

                # NLL loss
                log_prob = self.nf_model.log_prob(patch_embeddings_with_pos)
                nll_loss = -log_prob.mean()

                if torch.isnan(nll_loss) or nll_loss.item() > 1e8:
                    continue

                optimizer.zero_grad()
                nll_loss.backward()
                torch.nn.utils.clip_grad_norm_(slow_params, max_norm=0.5)
                optimizer.step()

                epoch_loss += nll_loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    current_lr = optimizer.param_groups[0]['lr']

                    if self.logger:
                        self.logger.log_batch(
                            task_id, epoch, num_epochs, batch_idx+1, len(train_loader),
                            nll_loss.item(), avg_loss, stage="SLOW",
                            extra_info={"LR": current_lr}
                        )
                    else:
                        print(f"  [SLOW] Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                              f"Loss: {nll_loss.item():.4f} | Avg: {avg_loss:.4f}")

            scheduler.step()
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            current_lr = optimizer.param_groups[0]['lr']

            # Get context gate/alpha info for logging
            extra_info = {"LR": current_lr}
            context_info = self.nf_model.get_context_info()
            if context_info:
                extra_info.update(context_info)

            if self.logger:
                self.logger.log_epoch(
                    task_id, epoch, num_epochs, avg_epoch_loss,
                    stage="SLOW", extra_info=extra_info
                )
            else:
                ctx_str = ""
                if 'gate_mean' in context_info:
                    ctx_str = f" | Gate: {context_info['gate_mean']:.4f}±{context_info['gate_std']:.4f}"
                elif 'alpha_mean' in context_info:
                    ctx_str = f" | Alpha: {context_info['alpha_mean']:.4f}"
                print(f"  📊 [SLOW] Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}{ctx_str}")

    def _build_prototype(self, task_id: int, task_classes: List[str],
                         train_loader: DataLoader):
        """Build prototype for routing."""
        self._log(f"\n📦 Building prototype for Task {task_id}...")

        all_image_features = []
        self.nf_model.eval()
        self.vit_extractor.eval()

        with torch.no_grad():
            for images, _, _, _, _ in train_loader:
                images = images.to(self.device)
                image_features = self.vit_extractor.get_image_level_features(images)
                all_image_features.append(image_features)

        prototype = TaskPrototype(task_id, task_classes, self.device)
        all_features = torch.cat(all_image_features, dim=0)
        prototype.update(all_features)
        prototype.finalize()

        self.router.add_prototype(task_id, prototype)

        if self.logger:
            self.logger.log_prototype_creation(
                task_id, prototype.n_samples, prototype.mean.shape
            )
        else:
            print(f"   ✅ Prototype stored: μ shape={prototype.mean.shape}, "
                  f"n_samples={prototype.n_samples}")

    def inference(self, images: torch.Tensor, task_id: Optional[int] = None):
        """
        Inference with automatic routing or specified task.

        Args:
            images: (B, C, H, W) input images
            task_id: If None, use router to predict task (if enabled)

        Returns:
            anomaly_scores: (B, H_patch, W_patch) spatial anomaly map
            image_scores: (B,) image-level scores
            predicted_tasks: (B,) predicted task IDs

        Respects ablation flags:
        - use_router: If False, always use task_id (oracle mode)
        - use_pos_embedding: If False, skip positional embedding
        """
        self.nf_model.eval()
        self.vit_extractor.eval()

        with torch.no_grad():
            images = images.to(self.device)

            # Route if task_id not specified and router is enabled
            if task_id is None and self.use_router and self.router is not None and len(self.router.prototypes) > 0:
                image_features = self.vit_extractor.get_image_level_features(images)
                predicted_tasks = self.router.route(image_features)
            else:
                predicted_tasks = torch.full((images.shape[0],),
                                            task_id if task_id is not None else 0,
                                            dtype=torch.long, device=self.device)

            # Process each unique task
            unique_tasks = predicted_tasks.unique()

            B = images.shape[0]
            anomaly_scores = torch.zeros(B, 37, 37, device=self.device)
            image_scores = torch.zeros(B, device=self.device)

            for t_id in unique_tasks:
                mask = (predicted_tasks == t_id)
                task_images = images[mask]

                if task_images.shape[0] == 0:
                    continue

                # Set active task
                self.nf_model.set_active_task(t_id.item())

                # Extract features
                patch_embeddings, spatial_shape = self.vit_extractor(
                    task_images, return_spatial_shape=True
                )
                # Apply positional embedding only if enabled
                if self.use_pos_embedding:
                    patch_embeddings_with_pos = self.pos_embed_generator(
                        spatial_shape, patch_embeddings
                    )
                else:
                    B_task = patch_embeddings.shape[0]
                    H, W = spatial_shape
                    patch_embeddings_with_pos = patch_embeddings.reshape(B_task, H, W, -1)

                # Compute anomaly scores
                task_anomaly_scores, task_image_scores = self._compute_anomaly_scores(
                    patch_embeddings_with_pos
                )

                # Resize if needed
                H, W = spatial_shape
                if anomaly_scores.shape[1] != H or anomaly_scores.shape[2] != W:
                    anomaly_scores = torch.zeros(B, H, W, device=self.device)

                anomaly_scores[mask] = task_anomaly_scores
                image_scores[mask] = task_image_scores

        return anomaly_scores, image_scores, predicted_tasks

    def _compute_anomaly_scores(self, patch_embeddings_with_pos: torch.Tensor):
        """
        Compute anomaly scores from patch embeddings.

        Now uses patch-wise log|det J| directly for proper spatial localization.
        This preserves the Flow's spatial information in the anomaly map.
        """
        B, H, W, D = patch_embeddings_with_pos.shape

        # Forward through NF - now returns patch-wise logdet: (B, H, W)
        z, logdet_patch = self.nf_model.forward(patch_embeddings_with_pos, reverse=False)

        # Patch-wise log p(z): (B, H, W)
        log_pz_patch = -0.5 * (z ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        # Patch-wise log p(x) = log p(z) + log|det J|: (B, H, W)
        # Now each patch has its own log|det J| value, preserving spatial variation
        patch_log_prob = log_pz_patch + logdet_patch

        # Anomaly score = -log p(x)
        anomaly_scores = -patch_log_prob

        # Image-level score (99th percentile)
        image_scores = torch.quantile(anomaly_scores.reshape(B, -1), 0.99, dim=1)

        return anomaly_scores, image_scores
