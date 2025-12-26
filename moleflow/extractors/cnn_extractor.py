"""
PatchCore Feature Extractor (CNN-based).

Extract patch embeddings from CNN backbones (e.g., ResNet, WideResNet).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from moleflow.extractors.hooks import NetworkFeatureAggregator
from moleflow.extractors.patch import PatchMaker
from moleflow.extractors.preprocessing import Preprocessing


class PatchCoreExtractor(nn.Module):
    """Extract patch embeddings from CNN backbones (e.g., ResNet, WideResNet)."""

    def __init__(self,
                 backbone_name="wide_resnet50_2",
                 layers=("layer2", "layer3"),
                 input_shape=(3, 224, 224),
                 patch_size=3,
                 patch_stride=1,
                 target_embed_dimension=1024,
                 device='cuda'):
        super(PatchCoreExtractor, self).__init__()

        self.device = device
        self.layers_to_extract_from = layers

        # Load backbone
        self.backbone = timm.create_model(backbone_name, pretrained=True)
        self.backbone.to(device)
        self.backbone.eval()

        # Setup feature aggregator
        self.feature_aggregator = NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from
        )

        # Calculate feature dimensions
        feature_dims = self.feature_aggregator.feature_dimensions(input_shape, device)

        # Setup patch maker
        self.patch_maker = PatchMaker(patch_size, stride=patch_stride)

        # Setup preprocessing (channel reduction)
        self.preprocessing = Preprocessing(feature_dims, target_embed_dimension)
        self.preprocessing.to(device)

        self.target_embed_dimension = target_embed_dimension

    def forward(self, images):
        """
        Extract patch embeddings from images.

        Args:
            images: (B, C, H, W) input images

        Returns:
            features: (B * Total_Patches, Target_Embed_Dimension)
        """
        images = images.to(self.device)

        # 1. Backbone feature extraction
        features_dict = self.feature_aggregator(images)
        features = [features_dict[layer] for layer in self.layers_to_extract_from]

        # 2. Patchify (unfold operation)
        features_info = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features_info]
        features = [x[0] for x in features_info]

        # 3. Align feature maps via interpolation
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )

            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape

            _features = _features.reshape(-1, *_features.shape[-2:])

            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)

            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        # Flatten patches
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # 4. Preprocessing and aggregation
        features = self.preprocessing(features)
        features = features.mean(dim=1)

        return features
