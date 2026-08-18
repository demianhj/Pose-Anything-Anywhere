# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dpt_head import DPTHead

class DWConvBlock(nn.Module):
    """Depthwise -> Pointwise -> BN -> ReLU"""
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, 1, p, groups=in_ch, bias=False)  # depthwise
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)               # pointwise
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)   # 每个通道独立卷
        x = self.pw(x)   # 通道混合
        x = self.bn(x)
        return self.act(x)
    
class MatchHead(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        blocks=2,
        H=518, W=518,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size
        self.patch_W = W // patch_size
        self.patch_H = H // patch_size
        self.num_cls = self.patch_W  * self.patch_H + 1  # +1 for "no match"

        in_ch = features * 4  # [f0, fs, f0-fs, f0*fs]
        layers = [DWConvBlock(in_ch, features)]
        for _ in range(blocks-1):
            layers.append(DWConvBlock(features, features))
        self.fuse = nn.Sequential(*layers)

        self.cls_head = nn.Conv2d(features, self.num_cls, kernel_size=1)
        self.off_head = nn.Conv2d(features, 2, kernel_size=1)

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=patch_size,  # Reduces spatial dimensions by factor of patch size
            pos_embed=False,
        )

    def forward(self, aggregated_tokens_list, images, patch_start_idx):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.

        Returns:
            feature_maps (torch.Tensor): Extracted feature maps from the tokens. (B, S, C, H, W)
            logits_cls: (B,H,W,C)
            pred_off: (B,H,W,2)
        """
        # Extract features from tokens
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        # Compute logits and offsets
        B, S, D, Hg, Wg = feature_maps.shape
        f0 = feature_maps[:, 0]           # (B,D,Hg,Wg)
        fs = feature_maps[:, 1:]          # (B,S-1,D,Hg,Wg)

        # 归一化再融合（可选）
        f0 = F.normalize(f0, dim=1)
        fs = F.normalize(fs, dim=2)

        # pair-wise 向量化：(B*(S-1), D, Hg, Wg)
        f0_rep = f0.unsqueeze(1).expand(-1, S-1, -1, -1, -1).reshape(B*(S-1), D, Hg, Wg)
        fs_rep = fs.reshape(B*(S-1), D, Hg, Wg)

        x = torch.cat([f0_rep, fs_rep, f0_rep - fs_rep, f0_rep * fs_rep], dim=1)  # (B*(S-1),4D,Hg,Wg)
        fused = self.fuse(x)  # (B*(S-1), C', Hg, Wg)

        logits = self.cls_head(fused).permute(0,2,3,1).contiguous()  # (B*(S-1),Hg,Wg,C)
        offs = self.off_head(fused).permute(0,2,3,1).contiguous()  # (B*(S-1),Hg,Wg,2)

        logits = logits.view(B, S-1, Hg, Wg, self.num_cls)
        offs = offs.view(B, S-1, Hg, Wg, 2)
        return logits, offs

class MatchHead_v2(nn.Module):
    """
    Track head that uses DPT head to process tokens and BaseTrackerPredictor for tracking.
    The tracking is performed iteratively, refining predictions over multiple iterations.
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        blocks=2,
        H=518, W=518,
    ):
        """
        Initialize the TrackHead module.

        Args:
            dim_in (int): Input dimension of tokens from the backbone.
            patch_size (int): Size of image patches used in the vision transformer.
            features (int): Number of feature channels in the feature extractor output.
            iters (int): Number of refinement iterations for tracking predictions.
            predict_conf (bool): Whether to predict confidence scores for tracked points.
            stride (int): Stride value for the tracker predictor.
            corr_levels (int): Number of correlation pyramid levels
            corr_radius (int): Radius for correlation computation, controlling the search area.
            hidden_size (int): Size of hidden layers in the tracker network.
        """
        super().__init__()

        self.patch_size = patch_size
        self.patch_W = W // patch_size
        self.patch_H = H // patch_size
        self.num_cls = self.patch_W  * self.patch_H + 1  # +1 for "no match"

        in_ch = features * 4  # [f0, fs, f0-fs, f0*fs]
        layers = [DWConvBlock(in_ch, features)]
        for _ in range(blocks-1):
            layers.append(DWConvBlock(features, features))
        self.fuse = nn.Sequential(*layers)

        self.cls_head = nn.Conv2d(features, self.num_cls, kernel_size=1)
        self.off_head = nn.Conv2d(features, 2, kernel_size=1)

        # Feature extractor based on DPT architecture
        # Processes tokens into feature maps for tracking
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            down_ratio=patch_size,  # Reduces spatial dimensions by factor of patch size
            pos_embed=False,
        )

    def forward(self, aggregated_tokens_list, images, patch_start_idx, positive_frame=None):
        """
        Forward pass of the TrackHead.

        Args:
            aggregated_tokens_list (list): List of aggregated tokens from the backbone.
            images (torch.Tensor): Input images of shape (B, S, C, H, W) where:
                                   B = batch size, S = sequence length.
            patch_start_idx (int): Starting index for patch tokens.

        Returns:
            feature_maps (torch.Tensor): Extracted feature maps from the tokens. (B, S, C, H, W)
            logits_cls: (B,H,W,C)
            pred_off: (B,H,W,2)
        """
        # Extract features from tokens
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        # Compute logits and offsets
        B, S, D, Hg, Wg = feature_maps.shape
        f0 = feature_maps[:, 0]           # (B,D,Hg,Wg)
        fs = feature_maps[:, 1:]          # (B,S-1,D,Hg,Wg)

        # 归一化再融合（可选）
        f0 = F.normalize(f0, dim=1)
        fs = F.normalize(fs, dim=2)

        # pair-wise 向量化：(B*(S-1), D, Hg, Wg)
        f0_rep = f0.unsqueeze(1).expand(-1, S-1, -1, -1, -1).reshape(B*(S-1), D, Hg, Wg)
        fs_rep = fs.reshape(B*(S-1), D, Hg, Wg)

        x = torch.cat([f0_rep, fs_rep, f0_rep - fs_rep, f0_rep * fs_rep], dim=1)  # (B*(S-1),4D,Hg,Wg)
        fused = self.fuse(x)  # (B*(S-1), C', Hg, Wg)

        logits = self.cls_head(fused).permute(0,2,3,1).contiguous()  # (B*(S-1),Hg,Wg,C)
        offs = self.off_head(fused).permute(0,2,3,1).contiguous()  # (B*(S-1),Hg,Wg,2)

        logits = logits.view(B, S-1, Hg, Wg, self.num_cls)
        offs = offs.view(B, S-1, Hg, Wg, 2)

        if positive_frame is not None:
            ref_feature_maps = feature_maps[:, 1:][positive_frame]
            if len(ref_feature_maps.shape) == 4:
                ref_feature_maps = ref_feature_maps.unsqueeze(0)
            fp_0 = ref_feature_maps[:, 0]
            fp_1 = ref_feature_maps[:, 1]
            fp_0 = F.normalize(fp_0, dim=1)
            fp_1 = F.normalize(fp_1, dim=1)
            x_ref = torch.cat([fp_0, fp_1, fp_0 - fp_1, fp_0 * fp_1], dim=1)
            fused_ref = self.fuse(x_ref)
            logits_ref = self.cls_head(fused_ref).permute(0,2,3,1).contiguous()
            offs_ref = self.off_head(fused_ref).permute(0,2,3,1).contiguous()
            return logits, offs, logits_ref, offs_ref

        return logits, offs