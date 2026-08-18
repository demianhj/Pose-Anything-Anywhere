import torch
import torch.nn as nn
import torch.nn.functional as F

from vggt.mv_match.model.transformer import SparseToDenseTransformer
from vggt.mv_match.utils.model_utils import compute_feature_similarity, compute_correspondences
from vggt.mv_match.utils.loss_utils import compute_correspondence_loss
from vggt.mv_match.model.pointnet2.pointnet2_utils import QueryAndGroup
from vggt.mv_match.model.pointnet2.pytorch_utils import SharedMLP, Conv1d
from vggt.mv_match.utils.model_utils import pairwise_distance
import numpy as np

class FinePointMatching(nn.Module):
    def __init__(self, cfg, return_feat=False):
        super(FinePointMatching, self).__init__()
        self.cfg = cfg
        self.return_feat = return_feat
        self.nblock = self.cfg.nblock

        self.in_proj = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.out_dim)

        self.bg_token = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * .02)
        self.PE = PositionalEncoding(cfg.hidden_dim, r1=cfg.pe_radius1, r2=cfg.pe_radius2)

        self.transformers = []
        for _ in range(self.nblock):
            self.transformers.append(SparseToDenseTransformer(
                cfg.hidden_dim,
                num_heads=4,
                sparse_blocks=['self', 'cross'],
                dropout=None,
                activation_fn='ReLU',
                focusing_factor=cfg.focusing_factor,
                with_bg_token=True,
                replace_bg_token=True
            ))
        self.transformers = nn.ModuleList(self.transformers)

    def forward(self, p1, f1, geo1, fps_idx1, p2, f2, geo2, fps_idx2, gt_p1=None, gt_p2=None):
        B = p1.size(0)

        f1 = self.in_proj(f1) + self.PE(p1)
        f1 = torch.cat([self.bg_token.repeat(B,1,1), f1], dim=1) # adding bg

        f2 = self.in_proj(f2) + self.PE(p2)
        f2 = torch.cat([self.bg_token.repeat(B,1,1), f2], dim=1) # adding bg

        atten_list = []
        for idx in range(self.nblock):
            f1, f2 = self.transformers[idx](f1, geo1, fps_idx1, f2, geo2, fps_idx2)

            if gt_p1 is not None or idx==self.nblock-1:
                atten_list.append(compute_feature_similarity(
                    self.out_proj(f1),
                    self.out_proj(f2),
                    self.cfg.sim_type,
                    self.cfg.temp,
                    self.cfg.normalize_feat
                ))
        result = []
        if gt_p1 is not None:
            # compute the coarse correspondence
            dis_mat = torch.sqrt(pairwise_distance(p1, p2)) # pts1: query pts2: target
            dis1, label1 = dis_mat.min(2)
            fg_label1 = (dis1<=self.cfg.loss_dis_thres).float()
            pred_coarse_label = (fg_label1 * (label1.float()+1.0)).long()

            loss, pred_label, gt_label = compute_correspondence_loss(
                atten_list, gt_p1, gt_p2, dis_thres=self.cfg.loss_dis_thres
            )
            # visualize the feature
            feat1 = self.out_proj(f1)
            feat2 = self.out_proj(f2)
            vis_feat1 = feat1[:,1:,:][0]
            vis_feat2 = feat2[:,1:,:][0]
            # normalize feature
            vis_feat1 = F.normalize(vis_feat1, p=2, dim=-1).detach().cpu().numpy()
            vis_feat2 = F.normalize(vis_feat2, p=2, dim=-1).detach().cpu().numpy()
            vis_gt_p1 = gt_p1[0].detach().cpu().numpy()
            vis_gt_p2 = gt_p2[0].detach().cpu().numpy()
            vis_feature_data = {
                "pts1": vis_gt_p1,
                "pts2": vis_gt_p2,
                "feat1": vis_feat1,
                "feat2": vis_feat2
            }
            result = {
                "loss": loss,
                "pred_coarse_label": pred_coarse_label,
                "pred_label": pred_label,
                "gt_label": gt_label,
                "vis_feature_data": vis_feature_data
            }
        else:
            pred_label, pred_conf = compute_correspondences(atten_list[-1])
            result = {
                "pred_label": pred_label,
                "pred_conf": pred_conf
            }
        return result
            
        # if self.return_feat:
        #     return loss, self.out_proj(f1), self.out_proj(f2)
        # else:
        #     return loss

class PositionalEncoding(nn.Module):
    def __init__(self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True):
        super(PositionalEncoding, self).__init__()
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz)
        self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz)
        input_dim = 6 if use_xyz else 3

        self.mlp1 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp2 = SharedMLP([input_dim, 32, 64, 128], bn=bn)
        self.mlp3 = Conv1d(256, out_dim, 1, activation=None, bn=None)

    def forward(self, pts1, pts2=None):
        if pts2 is None:
            pts2 = pts1

        # scale1
        feat1 = self.group1(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat1 = self.mlp1(feat1)
        feat1 = F.max_pool2d(
            feat1, kernel_size=[1, feat1.size(3)]
        )

        # scale2
        feat2 = self.group2(
                pts1.contiguous(), pts2.contiguous(), pts1.transpose(1,2).contiguous()
            )
        feat2 = self.mlp2(feat2)
        feat2 = F.max_pool2d(
            feat2, kernel_size=[1, feat2.size(3)]
        )

        feat = torch.cat([feat1, feat2], dim=1).squeeze(-1)
        feat = self.mlp3(feat).transpose(1,2)
        return feat