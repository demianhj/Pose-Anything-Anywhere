import torch
import torch.nn as nn
from torch.nn import functional as F

from vggt.mv_match.model.fine_point_matching import FinePointMatching
from vggt.mv_match.model.transformer import GeometricStructureEmbedding
from vggt.mv_match.utils.model_utils import sample_pts_feats
from vggt.heads.dpt_head import DPTHead

def get_chosen_pixel_feats(img_feat, choose):
    shape = img_feat.size()
    if len(shape) == 3:
        pass
    elif len(shape) == 4:
        B, C, H, W = shape
        img_feat = img_feat.reshape(B, C, H*W)
    else:
        assert False

    choose = choose.unsqueeze(1).repeat(1, C, 1)
    x = torch.gather(img_feat, 2, choose).contiguous()
    return x.transpose(1,2).contiguous()

class match_net(nn.Module):
    def __init__(self,
                 cfg, 
                 dim_in,         
                 patch_size=14,
                 features=128):
        super(match_net, self).__init__()
        self.cfg = cfg
        self.coarse_npoint = cfg.coarse_npoint
        self.fine_npoint = cfg.fine_npoint

        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # Only output features, no activation
            pos_embed=False,
        )
        self.geo_embedding = GeometricStructureEmbedding(cfg.geo_embedding)
        self.fine_point_matching = FinePointMatching(cfg.fine_point_matching)

    def forward(self, aggregated_tokens_list, images, patch_start_idx, pred_pts, 
                sampled_choose, sampled_pts=None, positive_frames=None):
        # Extract features from tokens
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx) # (B, S, C, H, W)
        
        if positive_frames is not None: 
            # Only use the positive frames for matching
            feature_maps = feature_maps[positive_frames]
            if feature_maps.dim() == 4:
                feature_maps = feature_maps.unsqueeze(0)
            # Get dense point clouds and features based on the sampled choose
            pred_pts = pred_pts[positive_frames] 
            if pred_pts.dim() == 4:
                pred_pts = pred_pts.unsqueeze(0)
            pred_pts = pred_pts.reshape(pred_pts.size(0), pred_pts.size(1), -1, 3) # (B, S, N, 3)
            pred_pts_sampled = pred_pts.gather(2, sampled_choose.unsqueeze(-1).repeat(1,1,1,3)) # (B, S', N, 3)
            # use the first frame to center and normalize 
            centeriod = torch.mean(pred_pts_sampled[:,0], dim=1, keepdim=True).unsqueeze(1) # (B, 1, 1, 3)
            pred_pts_sampled = pred_pts_sampled - centeriod
            radius = torch.norm(pred_pts_sampled[:,0], dim=2).max(dim=1)[0]
            pred_pts_sampled = pred_pts_sampled / (radius.reshape(-1, 1, 1) + 1e-6)

            pred_feats_sampled = []
            for s in range(pred_pts_sampled.shape[1]):
                pred_feat_sampled = get_chosen_pixel_feats(feature_maps[:,s], sampled_choose[:,s])
                pred_feats_sampled.append(pred_feat_sampled)
            pred_feats_sampled = torch.stack(pred_feats_sampled, dim=1) # (B, S', N, C)
            # Iterate get the matching (3 pairs)
            gt_p0 = sampled_pts[:,0]
            pred_p0 = pred_pts_sampled[:,0]
            pts_f0 = pred_feats_sampled[:,0]
            gt_p1 = sampled_pts[:,1]
            pred_p1 = pred_pts_sampled[:,1]
            pts_f1 = pred_feats_sampled[:,1]
            gt_p2 = sampled_pts[:,2]
            pred_p2 = pred_pts_sampled[:,2]
            pts_f2 = pred_feats_sampled[:,2]
            results = []
            for i in range(3):
                if i == 0:
                    res = self.get_matching(pred_p0, pts_f0, pred_p1, pts_f1, gt_p0, gt_p1)
                    results.append(res)
                elif i == 1:
                    res = self.get_matching(pred_p0, pts_f0, pred_p2, pts_f2, gt_p0, gt_p2)
                    results.append(res)
                else:
                    res = self.get_matching(pred_p1, pts_f1, pred_p2, pts_f2, gt_p1, gt_p2)
                    results.append(res)
                    
        else:
            # Get dense point clouds and features based on the sampled choose
            pred_pts = pred_pts.reshape(pred_pts.shape[0], pred_pts.shape[1], -1, 3) # (B, S, N, 3)
            pred_pts_sampled = pred_pts.gather(2, sampled_choose.unsqueeze(-1).repeat(1,1,1,3)) # (B, S, N, 3)
            # use the first frame to center and normalize 
            centeriod = torch.mean(pred_pts_sampled[:,0], dim=1, keepdim=True).unsqueeze(1) # (B, 1, 1, 3)
            pred_pts_sampled = pred_pts_sampled - centeriod
            radius = torch.norm(pred_pts_sampled[:,0], dim=2).max(dim=1)[0]
            pred_pts_sampled = pred_pts_sampled / (radius.reshape(-1, 1, 1) + 1e-6)
            
            pred_feats_sampled = []
            for s in range(pred_pts_sampled.shape[1]):
                pred_feat_sampled = get_chosen_pixel_feats(feature_maps[:,s], sampled_choose[:,s])
                pred_feats_sampled.append(pred_feat_sampled)
            pred_feats_sampled = torch.stack(pred_feats_sampled, dim=1)
            # Iterate get the matching (first frame to other frames)
            pred_p0 = pred_pts_sampled[:,0]
            pts_f0 = pred_feats_sampled[:,0]
            results = []
            for s in range(1, pred_pts_sampled.shape[1]):
                pred_p1 = pred_pts_sampled[:,s]
                pts_f1 = pred_feats_sampled[:,s]
                res = self.get_matching(pred_p0, pts_f0, pred_p1, pts_f1)
                results.append(res)

        return results
    
    def get_matching(self, dense_po, dense_fo, dense_pm, dense_fm, gt_po=None, gt_pm=None):
        # pre-compute geometric embeddings for geometric transformer
        bg_point = torch.ones(dense_pm.size(0),1,3).float().to(dense_pm.device) * 100

        sparse_pm, _, fps_idx_m = sample_pts_feats(
            dense_pm, dense_fm, self.coarse_npoint, return_index=True
        )
        geo_embedding_m = self.geo_embedding(torch.cat([bg_point, sparse_pm], dim=1))

        sparse_po, _, fps_idx_o = sample_pts_feats(
            dense_po, dense_fo, self.coarse_npoint, return_index=True
        )
        geo_embedding_o = self.geo_embedding(torch.cat([bg_point, sparse_po], dim=1))

        # fine_point_matching
        result = self.fine_point_matching(
            dense_po, dense_fo, geo_embedding_o, fps_idx_o,
            dense_pm, dense_fm, geo_embedding_m, fps_idx_m,
            gt_po, gt_pm
        )

        return result

