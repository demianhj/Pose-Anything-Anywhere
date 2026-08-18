import torch
import numpy as np 
from torchmetrics import MeanMetric
from vggt.models.aggregator import Aggregator
from vggt.heads.dpt_head import DPTHead
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from training.vis_utils import *
from training.loss import compute_depth_loss, compute_point_loss
import wandb
import matplotlib
matplotlib.use("Agg") 
import os
import psutil
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from vggt.mv_match.model.pose_estimation_model import match_net
from vggt.mv_match.utils.vis_utils import vis_gt_matching, vis_pred_matching
from vggt.mv_match.utils.vis_utils import vis_feat

@rank_zero_only
def safe_log(pl_module, key, value, **kwargs):
    pl_module.log(key, value, **kwargs)

@rank_zero_only
def log_images(logger, data_dict):
    logger.experiment.log(data_dict)

def get_scheduler(optimizer, warmup_steps, hold_steps, decay_end, base_lr, min_lr):
    def lr_lambda(current_step):
        # --- Warmup: 0 → base_lr ---
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # --- Hold: constant base_lr ---
        elif current_step < hold_steps:
            return 1.0
        # --- Decay: base_lr → min_lr ---
        elif current_step < decay_end:
            progress = (current_step - hold_steps) / float(decay_end - hold_steps)
            return 1.0 - (1.0 - min_lr / base_lr) * progress
        # --- After decay_end: fixed min_lr ---
        else:
            return min_lr / base_lr

    return LambdaLR(optimizer, lr_lambda)

class PANY(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.aggregator = Aggregator(img_size=self.config.img_size, patch_size=self.config.patch_size, 
                                     patch_embed=self.config.patch_embed,
                                     embed_dim=self.config.embed_dim, depth=self.config.depth)
        self.depth_head = DPTHead(dim_in=2 * self.config.embed_dim, output_dim=2,
                                  intermediate_layer_idx=self.config.intermediate_layer_idx, 
                                  activation="exp", conf_activation="expp1")
        self.point_head = DPTHead(dim_in=2 * self.config.embed_dim, output_dim=4, 
                                intermediate_layer_idx=self.config.intermediate_layer_idx,
                                activation="inv_log", conf_activation="expp1")
        self.track_head = match_net(cfg=self.config.sam6d_model, dim_in=2 * self.config.embed_dim)
        
        self.depth_loss = compute_depth_loss
        self.pts_loss = compute_point_loss

        self.train_loss_epoch = MeanMetric()
        self.train_depth_loss = MeanMetric()
        self.train_pts_loss = MeanMetric()
        self.train_match_loss = MeanMetric()

        self.val_loss_epoch = MeanMetric(sync_on_compute=False)
        self.val_depth_loss = MeanMetric(sync_on_compute=False)
        self.val_pts_loss = MeanMetric(sync_on_compute=False)
        self.val_match_loss = MeanMetric(sync_on_compute=False)

    def forward(self, images: torch.Tensor, sampled_choose, sampled_pts, masks, positive_frames=None):
        """
        Forward pass of the PANY model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            sampled_choose (torch.Tensor): Indices of sampled pixels for point matching.
            sampled_pts (torch.Tensor): 3D coordinates of sampled points for point matching.
            positive_frames (optional): List of indices indicating positive frames for matching. Defaults to None.

        Returns:
            dict: A dictionary containing the following predictions:
                - points (torch.Tensor): Reference Object Coordinates for each pixel with shape [B, S, H, W, 3]
                - points_conf (torch.Tensor): Confidence scores for Reference Object Coordinates with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}
        
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):  
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["points"] = pts3d
                predictions["points_conf"] = pts3d_conf
            
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
        
            if self.track_head is not None:
                masked_pts3d = pts3d * masks.unsqueeze(-1)
                if positive_frames is not None:
                    match_res = self.track_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, pred_pts=masked_pts3d, 
                                                   sampled_choose=sampled_choose, sampled_pts=sampled_pts, positive_frames=positive_frames)
                else:
                    match_res = self.track_head(aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, pred_pts=masked_pts3d, sampled_choose=sampled_choose)
                predictions["match_res"] = match_res
                
        predictions["images"] = images

        return predictions
    
    def training_step(self, batch, batch_idx):
        images = batch["images"]
        masks = batch["masks"]
        if "positive_frames" in batch:
            positive_frames = batch["positive_frames"]
        else:
            positive_frames = None
        sampled_choose = batch["sampled_choose"]
        sampled_pts = batch["sampled_pts"]

        predictions = self.forward(images, sampled_choose, sampled_pts, masks, positive_frames)
        loss = 0
        # Compute the depth loss
        if self.depth_head is not None:
            pred_depth = predictions["depth"]
            gt_depth = batch["gt_depthmaps"]

            depth_loss_dict = self.depth_loss(predictions, batch)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            loss += depth_loss
        # Compute the point loss
        if self.point_head is not None:
            pred_pts = predictions["points"]
            conf_pts = predictions["points_conf"]
            gt_pts = batch["gt_pointmaps"]

            pts_loss_dict = self.pts_loss(pred_pts, conf_pts, gt_pts, masks) # vggt
            loss_conf_point = pts_loss_dict['loss_conf_point']
            loss_reg_point = pts_loss_dict['loss_reg_point']
            loss_grad_point = pts_loss_dict['loss_grad_point']
            point_loss = loss_conf_point + loss_reg_point + loss_grad_point
            loss += point_loss
        # Compute the match loss
        if self.track_head is not None:
            match_res = predictions["match_res"]
            match_loss = sum(item['loss'] for item in match_res)[0] / len(match_res)
            loss += match_loss

        self.train_loss_epoch.update(loss.detach().cpu().item())
        self.train_depth_loss.update(depth_loss.detach().cpu().item())
        self.train_pts_loss.update(point_loss.detach().cpu().item())  
        self.train_match_loss.update(match_loss.detach().cpu().item())
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]

        # monitor system memory usage
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 ** 2

        safe_log(
            self,
            "train_loss",
            loss.detach().cpu().item(),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            sync_dist=False,
        )
        safe_log(self, "lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        safe_log(self, "cpu", mem_mb, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Visualization logging
        if self.config.enable_plotting and self.trainer.is_global_zero:
            if self.global_step % self.config.log_every_n_steps == 0:
                num_ref = images.shape[1]
                vis_depth_res = []
                vis_pts_res = []
                for idx in range(num_ref):
                    file_path = batch["filepaths"][idx][0]
                    vis_RGB = images[0, idx].permute(1, 2, 0).cpu().numpy()
                    vis_mask = masks[0, idx].cpu().numpy()
                    if self.depth_head is not None:
                        # visualize depth map
                        vis_depth_res.append(visualize_depth_res(vis_RGB, gt_depth[0, idx], pred_depth[0, idx].squeeze(-1), vis_mask))
                    if self.point_head is not None:
                        # visualize pts map
                        vis_gt_pts = visualize_pts(gt_pts[0, idx].cpu().numpy(), vis_mask, background_color=(0, 0, 0))
                        vis_pred_pts = visualize_pts(pred_pts[0, idx].detach().cpu().numpy(), vis_mask, background_color=(0, 0, 0))
                        vis_pts_res.append(visualize_RGB_pts(file_path, vis_RGB, vis_gt_pts, vis_pred_pts, dpi=100))

                if self.depth_head is not None:
                    vis_depth_res = pad_to_max_width(vis_depth_res)
                if self.point_head is not None:
                    vis_pts_res = pad_to_max_width(vis_pts_res)                            
                    # visualize in 3d space
                    query_gt_pts = gt_pts[0, 0].detach().cpu().numpy() * masks[0, 0][:,:,None].cpu().numpy()
                    ref_gt_pts = gt_pts[0, 1].detach().cpu().numpy() * masks[0, 1][:,:,None].cpu().numpy()
                    vis_gt_pts_res_3d = visualize_pts_in_3d(query_gt_pts.reshape(-1,3), ref_gt_pts.reshape(-1,3))

                    query_pred_pts = pred_pts[0, 0].detach().cpu().numpy() * masks[0, 0][:,:,None].cpu().numpy()
                    ref_pred_pts = pred_pts[0, 1].detach().cpu().numpy() * masks[0, 1][:,:,None].cpu().numpy()
                    vis_pred_pts_res_3d = visualize_pts_in_3d(query_pred_pts.reshape(-1,3), ref_pred_pts.reshape(-1,3))

                if self.track_head is not None:
                    # for visualization
                    sampled_choose_1 = sampled_choose[0, 0, :]
                    sampled_choose_2 = sampled_choose[0, 1, :]
                    sampled_choose_3 = sampled_choose[0, 2, :]
                    positive_images = images[positive_frames]
                    vis_img_1 = (positive_images[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    vis_img_2 = (positive_images[1].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    vis_img_3 = (positive_images[2].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                    whole_gt_vis = []
                    whole_pred_vis = []
                    whole_pred_coarse_vis = []
                    vis_feature_maps = []
                    for idx, item in enumerate(match_res):
                        pred_coarse_label = item['pred_coarse_label']
                        pred_label = item['pred_label']
                        gt_label = item['gt_label']
                        vis_feature_data = item['vis_feature_data']
                        vis_feature_maps.append(vis_feat(vis_feature_data["pts1"], vis_feature_data["pts2"], vis_feature_data["feat1"], vis_feature_data["feat2"]))
                        # visualize the matching results 
                        if idx == 0:      
                            vis_gt_match = vis_gt_matching(vis_img_1, vis_img_2, sampled_choose_1, sampled_choose_2, gt_label)
                            vis_pred_match = vis_pred_matching(vis_img_1, vis_img_2, sampled_choose_1, sampled_choose_2, pred_label, gt_label=gt_label)
                            vis_pred_coarse_match = vis_pred_matching(vis_img_1, vis_img_2, sampled_choose_1, sampled_choose_2, pred_coarse_label, gt_label=gt_label)
                            whole_gt_vis.append(vis_gt_match)
                            whole_pred_vis.append(vis_pred_match)
                            whole_pred_coarse_vis.append(vis_pred_coarse_match)
                        elif idx == 1:
                            vis_gt_match = vis_gt_matching(vis_img_1, vis_img_3, sampled_choose_1, sampled_choose_3, gt_label)
                            vis_pred_match = vis_pred_matching(vis_img_1, vis_img_3, sampled_choose_1, sampled_choose_3, pred_label, gt_label=gt_label)
                            vis_pred_coarse_match = vis_pred_matching(vis_img_1, vis_img_3, sampled_choose_1, sampled_choose_3, pred_coarse_label, gt_label=gt_label)
                            whole_gt_vis.append(vis_gt_match)
                            whole_pred_vis.append(vis_pred_match)
                            whole_pred_coarse_vis.append(vis_pred_coarse_match)
                        elif idx == 2:
                            vis_gt_match = vis_gt_matching(vis_img_2, vis_img_3, sampled_choose_2, sampled_choose_3, gt_label)
                            vis_pred_match = vis_pred_matching(vis_img_2, vis_img_3, sampled_choose_2, sampled_choose_3, pred_label, gt_label=gt_label)
                            vis_pred_coarse_match = vis_pred_matching(vis_img_2, vis_img_3, sampled_choose_2, sampled_choose_3, pred_coarse_label, gt_label=gt_label)
                            whole_gt_vis.append(vis_gt_match)
                            whole_pred_vis.append(vis_pred_match)
                            whole_pred_coarse_vis.append(vis_pred_coarse_match)
                        else:
                            raise NotImplementedError
                    vis_gt_match = pad_to_max_width(whole_gt_vis)
                    vis_pred_match = pad_to_max_width(whole_pred_vis)
                    vis_pred_coarse_match = pad_to_max_width(whole_pred_coarse_vis)
                    vis_feature_maps = pad_to_max_width(vis_feature_maps)
                    # cv2.imwrite("matches_gt.png", vis_gt_match)
                    # cv2.imwrite("matches_pred.png", vis_pred_match)
                    
                log_images(self.logger, {
                    "train/vis_depth": wandb.Image(vis_depth_res, caption="train vis depth"),
                    "train/vis_pts": wandb.Image(vis_pts_res, caption="train vis pts"),
                    "train/gt_pts3d": wandb.Object3D(vis_gt_pts_res_3d),
                    "train/pred_pts3d": wandb.Object3D(vis_pred_pts_res_3d),
                    "train/vis_feature": wandb.Image(vis_feature_maps, caption="train vis feature"),
                    "train/vis_pred_coarse_match": wandb.Image(vis_pred_coarse_match, caption="train vis pred coarse match"),
                    "train/vis_pred_match": wandb.Image(vis_pred_match, caption="train vis match"),
                    "train/vis_gt_match": wandb.Image(vis_gt_match, caption="train vis gt match")
                })

        return loss 
    
    def validation_step(self, batch, batch_idx):
        images = batch["images"]
        masks = batch["masks"]
        if "positive_frames" in batch:
            positive_frames = batch["positive_frames"]
        else:
            positive_frames = None
        sampled_choose = batch["sampled_choose"]
        sampled_pts = batch["sampled_pts"]

        with torch.no_grad():
            predictions = self.forward(images, sampled_choose, sampled_pts, masks, positive_frames)
            loss = 0
            # Compute the depth loss
            if self.depth_head is not None:
                pred_depth = predictions["depth"]
                gt_depth = batch["gt_depthmaps"]
                depth_loss_dict = self.depth_loss(predictions, batch)
                depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
                loss += depth_loss
            # Compute the point loss
            if self.point_head is not None:
                pred_pts = predictions["points"]
                conf_pts = predictions["points_conf"]
                gt_pts = batch["gt_pointmaps"]

                pts_loss_dict = self.pts_loss(pred_pts, conf_pts, gt_pts, masks) # vggt
                loss_conf_point = pts_loss_dict['loss_conf_point']
                loss_reg_point = pts_loss_dict['loss_reg_point']
                loss_grad_point = pts_loss_dict['loss_grad_point']
                point_loss = loss_conf_point + loss_reg_point + loss_grad_point
                loss += point_loss
            # Compute the match loss
            if self.track_head is not None:
                match_res = predictions["match_res"]
                match_loss = sum(item['loss'] for item in match_res)[0] / len(match_res)
                loss += match_loss
                    
            self.val_loss_epoch.update(loss.detach().cpu().item())
            self.val_depth_loss.update(depth_loss.detach().cpu().item())
            self.val_pts_loss.update(point_loss.detach().cpu().item())
            self.val_match_loss.update(match_loss.detach().cpu().item())

            safe_log(
                self,
                "val_loss",
                loss.detach().cpu().item(),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                sync_dist=False,
            )

            # Visualization logging
            if self.config.enable_plotting:
                # if self.global_step % self.config.log_every_n_steps == 0:
                if batch_idx % 10 == 0:
                    num_ref = images.shape[1]
                    vis_depth_res = []
                    vis_pts_res = []
                    for idx in range(num_ref):
                        file_path = batch["filepaths"][idx][0]
                        vis_RGB = images[0, idx].permute(1, 2, 0).cpu().numpy()
                        vis_mask = masks[0, idx].cpu().numpy()
                        if self.depth_head is not None:
                            # visualize depth map
                            vis_depth_res.append(visualize_depth_res(vis_RGB, gt_depth[0, idx], pred_depth[0, idx].squeeze(-1), vis_mask))
                        if self.point_head is not None:
                            # visualize pts map
                            vis_gt_pts = visualize_pts(gt_pts[0, idx].cpu().numpy(), vis_mask, background_color=(0, 0, 0))
                            vis_pred_pts = visualize_pts(pred_pts[0, idx].detach().cpu().numpy(), vis_mask, background_color=(0, 0, 0))
                            vis_pts_res.append(visualize_RGB_pts(file_path, vis_RGB, vis_gt_pts, vis_pred_pts, dpi=100))
                    
                    if self.depth_head is not None:
                        vis_depth_res = pad_to_max_width(vis_depth_res)
                    if self.point_head is not None:
                        vis_pts_res = pad_to_max_width(vis_pts_res)                            
                        # visualize in 3d space
                        query_gt_pts = gt_pts[0, 0].detach().cpu().numpy() * masks[0, 0][:,:,None].cpu().numpy()
                        ref_gt_pts = gt_pts[0, 1].detach().cpu().numpy() * masks[0, 1][:,:,None].cpu().numpy()
                        vis_gt_pts_res_3d = visualize_pts_in_3d(query_gt_pts.reshape(-1,3), ref_gt_pts.reshape(-1,3))

                        query_pred_pts = pred_pts[0, 0].detach().cpu().numpy() * masks[0, 0][:,:,None].cpu().numpy()
                        ref_pred_pts = pred_pts[0, 1].detach().cpu().numpy() * masks[0, 1][:,:,None].cpu().numpy()
                        vis_pred_pts_res_3d = visualize_pts_in_3d(query_pred_pts.reshape(-1,3), ref_pred_pts.reshape(-1,3))
                    
                    if self.track_head is not None:
                        # for visualization
                        sampled_choose_1 = sampled_choose[0, 0, :]
                        sampled_choose_2 = sampled_choose[0, 1, :]
                        sampled_choose_3 = sampled_choose[0, 2, :]
                        positive_images = images[positive_frames]
                        vis_img_1 = (positive_images[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                        vis_img_2 = (positive_images[1].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                        vis_img_3 = (positive_images[2].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                        whole_gt_vis = []
                        whole_pred_vis = []
                        whole_pred_coarse_vis = []
                        vis_feature_maps = []
                        for idx, item in enumerate(match_res):
                            pred_coarse_label = item['pred_coarse_label']
                            pred_label = item['pred_label']
                            gt_label = item['gt_label']
                            vis_feature_data = item['vis_feature_data'] 
                            vis_feature_maps.append(vis_feat(vis_feature_data["pts1"], vis_feature_data["pts2"], vis_feature_data["feat1"], vis_feature_data["feat2"]))                                       
                            # visualize the matching results 
                            if idx == 0:      
                                vis_gt_match = vis_gt_matching(vis_img_1, vis_img_2, sampled_choose_1, sampled_choose_2, gt_label)
                                vis_pred_match = vis_pred_matching(vis_img_1, vis_img_2, sampled_choose_1, sampled_choose_2, pred_label, gt_label=gt_label)
                                vis_pred_coarse_match = vis_pred_matching(vis_img_1, vis_img_2, sampled_choose_1, sampled_choose_2, pred_coarse_label, gt_label=gt_label)
                                whole_gt_vis.append(vis_gt_match)
                                whole_pred_vis.append(vis_pred_match)
                                whole_pred_coarse_vis.append(vis_pred_coarse_match)
                            elif idx == 1:
                                vis_gt_match = vis_gt_matching(vis_img_1, vis_img_3, sampled_choose_1, sampled_choose_3, gt_label)
                                vis_pred_match = vis_pred_matching(vis_img_1, vis_img_3, sampled_choose_1, sampled_choose_3, pred_label, gt_label=gt_label)
                                vis_pred_coarse_match = vis_pred_matching(vis_img_1, vis_img_3, sampled_choose_1, sampled_choose_3, pred_coarse_label, gt_label=gt_label)
                                whole_gt_vis.append(vis_gt_match)
                                whole_pred_vis.append(vis_pred_match)
                                whole_pred_coarse_vis.append(vis_pred_coarse_match)
                            elif idx == 2:
                                vis_gt_match = vis_gt_matching(vis_img_2, vis_img_3, sampled_choose_2, sampled_choose_3, gt_label)
                                vis_pred_match = vis_pred_matching(vis_img_2, vis_img_3, sampled_choose_2, sampled_choose_3, pred_label, gt_label=gt_label)
                                vis_pred_coarse_match = vis_pred_matching(vis_img_2, vis_img_3, sampled_choose_2, sampled_choose_3, pred_coarse_label, gt_label=gt_label)
                                whole_gt_vis.append(vis_gt_match)
                                whole_pred_vis.append(vis_pred_match)
                                whole_pred_coarse_vis.append(vis_pred_coarse_match)
                            else:
                                raise NotImplementedError
                        vis_gt_match = pad_to_max_width(whole_gt_vis)
                        vis_pred_match = pad_to_max_width(whole_pred_vis)
                        vis_pred_coarse_match = pad_to_max_width(whole_pred_coarse_vis)
                        vis_feature_maps = pad_to_max_width(vis_feature_maps)                   
                        # vis_gt_match = cv2.cvtColor(vis_gt_match, cv2.COLOR_RGB2BGR)
                        # vis_pred_match = cv2.cvtColor(vis_pred_match, cv2.COLOR_RGB2BGR)
                        # vis_pred_coarse_match = cv2.cvtColor(vis_pred_coarse_match, cv2.COLOR_RGB2BGR)
                        # cv2.imwrite("matches_gt.png", vis_gt_match)
                        # cv2.imwrite("matches_pred.png", vis_pred_match)
                        # cv2.imwrite("matches_pred_coarse.png", vis_pred_coarse_match)
                        
                    log_images(self.logger, {
                        "val/vis_depth": wandb.Image(vis_depth_res, caption="val vis depth"),
                        "val/vis_pts": wandb.Image(vis_pts_res, caption="val vis pts"),
                        "val/gt_pts3d": wandb.Object3D(vis_gt_pts_res_3d),
                        "val/pred_pts3d": wandb.Object3D(vis_pred_pts_res_3d),
                        "val/vis_feature_maps": wandb.Image(vis_feature_maps, caption="val vis feature maps"),
                        "val/vis_pred_coarse_match": wandb.Image(vis_pred_coarse_match, caption="val vis coarse match"),
                        "val/vis_pred_match": wandb.Image(vis_pred_match, caption="val vis match"),
                        "val/vis_gt_match": wandb.Image(vis_gt_match, caption="val vis gt match")
                    })

        return loss 
                
    def on_train_epoch_start(self):
        self.train_loss_epoch.reset()
        self.train_depth_loss.reset()
        self.train_pts_loss.reset()
        self.train_match_loss.reset()

    def on_train_epoch_end(self):
        avg_train_loss = self.train_loss_epoch.compute()
        avg_train_depth_loss = self.train_depth_loss.compute()
        avg_train_pts_loss = self.train_pts_loss.compute()
        avg_train_match_loss = self.train_match_loss.compute()
        self.log_dict(
            {
                "train_loss_epoch": avg_train_loss,
                "train_depth_loss_epoch": avg_train_depth_loss,
                "train_pts_loss_epoch": avg_train_pts_loss,
                "train_match_loss_epoch": avg_train_match_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        self.val_loss_epoch.reset()
        self.val_depth_loss.reset()
        self.val_pts_loss.reset()
        self.val_match_loss.reset()

    def on_validation_epoch_end(self):
        avg_val_loss = self.val_loss_epoch.compute()
        avg_val_depth_loss = self.val_depth_loss.compute()
        avg_val_pts_loss = self.val_pts_loss.compute()
        avg_val_match_loss = self.val_match_loss.compute()
        self.log_dict(
            {
                "val_loss_epoch": avg_val_loss,
                "val_depth_loss_epoch": avg_val_depth_loss,
                "val_pts_loss_epoch": avg_val_pts_loss,
                "val_match_loss_epoch": avg_val_match_loss,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.learning_rate)

        scheduler = get_scheduler(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            hold_steps=self.config.hold_steps,
            decay_end=self.config.decay_end_steps,
            base_lr=self.config.learning_rate,
            min_lr=self.config.min_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 按 iteration 调整
                "frequency": 1,
            }
        }
        



    

        

    