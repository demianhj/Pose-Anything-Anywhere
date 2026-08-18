import os
import os.path as osp
import torch
import numpy as np
import cv2
from PIL import Image
import yaml

import sys
sys.path.append(os.getcwd())
from vggt.utils.geometry import unproject_depth_map_to_point_map, depth_to_cam_coords_points
from hydra import initialize, compose
from inference_utils.visualization import *
from inference_utils.utils import center_crop, crop_input, to_tensor
from inference_utils.model import load_model
from inference_utils.pose_estimation import estimate_pose_from_2d3d, robust_umeyama

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.profilers import AdvancedProfiler
from datetime import datetime
from training.vis_utils import visualize_pts
from inference_utils.oryon_utils.datasets import NOCSDataset, TOYLDataset
from inference_utils.oryon_utils.misc import set_deterministic_seed
from inference_utils.oryon_utils import viz
from inference_utils.oryon_utils.evaluator import Evaluator
from inference_utils.refine import get_correspondence_torch

class PANY_Pipeline(LightningModule):
    """
    This class is a PyTorch Lightning system and contain the core of the major steps made during the training of a NN
    """

    def __init__(self, args):
        r"""
        This functions setup the model and the NN loss
        """
        super().__init__()

        self.args = args
        self.evaluator = Evaluator(args.exp_tag, compute_vsd=True, compute_iou=False)

        with initialize(version_base=None, config_path="../training"):
            config = compose(config_name=args.model_config)
    
        # Initialize the PANY model
        if args.ckpt_path:
            self.model = load_model(config, args.ckpt_path)

        self.model = self.model.to(self.device).eval()
        self.use_depth = args.use_depth
       
############ GETTER METHODS ##################

    def get_dataset(self, eval:bool) -> torch.utils.data.Dataset:

        if eval:
            dataset_name = self.args.dataset.test.name
        else:
            dataset_name = self.args.dataset.train.name
        
        if dataset_name == 'nocs':
            return NOCSDataset(self.args, eval)
        elif dataset_name == 'toyl':
            return TOYLDataset(self.args, eval)
        else:
            raise RuntimeError(f"Dataset {dataset_name} not supported")
        
################## VALIDATION ##########################
            
    def valid_visualization(self, batch: dict, pred_pose: np.ndarray, batch_idx: int, test:bool=False):
        '''
        Visualize various information about a validation sample
        Note: pred_corrs and gt corrs here are all in initial input shape
        '''

        if test:
            dataset = self.test_dataset
        else:
            dataset = self.valid_dataset

        gt_pose = batch['query']['pose'][batch_idx].clone().cpu().numpy()

        scene_a, img_a, obj_a = batch['anchor']['instance_id'][batch_idx].split(' ')
        scene_q, img_q, _ = batch['query']['instance_id'][batch_idx].split(' ')
        instance_id = batch['instance_id'][batch_idx]
        
        out_path = f'{self.args.tmp.results_out}/pose_viz/{self.args.dataset.test.name}_{self.args.dataset.test.split}_epoch{self.current_epoch}_{instance_id}_{self.args.test.mask}'

        obj_model, _, _ = dataset.get_obj_info(obj_a)
        obj_model = obj_model['pts'] / 1000.

        item_a = dataset.get_item(int(scene_a), int(img_a), obj_a, 'oracle')
        item_q = dataset.get_item(int(scene_q), int(img_q), obj_a, 'oracle')

        os.makedirs(os.path.dirname(out_path), exist_ok=True)        
        viz.pred_pose(item_a['rgb'], item_q['rgb'], gt_pose, pred_pose, dataset.K, obj_model, out_path+'_pose.png')

################ TEST ################################
    
    def on_test_start(self):
        self.pred_file, self.metric_file = self.get_pred_filename()
        if self.args.debug_valid:
            print("WARNING: USING GROUND TRUTH CORRESPONDENCES!!")

        if self.args.seed is not None:
            set_deterministic_seed(self.args.seed)
        else:
            set_deterministic_seed(1)
        # init metrics file
        self.evaluator.add_object_info(*self.test_dataset.get_object_info())
        self.evaluator.init_test()
        # add to the evaluator the object models
        return super().on_test_start()
    
    def test_step(self, batch):
        BS = batch['anchor']['rgb'].shape[0]
        target_size = (518, 518)
        for i_b in range(BS):
            instance_id_a, instance_id_q = batch['anchor']['instance_id'][i_b], batch['query']['instance_id'][i_b]
            # instance_id = batch['instance_id'][i_b]
            # if instance_id != "3_37_3_51_3":
            #     continue
            #-----------------------------------------------------
            anchor_image = batch['anchor']['orig_rgb'][i_b]
            anchor_mask = batch['anchor']['mask'][i_b]
            anchor_depth = batch['anchor']['orig_depth'][i_b]
            anchor_camera = batch['anchor']['camera'][i_b]
            anchor_pose = batch['anchor']['pose'][i_b]

            query_image = batch['query']['orig_rgb'][i_b]
            query_mask = batch['query']['mask'][i_b]
            query_depth = batch['query']['orig_depth'][i_b]
            query_camera = batch['query']['camera'][i_b]

            anchor_depth = anchor_depth.int()
            query_depth = query_depth.int()

            # correct the mask
            anchor_mask = anchor_mask & (anchor_depth > 0)
            query_mask = query_mask & (query_depth > 0)

            # process the anchor image and mask
            anchor_image = anchor_image * anchor_mask.unsqueeze(0) # Apply mask to image
            anchor_depth = anchor_depth * anchor_mask

            anchor_image, anchor_mask, anchor_depth, anchor_camera, anchor_pose = center_crop(anchor_image, 
                                                                                            anchor_mask, 
                                                                                            anchor_depth, 
                                                                                            anchor_camera, 
                                                                                            anchor_pose, 
                                                                                            target_image_shape=target_size,
                                                                                           )

            anchor_image = Image.fromarray(anchor_image.astype(np.uint8))
            anchor_image = to_tensor(anchor_image)
            anchor_depth = anchor_depth.astype(np.float32)
            anchor_pose[:3, 3] *= 1000
            anchor_point_cloud = unproject_depth_map_to_point_map(anchor_depth[None], anchor_pose[None], anchor_camera[None])[0]
            # vis_pc(anchor_pc.reshape(-1,3))
            
            # load the query image and mask
            query_image, query_mask, query_depth, updated_cam_K, _ = crop_input(query_image, 
                                                                             query_mask, 
                                                                             query_depth, 
                                                                             query_camera, 
                                                                             target_size=target_size,
                                                                             )

            query_mask = query_mask.astype(bool)
            vis_query_mask = query_mask.copy()

            # vis = query_image.permute(1,2,0)
            # image_np = (vis.numpy() * 255).astype(np.uint8)  # scale to 0–255 if needed
            # image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            # cv2.imwrite("query_image.png", image_bgr)

            anchor_image = anchor_image.unsqueeze(0) 
            query_image = query_image.unsqueeze(0) 
            images = torch.cat([anchor_image, query_image], dim=0).to(self.device)
            # images = torch.cat([query_image, anchor_image], dim=0).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    images = images[None]  # add batch dimension
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images)

                # Predict Point Maps
                point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
                point_map = point_map.squeeze(0).cpu().numpy() 
                point_conf = point_conf.squeeze(0).cpu().numpy()
                      
                # # Visualize the results 
                # points_flat = point_map.reshape(-1, 3)
                # colors = images.squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
                # colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
                # masks = np.concatenate([anchor_mask[None], query_mask[None]], axis=0)
                # masks_flat = masks.reshape(-1).astype(bool)
                # save_pointcloud(points=points_flat[masks_flat], colors=colors_flat[masks_flat])
                
                # Align the pred point cloud with the gt point cloud
                pred_anchor_pc = point_map[0]
                pred_query_pc = point_map[1]
                pred_anchor_pc = pred_anchor_pc * anchor_mask[:,:,None]  # shape (H, W, 3)
                pred_query_pc = pred_query_pc * query_mask[:,:,None]  # shape (H, W, 3)

                # Get the corrsponding points between pred and gt in pixel
                src = pred_anchor_pc[anchor_mask > 0]  # shape (N, 3)
                dst = anchor_point_cloud[anchor_mask > 0]   # shape (N, 3)
                conf_anchor = point_conf[0][anchor_mask > 0]
                
                # Estimate transformation and Transform pred_pc
                scale, R, t = robust_umeyama(src, dst, conf_anchor, conf_anchor, with_scaling=True)

                pred_anchor_aligned = (scale * R @ src.T).T + t
                # visualize the aligned point cloud
                # save_pc_together(src.reshape(-1,3), dst.reshape(-1,3))
                # save_pc_together(pred_anchor_aligned.reshape(-1,3), dst.reshape(-1,3))
                # Apply the transformation to the target point cloud which has index 1
                pred_query_aligned = scale * (pred_query_pc @ R.T) + t
                pred_query_aligned = pred_query_aligned * query_mask[:, :, None]  # shape (H, W, 3)
                # save_pc_together(pred_query_aligned.reshape(-1,3), dst.reshape(-1,3))

                # unproject the depth map to 3d points
                query_depth = query_depth * query_mask
                query_pc = depth_to_cam_coords_points(query_depth, updated_cam_K)

                # ----------------------------------------------------
                # pred_query_aligned, anchor_matched, query_mask = get_correspondence(pred_query_aligned, dst, k_nn=5)
                pred_query_aligned, anchor_matched, query_mask = get_correspondence_torch(pred_query_aligned, dst, k_nn=5)
                conf_query = point_conf[1][query_mask > 0]
                query_pc = query_pc[query_mask].reshape(-1,3)
                # ----------------------------------------------------
                # query_pc = query_pc[query_mask].reshape(-1,3)
                # pred_query_aligned = pred_query_aligned[query_mask].reshape(-1,3)
                # conf_query = point_conf[1][query_mask > 0]

                # # remove all the zero points in query_pc and the corrsponding pred_query_aligned
                # non_zero_mask = np.linalg.norm(query_pc, axis=1) > 0
                # query_pc = query_pc[non_zero_mask]
                # pred_query_aligned = pred_query_aligned[non_zero_mask]
                # conf_query = conf_query[non_zero_mask]
                # # save_pc_together(query_pc, pred_query_aligned)

                # umeyama alignment
                if self.use_depth:
                    try:
                        scale, R, t = robust_umeyama(pred_query_aligned, query_pc, conf_query, conf_query, with_scaling=False)
                    except Exception as e:
                        print(f"Error in pose estimation: {e}")
                        scale, R, t = 1.0, np.eye(3), np.zeros(3)
                    # ---------------------------------------------------
                    # transform_pose = np.eye(4)
                    # transform_pose[:3, :3] = R
                    # transform_pose[:3, 3] = t 
                    # transformed_query_pc = cam_coords_points_to_world_coords_points(query_pc, transform_pose[:3, :4])
                        
                    # transformed_query is red and anchor_matched is green
                    # save_pc_together(transformed_query_pc.reshape(-1,3), anchor_matched.reshape(-1,3))
                    # ---------------------------------------------------
                    pred_pose = np.eye(4)
                    pred_pose[:3,:3] = R
                    pred_pose[:3,3] = t / 1000  # convert to meters
                else:
                    # reshape the pred_query_aligned to (H,W,3) by adding back the zero points
                    H, W = query_mask.shape
                    pred_query_aligned_full = np.zeros((H, W, 3))
                    pred_query_aligned_full[query_mask] = pred_query_aligned
                    # get the pose use 2d-3d between target image 2d and pred_target_aligned
                    pred_pose = estimate_pose_from_2d3d(pred_query_aligned_full, query_mask, updated_cam_K)
                    pred_pose[:3,3] /= 1000  # convert to meters
            #-----------------------------------------------------
            pred_pose = torch.from_numpy(pred_pose).float()
            self.evaluator.register_test({
                'gt_pose': batch['query']['pose'][i_b].unsqueeze(0),
                'pred_pose': pred_pose.unsqueeze(0),
                'pred_pose_rel': pred_pose.unsqueeze(0),
                'cls_id': [batch['cls_id'][i_b]],
                'camera' : [batch['query']['camera'][i_b].cpu().numpy()],
                'depth' : [batch['query']['eval_depth'][i_b].squeeze().cpu().numpy()],
                'instance_id': [batch['instance_id'][i_b]]
            })
            self.valid_visualization(batch, pred_pose.clone().numpy(), i_b, test=True)
            print(f"Test {i_b}/{BS} - {instance_id_a} - {instance_id_q}")
            
            self.add_pred_pose(instance_id_a, instance_id_q, pred_pose.cpu().numpy())

            # visualization nocs
            output_dir = f'{self.args.tmp.results_out}/nocs_viz'
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'{self.args.dataset.test.name}_{self.args.dataset.test.split}_epoch{self.current_epoch}_{i_b}_{self.args.test.mask}.png')
            # concat the ref nocs and pred nocs
            vis_anchor_nocs = visualize_pts(pred_anchor_pc, mask=anchor_mask)
            vis_query_nocs = visualize_pts(pred_query_pc, mask=vis_query_mask)
            # resize the nocs to 192 x 192
            vis_anchor_nocs = cv2.resize(vis_anchor_nocs, (192,192), interpolation=cv2.INTER_NEAREST)
            vis_query_nocs = cv2.resize(vis_query_nocs, (192,192), interpolation=cv2.INTER_NEAREST)
            nocs_vis = np.concatenate([vis_anchor_nocs, vis_query_nocs], axis=1)
            viz.save_array_to_image(nocs_vis, save_path)


    def on_test_end(self):
        '''
        Aggregate and print final results
        '''

        self.pred_file.close()

        self.evaluator.test_summary()
        self.evaluator.save(self.metric_file)
        print(self.evaluator.get_latex_str())
        self.metric_file.close()

        return super().on_test_end()

    def get_pred_filename(self):
        '''
        Generate the prediction file name by using timestamps. Returns file pointers of results file and metrics files
        '''
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H%M")
        rand_seed = np.random.randint(0,1000)
        pred_file = f'{self.args.dataset.test.name}_{self.args.dataset.test.split}_{self.args.dataset.test.obj}_{dt_string}_{rand_seed}.csv'
        metric_file = f'{self.args.dataset.test.name}_{self.args.dataset.test.split}_{self.args.dataset.test.obj}_{dt_string}_{rand_seed}.json'
        os.makedirs(self.args.tmp.results_out, exist_ok=True)
        fp = open(os.path.join(self.args.tmp.results_out, pred_file),'w')
        fm = open(os.path.join(self.args.tmp.results_out, metric_file),'w')
        dest = os.path.join(self.args.tmp.results_out,f'config_{dt_string}_{rand_seed}.yaml')
        OmegaConf.save(self.args, dest)
        
        return fp, fm

    def add_pred_pose(self, id_a: str, id_q: str, pred_pose: np.ndarray):
        '''
        Saves a precicted pose in the prediction file
        '''
        pred_pose = ' '.join([str(n) for n in pred_pose[:3,:].flatten()])
        line = ','.join([id_a, id_q, pred_pose])#, rle_a, rle_q])
        line += '\n'
        self.pred_file.write(line)

###################### DATALOADERS ######################
    def get_test_dataloader(self):

        args = self.args
        
        test_set = self.get_dataset(eval=True)
        print("TESTING on {}, split {}, object split {}. Samples: {}".format(test_set.name, test_set.split, test_set.obj, test_set.__len__()))
        self.test_dataset = test_set
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.dataset.batch_size,
            collate_fn=test_set.collate,
            shuffle=False,
            num_workers=0
        )

        return test_loader


def eval_pipeline(args: DictConfig) -> None:

    torch.set_float32_matmul_precision('medium')
    system = PANY_Pipeline(args)
    
    if args.profiler:
        profiler = AdvancedProfiler(args.tmp.logs_out,'profiler_log')
    else:
        profiler = None

    trainer = Trainer(
        logger = None,
        profiler=profiler,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        accelerator="cuda",
        precision=32,
        log_every_n_steps=10,
        devices=1,
        num_nodes=1,
    )
    print(args.exp_tag)
    print("TEST CONFIGURATION:")
    for k,v in args.test.items():
        print(f'{k} : {v}')
    test_data = system.get_test_dataloader()
    trainer.test(system, test_data)

if __name__ == '__main__':
    with open("scripts/configs/toyota_rgbd.yaml", "r") as f:
        raw_cfg = yaml.safe_load(f)

    # with open("scripts/configs/nocs_real275_rgbd.yaml", "r") as f:
    #     raw_cfg = yaml.safe_load(f)
    
    # Optional: Convert dict to OmegaConf if needed
    args = OmegaConf.create(raw_cfg)
    
    eval_pipeline(args)