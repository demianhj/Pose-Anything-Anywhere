import os
import torch
import numpy as np
from PIL import Image
import yaml
import pandas as pd

import sys
sys.path.append(os.getcwd())
from vggt.utils.geometry import unproject_depth_map_to_point_map, depth_to_cam_coords_points
from hydra import compose, initialize_config_dir
from inference_utils.visualization import *
from inference_utils.utils import center_crop, crop_input, to_tensor
from inference_utils.model import load_model
from inference_utils.pose_estimation import robust_umeyama, estimate_pose_from_2d3d 

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule
from inference_utils.oryon_utils import viz
from training.vis_utils import visualize_pts
from inference_utils.datasets import LMO_Dataset
from inference_utils.oryon_utils.pcd import get_diameter
from inference_utils.oryon_utils.metrics import compute_add, compute_adds
from inference_utils.oryon_utils.misc import format_sym_set

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
            
        config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "training"))
        with initialize_config_dir(version_base=None, config_dir=config_dir):
            config = compose(config_name=args.model_config)
    
        # Initialize the VGGT model
        if args.ckpt_path:
            self.model = load_model(config, args.ckpt_path)

        self.model = self.model.to(self.device).eval()

        self.metrics = {}
        self.pose_res = []
        self.output_base = args.output_base 
       
############ GETTER METHODS ##################

    def get_dataset(self) -> torch.utils.data.Dataset:
        dataset_name = self.args.dataset.name
        
        if dataset_name == "lmo":
            return LMO_Dataset(self.args)
        else:
            raise RuntimeError(f"Dataset {dataset_name} not supported")

################ TEST ################################
    def on_test_start(self):
        self.add_object_info(*self.test_dataset.get_object_info())
        return super().on_test_start()

    def test_step(self, batch):
        BS = batch['anchor_image'].shape[0]
        target_size = (518, 518)
        for i_b in range(BS):
            obj_id = batch['obj_id'][i_b].cpu().item()
            scene_id = batch['scene_id'][i_b].cpu().item()
            image_id = batch['image_id'][i_b].cpu().item()
            anchor_image = batch['anchor_image'][i_b]
            anchor_mask = batch['anchor_mask'][i_b]
            anchor_depth = batch['anchor_depth'][i_b]
            anchor_camera = batch['anchor_intri'][i_b]
            anchor_pose = batch['anchor_extri'][i_b]

            query_image = batch['query_image'][i_b]
            query_mask = batch['query_mask'][i_b]
            query_depth = batch['query_depth'][i_b]
            query_camera = batch['query_intri'][i_b]
            query_gt_pose = batch['query_gt_pose'][i_b].cpu().numpy() # for eval

            # process the anchor image and mask
            anchor_image, anchor_mask, anchor_depth, anchor_camera, anchor_pose = center_crop(anchor_image, 
                                                                                    anchor_mask, 
                                                                                    anchor_depth, 
                                                                                    anchor_camera, 
                                                                                    anchor_pose, 
                                                                                    target_image_shape=target_size)
            anchor_image = Image.fromarray(anchor_image.astype(np.uint8))
            anchor_image = to_tensor(anchor_image)
            anchor_depth = anchor_depth.astype(np.float32)
            anchor_point_cloud = unproject_depth_map_to_point_map(anchor_depth[None], anchor_pose[None], anchor_camera[None])[0]
            
            # load the query image and mask
            query_image, query_mask, query_depth, updated_cam_K, _ = crop_input(query_image, 
                                                                             query_mask, 
                                                                             query_depth, 
                                                                             query_camera, 
                                                                             target_size=target_size)
            query_mask = query_mask.astype(bool)

            if query_mask.sum() < 100:
                print(f"Skip empty mask for scene {scene_id}, image {image_id}, object {obj_id}")
                continue

            anchor_image = anchor_image.unsqueeze(0) 
            query_image = query_image.unsqueeze(0) 
            images = torch.cat([anchor_image, query_image], dim=0).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    images = images[None]  # add batch dimension
                    aggregated_tokens_list, ps_idx = self.model.aggregator(images)

                # Predict Point Maps
                point_map, point_conf = self.model.point_head(aggregated_tokens_list, images, ps_idx)
                point_map = point_map.squeeze(0).cpu().numpy() 
                point_conf = point_conf.squeeze(0).cpu().numpy()
                      
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
                pred_query_aligned = scale * (pred_query_pc @ R.T) + t
                pred_query_aligned = pred_query_aligned * query_mask[:, :, None]  # shape (H, W, 3)

                # # PnP
                pred_pose = estimate_pose_from_2d3d(pred_query_aligned, query_mask, updated_cam_K)
                
                # save the pose result
                pred_pose = torch.from_numpy(pred_pose).float()
                R_flat = pred_pose[:3, :3].reshape(-1)
                t_flat = pred_pose[:3, 3]  
                self.pose_res.append(
                    {
                    "scene_id": scene_id,
                    "im_id": image_id,
                    "obj_id": obj_id,
                    "score": 1.0,  # Placeholder for score, can be updated later
                    "R": " ".join(map(str, R_flat.tolist())),
                    "t": " ".join(map(str, t_flat.tolist())),
                    "time": 0.0,  # Placeholder for time, can be updated later
                    }
                )
            # Eval the prediction -----------------------------------------------------
            obj_model, obj_diam, obj_sym = self.get_obj_info(obj_id)
            add_diam = get_diameter(obj_model['pts'])

            if obj_sym.shape[0] > 1:
                add = compute_add(obj_model['pts'], pred_pose.numpy(), query_gt_pose)
                adds = compute_adds(obj_model['pts'], pred_pose.numpy(), query_gt_pose)
            else:
                add = compute_add(obj_model['pts'], pred_pose.numpy(), query_gt_pose)
                adds = add

            if obj_id not in self.metrics:
                self.metrics[obj_id] = {
                    'ADD-0.1d': [],
                    'ADD(S)-0.1d': []
                }
            self.metrics[obj_id]['ADD(S)-0.1d'].append(float(adds <= add_diam * 0.1))
            self.metrics[obj_id]['ADD-0.1d'].append(float(add <= add_diam * 0.1))
            try:
                # visualization nocs
                os.makedirs(f"{self.output_base}/nocs_vis/{obj_id:02d}", exist_ok=True)
                save_path = os.path.join(f"{self.output_base}/nocs_vis/{obj_id:02d}", f"{scene_id:06d}_{image_id:06d}.png")
                # concat the ref nocs and pred nocs
                vis_anchor_nocs = visualize_pts(pred_anchor_pc, mask=anchor_mask)
                vis_query_nocs = visualize_pts(pred_query_pc, mask=query_mask)
                # resize the nocs to 192 x 192
                vis_anchor_nocs = cv2.resize(vis_anchor_nocs, (192,192), interpolation=cv2.INTER_NEAREST)
                vis_query_nocs = cv2.resize(vis_query_nocs, (192,192), interpolation=cv2.INTER_NEAREST)
                nocs_vis = np.concatenate([vis_anchor_nocs, vis_query_nocs], axis=1)
                viz.save_array_to_image(nocs_vis, save_path)
            except Exception as e:
                print(f"Visualization failed for scene {scene_id}, image {image_id}, object {obj_id}: {e}")


    def on_test_end(self):
        '''
        Aggregate and print final results
        '''   
        # save the pose results to a csv file
        csv_path = os.path.join(self.output_base, f'pose_results.csv')
        df = pd.DataFrame(self.pose_res)
        df.to_csv(csv_path, index=False)

        # save the metrics to a log file
        log_path = os.path.join(self.output_base, 'log.txt')
        with open(log_path, 'w') as log_file:
            for obj_id in sorted(self.metrics.keys()):
                add_vals = self.metrics[obj_id]['ADD-0.1d']
                adds_vals = self.metrics[obj_id]['ADD(S)-0.1d']
                
                add_mean = np.mean(add_vals) if add_vals else 0.0
                adds_mean = np.mean(adds_vals) if adds_vals else 0.0
                
                line = f'Object {obj_id:02d} - ADD-0.1d: {add_mean:.4f}, ADD(S)-0.1d: {adds_mean:.4f}'
                print(line)
                log_file.write(line + '\n')

            all_add = []
            all_adds = []
            for obj_id in self.metrics:
                all_add.extend(self.metrics[obj_id]['ADD-0.1d'])
                all_adds.extend(self.metrics[obj_id]['ADD(S)-0.1d'])

            total_add_line = f'Total ADD-0.1d: {np.mean(all_add):.4f}'
            total_adds_line = f'Total ADD(S)-0.1d: {np.mean(all_adds):.4f}'
            
            print(total_add_line)
            print(total_adds_line)
            log_file.write(total_add_line + '\n')
            log_file.write(total_adds_line + '\n')

        return super().on_test_end()
    
    def add_object_info(self, obj_models: dict, obj_diams: dict, obj_symms: dict):
        # these are supposed to be in mm!
        self.obj_models = obj_models
        self.obj_diams = obj_diams
        self.obj_symms = {k: format_sym_set(sym_set) for k, sym_set in obj_symms.items()}

    def get_obj_info(self, obj_id):
        '''
        Get object ID
        '''
        return self.obj_models[obj_id], self.obj_diams[obj_id], self.obj_symms[obj_id]

###################### DATALOADERS ######################
    def get_test_dataloader(self):

        args = self.args
        
        test_set = self.get_dataset()
        self.test_dataset = test_set
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=args.dataset.batch_size,
            shuffle=False,
            num_workers=8
        )

        return test_loader


def eval_pipeline(args: DictConfig) -> None:

    torch.set_float32_matmul_precision('medium')
    system = PANY_Pipeline(args)

    trainer = Trainer(
        logger=None,
        profiler=None,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        accelerator="cuda",
        precision=32,
        log_every_n_steps=10,
        devices=1,
        num_nodes=1,
    )
    test_data = system.get_test_dataloader()
    trainer.test(system, test_data)

if __name__ == '__main__':
    with open("scripts/configs/lmo_rgb.yaml", "r") as f:
        raw_cfg = yaml.safe_load(f)
    
    args = OmegaConf.create(raw_cfg)
    
    eval_pipeline(args)
