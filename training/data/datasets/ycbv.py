# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path as osp
import logging
from collections import defaultdict
import torch
import cv2
import random
import numpy as np
from tqdm import tqdm  

# import open3d as o3d
from torch.utils.data import DataLoader, RandomSampler
from pytorch_lightning import LightningDataModule
from torchvision import transforms as TF
from scipy.spatial import cKDTree

import sys
sys.path.append('/home/tum/Documents/vggt')
from training.data.dataset_util import *
from training.data.base_dataset import BaseDataset
from training.data.augmentation import get_image_augmentation
from training.data.datasets_utils import misc
from training.data.datasets_utils.structs import CameraModel
from vggt.utils.geometry import unproject_depth_map_to_point_map
from training.vis_utils import visualize_matches
    
class YCBVDataModule(LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, steps_per_epoch):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.steps_per_epoch= steps_per_epoch

    def train_dataloader(self):
        # replacement=True 允许放回抽样，保证无论 dataset 多大都能抽够 samples_per_epoch
        sampler = RandomSampler(
            self.dataset,
            replacement=True,
            num_samples=self.steps_per_epoch * self.batch_size
        )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            drop_last=True,   # 保证每个 epoch 都整除 batch_size
        )
    
class YCBVDataset(BaseDataset):
    def __init__(
        self,
        common_conf
    ):
        """
        Initialize the YCBVDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            YCBV_DIR (str): Directory path to YCBV data.
            YCBV_ANNOTATION_DIR (str): Directory path to YCBV annotations.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        Raises:
            ValueError: If YCBV_DIR or YCBV_ANNOTATION_DIR is not specified.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.batch_size = common_conf.batch_size
        self.load_depth = common_conf.load_depth
        self.mask_depth = common_conf.mask_depth         
        self.to_tensor = TF.ToTensor()
        self.crop_rel_pad = common_conf.crop_rel_pad
        self.view_outplane_thr = common_conf.view_outplane_thr # degree, threshold to distinguish nearby and distant views
        self.view_inplane_thr = common_conf.view_inplane_thr # degree, threshold to distinguish in-plane rotation difference
        self.YCBV_DIR = common_conf.data.train_base
        self.num_views = common_conf.num_views
        self.n_sample_point = common_conf.n_sample_point

        # --- Augmentation Settings ---
        # Controls whether to apply identical color jittering across all frames in a sequence
        self.cojitter = common_conf.augs.cojitter
        # Probability of using shared jitter vs. frame-specific jitter
        self.cojitter_ratio = common_conf.augs.cojitter_ratio
        # Initialize image augmentations (color jitter, grayscale, gaussian blur)
        self.image_aug = get_image_augmentation(
            color_jitter=common_conf.augs.color_jitter,
            gray_scale=common_conf.augs.gray_scale,
            gau_blur=common_conf.augs.gau_blur,
        )
        logging.info(f"YCBV_DIR is {self.YCBV_DIR}")
        
        self.scene_ids = [0, 1, 2, 3, 4]
        if self.scene_ids == []:
            self.scene_ids = list(range(50))
        self.obj_ids = list(range(1,22))
        # ---------------------------
        # Get the 3D annotation data
        self.total_data = []
        self.annotation = defaultdict(list)
        rgb_tpath = f"{self.YCBV_DIR}/{{scene_id:06d}}/rgb/{{image_id:06d}}.jpg"
        depth_tpath = f"{self.YCBV_DIR}/{{scene_id:06d}}/depth/{{image_id:06d}}.png"
        mask_tpath = f"{self.YCBV_DIR}/{{scene_id:06d}}/mask_visib/{{image_id:06d}}_{{inst_id:06d}}.png"
        for scene_id in tqdm(self.scene_ids):
            scene_camera_path = osp.join(self.YCBV_DIR, f"{scene_id:06d}", "scene_camera.json")
            scene_gt_path = osp.join(self.YCBV_DIR, f"{scene_id:06d}", "scene_gt.json")
            scene_gt_info_path = osp.join(self.YCBV_DIR, f"{scene_id:06d}", "scene_gt_info.json")
            with open(scene_camera_path, "r") as f:
                scene_camera = json.load(f)
            with open(scene_gt_path, "r") as f:
                scene_gt = json.load(f)
            with open(scene_gt_info_path, "r") as f:
                scene_gt_info = json.load(f)

            image_ids = list(scene_camera.keys())
            for image_id in image_ids:
                rgb_path = rgb_tpath.format(scene_id=scene_id, image_id=int(image_id))
                depth_path = depth_tpath.format(scene_id=scene_id, image_id=int(image_id))
                current_frame_gt = scene_gt[image_id]
                current_frame_gt_info = scene_gt_info[image_id]
                current_camera = scene_camera[image_id]
                for inst_id, inst in enumerate(current_frame_gt):
                    inst_gt_info = current_frame_gt_info[inst_id]
                    if inst_gt_info["visib_fract"] < 0.8:
                        # logging.warning(f"Instance {inst_id} in image {image_id} has visibility fraction < 0.5, skipping.")
                        continue
                    obj_id = inst["obj_id"]
                    mask_path = mask_tpath.format(scene_id=scene_id, image_id=int(image_id), inst_id=int(inst_id))
                    # check if mask is valid (the image size is [480,640] and assume the mask is valid if larger than 1%)
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255.0
                    if mask is None or mask.shape != (480, 640) or np.sum(mask > 0) < 0.001 * 480 * 640:
                        # logging.warning(f"Invalid mask at {mask_path}, skipping this instance.")
                        continue
                    
                    intri = np.array(current_camera["cam_K"]).reshape(3, 3)
                    extri = np.eye(4)
                    extri[:3,:3] = np.array(inst["cam_R_m2c"]).reshape(3,3)
                    extri[:3,3] = np.array(inst["cam_t_m2c"]).reshape(3)

                    data_idx = len(self.total_data)
                    self.total_data.append({
                        "image_path": rgb_path,
                        "depth_path": depth_path,
                        "mask_path": mask_path,
                        "obj_id": obj_id,
                        "extri": extri.tolist(),
                        "intri": intri.tolist(),
                        "extri_R": extri[:3, :3],
                        })
                    self.annotation[obj_id].append({
                        "data_idx": data_idx,
                        "extri_R": extri[:3, :3],
                    })

        self.len_train = len(self.total_data)     
        self.indices = self.create_index_list(self.len_train, self.batch_size)              
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Omni6D Data size: {self.len_train}")
        logging.info(f"{status}: Omni6D Data length of training set: {len(self)}")
    
    def create_index_list(self, dataset_len, batch_size):
        indices = []
        for i in range(0, dataset_len, batch_size):
            img_per_seq = 3
            # img_per_seq = np.random.randint(self.num_views[0], self.num_views[1])
            aspect_ratio = 1.0  # Fixed aspect ratio for simplicity
            # For each item in the batch, use the same img_per_seq and aspect_ratio
            for j in range(batch_size):
                indices.append((i + j, img_per_seq, aspect_ratio))
        return indices

    def compute_out_plane_rotation(self, R1, R_batch):
        """
        计算参考旋转矩阵 R1 与一批旋转矩阵之间的平面外旋转误差。

        参数:
            R1 (np.array): (3,3) 参考旋转矩阵 (真实姿态)。
            R_batch (np.array): (N,3,3) 一批旋转矩阵 (预测姿态)。

        返回:
            np.array: (N,) 每个候选与 R1 的平面外旋转误差角度（度）。
        """
        R1 = np.asarray(R1).reshape(3, 3)
        R_batch = np.asarray(R_batch).reshape(-1, 3, 3)

        # 相对旋转: R_err = R2 * R1^T
        R_err = R_batch @ R1.T   # (N,3,3)

        # 提取 cos(angle) = R_err[2,2]
        cos_angles = R_err[:, 2, 2]

        # 限制范围并转角度
        angles_rad = np.arccos(np.clip(cos_angles, -1.0, 1.0))
        angles_deg = np.degrees(angles_rad)
        return angles_deg

    def compute_in_plane_rotation(self, R1, R_batch):
        """
        计算参考旋转矩阵 R1 与一批旋转矩阵之间的平面内旋转差异（roll差异）。

        Args:
            R1 (np.array): (3,3) 参考旋转矩阵
            R_batch (np.array): (N,3,3) 一批旋转矩阵

        Returns:
            np.array: (N,) 每个候选与 R1 的 in-plane rotation 差异角度（度）
        """
        R1 = np.asarray(R1).reshape(3, 3)
        R_batch = np.asarray(R_batch).reshape(-1, 3, 3)

        # 相对旋转: R_err = R2 * R1^T
        R_err = R_batch @ R1.T   # (N,3,3)

        # 提取绕 z 轴的旋转角度
        angles_rad = np.arctan2(R_err[:, 1, 0], R_err[:, 0, 0])
        angles_deg = np.degrees(angles_rad)

        # 把结果映射到 [0, 180]
        angles_deg = np.abs(angles_deg)
        angles_deg = np.where(angles_deg > 180, 360 - angles_deg, angles_deg)

        return angles_deg
    
    def normalize_pointcloud(self, pts3d, valid_mask, eps=1e-3):
        """
        pts3d: S, H, W, 3
        valid_mask: S, H, W
        """
        # 1) per‐point Euclidean distance to the origin
        dist = np.linalg.norm(pts3d, axis=-1)

        # 2) sum up only the valid ones, count them
        dist_sum = np.sum(dist * valid_mask)   # scalar
        valid_count = np.sum(valid_mask)          # scalar

        # 3) the “scale” is their mean distance, clamped
        avg_scale = dist_sum / (valid_count + eps)
        avg_scale = np.clip(avg_scale, a_min=eps, a_max=5e3)

        # 4) divide everything by that one number
        pts3d_norm = pts3d / avg_scale

        return pts3d_norm, avg_scale
  
    def get_data(self, seq_index, img_per_seq, aspect_ratio=1.0):
        """
        Retrieve data for a specific sequence.

        Args:
            ids (list): Specific IDs to retrieve.
        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        # check if first frame is valid
        first_frame_data = self.total_data[seq_index]
        first_frame_obj_id = first_frame_data['obj_id']
        
        # get the reference views
        reference_cands = self.annotation[first_frame_obj_id]
        reference_cands_rot = np.array([item['extri_R'] for item in reference_cands])
        first_frame_rot = first_frame_data['extri_R']
        rot_diff = self.compute_out_plane_rotation(first_frame_rot, reference_cands_rot)

        # select the N reference views (2 nearby + N-2 distant) 
        num_reference = img_per_seq - 1
        nearby_cands = [item for i, item in enumerate(reference_cands) if rot_diff[i] < self.view_outplane_thr and item['data_idx'] != seq_index]
        if len(nearby_cands) < 2:
            return None
        # # make sure the in plane rotation difference is in the thr
        # rot_diff_in = self.compute_in_plane_rotation(first_frame_rot, np.array([item['extri_R'] for item in nearby_cands]))
        # nearby_cands = [item for i, item in enumerate(nearby_cands) if rot_diff_in[i] > self.view_inplane_thr]
        # if len(nearby_cands) < 2:
        #     return None
        selected_nearby = random.sample(nearby_cands, 2) # fix the number of nearby views to 2
        
        num_distant = num_reference - 2
        distant_cands = [item for i, item in enumerate(reference_cands) if rot_diff[i] >= self.view_outplane_thr and item['data_idx'] != seq_index]
        if len(distant_cands) < num_distant:
            return None
        selected_distant = random.sample(distant_cands, num_distant)

        # get the view data
        views_with_labels = [(item, True) for item in selected_nearby] + [(item, False) for item in selected_distant]
        random.shuffle(views_with_labels)
        metadata = [first_frame_data] + [self.total_data[item['data_idx']] for item, _ in views_with_labels]
        positive_frames = [True] + [label for _, label in views_with_labels]
        positive_frames = np.array(positive_frames, dtype=bool)

        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        masks = []
        depth_maps = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        filepaths = [] 
        choose_list = []  

        for idx, anno in enumerate(metadata):
            filepath = anno["mask_path"] # For Debug
            image_path = anno["image_path"]
            image = read_image_cv2(image_path)

            if self.load_depth:
                depth_path = anno["depth_path"]
                depth_map = read_depth(depth_path, 0.1)

                if self.mask_depth:
                    mvs_mask_path = anno["mask_path"]
                    mvs_mask = cv2.imread(mvs_mask_path, cv2.IMREAD_GRAYSCALE) > 128
                    depth_map[~mvs_mask] = 0
                    
                    if self.training and self.image_aug is not None:
                        # Convert boolean mask → uint8 (0 or 255)
                        mask_u8 = (mvs_mask > 0).astype(np.uint8) * 255
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        mask_dilated_u8 = cv2.dilate(mask_u8, kernel, iterations=3)
                        mask_dilated = mask_dilated_u8 > 0
                        # apply
                        image[~mask_dilated] = 0
                    else:
                        # Apply the mask to image
                        image[~mvs_mask] = 0
            else:
                depth_map = None
            
            extri_opencv = np.array(anno["extri"])
            intri_opencv = np.array(anno["intri"])
            
            # Centeral Crop the image
            # ----------------------------------
            orig_mask_modal = mvs_mask.astype(np.uint8)
            orig_image_np_hwc = image.copy()
            orig_depth_map = depth_map.copy()
            orig_camera_c2w = CameraModel(
                width=image.shape[1],
                height=image.shape[0],
                f=(intri_opencv[0, 0], intri_opencv[1, 1]),
                c=(intri_opencv[0, 2], intri_opencv[1, 2]),
                T_world_from_eye=np.linalg.inv(extri_opencv))
            
            # Get the bbox from mask
            orig_box_amodal = misc.get_bbox(orig_mask_modal)
            # Get box for cropping.
            crop_box = misc.calc_crop_box(
                box=orig_box_amodal,
                make_square=True,
            )
            # Construct a virtual camera focused on the crop.
            crop_camera_model_c2w = misc.construct_crop_camera(
                box=crop_box,
                camera_model_c2w=orig_camera_c2w,
                viewport_size=target_image_shape,
                viewport_rel_pad=self.crop_rel_pad,
            )

            # Map images to the virtual camera.
            image = misc.warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_image_np_hwc,
                interpolation=cv2.INTER_LINEAR,
            )
            mask = misc.warp_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_image=orig_mask_modal,
                interpolation=cv2.INTER_NEAREST,
            )
            depth_map = misc.warp_depth_image(
                src_camera=orig_camera_c2w,
                dst_camera=crop_camera_model_c2w,
                src_depth_image=orig_depth_map,
            )
            extri = np.linalg.inv(crop_camera_model_c2w.T_world_from_eye)
            fx, fy = crop_camera_model_c2w.f
            cx, cy = crop_camera_model_c2w.c
            intri = np.array([fx, 0, cx,
                              0, fy, cy,
                              0, 0, 1]).reshape(3, 3) 
            # ----------------------------------
            original_size = np.array(image.shape[:2])

            # Compute camera pose
            if idx == 0: # first frame is the reference frame
                ref_extri = extri.copy()
                extri = np.eye(4)
            else:
                # Transform the points to reference system
                relative_pose = extri @ np.linalg.inv(ref_extri)
                extri = relative_pose.copy()

            # shrink the mask
            kernel = np.ones((3, 3), np.uint8)
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            mask = cv2.erode(mask_uint8, kernel, iterations=3)
            mask = (mask > 0).astype(np.uint8)

            # convert image from 255 to 1.0
            image = self.to_tensor(image)

            # sample the points for training
            choose = mask.astype(np.float32).flatten().nonzero()[0]
            if len(choose) < 32:
                return None
            if len(choose) <= self.n_sample_point:
                choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_point, replace=False)
            choose = choose[choose_idx]

            images.append(image)
            masks.append(mask)
            depth_maps.append(depth_map)
            extrinsics.append(extri)
            intrinsics.append(intri)
            original_sizes.append(original_size)
            filepaths.append(filepath)
            choose_list.append(choose)

        images = np.stack(images, axis=0) 
        masks = np.stack(masks, axis=0).astype(bool)
        depth_maps = np.stack(depth_maps, axis=0)
        extrinsics = np.stack(extrinsics, axis=0)  # (S, 4, 4)
        extrinsics = extrinsics[:, :3, :]  # (S, 3, 4), remove the last row
        intrinsics = np.stack(intrinsics, axis=0) # (S, 3, 3)
        choose_list = np.stack(choose_list, axis=0) # (S, n_sample_point)

        # compute point maps
        point_maps = unproject_depth_map_to_point_map(depth_maps, extrinsics, intrinsics)
        # Apply the masks on the point cloud
        point_maps = point_maps * masks[..., None]  # (S, H, W, 3)
        point_maps, scales = self.normalize_pointcloud(point_maps, masks)
        # update the camera extrinsics and depth maps
        extrinsics[:, :3, 3] /= (scales + 1e-6)
        depth_maps = depth_maps / (scales + 1e-6)

        # get the target points
        positive_pt_maps = point_maps[positive_frames]  # (S', H, W, 3)
        sampled_choose = choose_list[positive_frames]  # (S', n_sample_point)
        sampled_pts = [positive_pt_maps[i].reshape(-1, 3)[sampled_choose[i]] for i in range(len(positive_pt_maps))]
        sampled_pts = np.stack(sampled_pts, axis=0).astype(np.float32) # (S', n_sample_point, 3)

        # --- Apply Color Augmentation (training mode only) ---
        if self.training and self.image_aug is not None:
            images = torch.from_numpy(images).float()  # Convert to tensor
            if self.cojitter and random.random() > self.cojitter_ratio:
                # Apply the same color jittering transformation to all frames
                images = self.image_aug(images)
            else:
                # Apply different color jittering to each frame individually
                for aug_img_idx in range(len(images)):
                    images[aug_img_idx] = self.image_aug(images[aug_img_idx])
            images = images.numpy()  # Convert back to numpy array
        # # visualization ---------------------
        # # check the updated depth map, camera pose and point cloud
        # whole_pts = unproject_depth_map_to_point_map(depth_maps, extrinsics, intrinsics)
        # # Apply the masks on the point cloud
        # whole_pts = whole_pts * masks[..., None]  # (S, H, W, 3)
        # whole_pts = whole_pts.reshape(-1,3)

        # ref_pts = point_maps[:1].reshape(-1,3)  # Use the first frame as reference
        # query_pts = point_maps[1:].reshape(-1,3)
        # all_pts = np.vstack([ref_pts, query_pts, whole_pts])
        # # Create a color array: green for ref, red for query
        # green = np.tile([0.0,1.0,0.0], (len(ref_pts), 1))
        # red = np.tile([1.0,0.0,0.0], (len(query_pts), 1))
        # blue = np.tile([0.0,0.0,1.0], (len(whole_pts), 1))  
        # all_colors= np.vstack([green, red, blue])
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(all_pts)
        # pcd.colors = o3d.utility.Vector3dVector(all_colors)
        # o3d.io.write_point_cloud("pts.ply", pcd)
        # ---------------------
        batch = {
            "frame_num": len(extrinsics),
            "images": images,
            "masks": masks,
            "gt_depthmaps": depth_maps,
            "gt_pointmaps": point_maps,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "original_sizes": original_sizes,
            "filepaths": filepaths, # for debug
            "positive_frames": positive_frames,
            "sampled_pts": sampled_pts,
            "sampled_choose": sampled_choose,
        }
        return batch

if __name__ == "__main__":
    import argparse
    from hydra import initialize, compose
    import time

    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="vggt_sam6d_model",
        help="Name of the config file (without .yaml extension, default: vggt_model)"
    )
    args = parser.parse_args()
    with initialize(version_base=None, config_path="../.."):
        config = compose(config_name=args.config)
    dataset = YCBVDataset(common_conf=config.common_config)

    start = time.time()
    for idx in range(20):
        batch = dataset.get_data(seq_index=idx, img_per_seq=8, aspect_ratio=1.0)
    print("Data fetching time:", time.time() - start)
