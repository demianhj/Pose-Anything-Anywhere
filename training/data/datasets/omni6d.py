import os
import json
import cv2
import numpy as np
from openexr_numpy import imread
import torch
import trimesh
import open3d as o3d
import random
import logging
import math
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning import LightningDataModule
from collections import defaultdict
from torchvision import transforms as TF
from scipy.spatial.transform import Rotation

import sys
sys.path.append('/home/tum/Documents/vggt')
from training.data.base_dataset import BaseDataset
from training.data.augmentation import get_image_augmentation
from training.data.datasets_utils import misc
from training.data.datasets_utils.structs import CameraModel
from vggt.utils.geometry import unproject_depth_map_to_point_map
from training.vis_utils import visualize_matches

class DynamicSkipIterable(IterableDataset):
    def __init__(self, base_dataset, num_samples):
        """
        base_dataset:   一个实现了 __len__ / __getitem__ 的 Dataset，
                        但 __getitem__(i) 可能返回 None 表示“跳过”。
        num_samples:    本 epoch 中需要产出的有效样本数（跳过 None 不计入）。
        """
        super().__init__()
        self.base = base_dataset
        self.N = len(base_dataset)
        self.num_samples = num_samples

    def __iter__(self):
        count = 0
        while count < self.num_samples:
            idx  = torch.randint(0, self.N, (1,)).item()
            data = self.base[idx]
            if data is None:
                continue
            yield data
            count += 1

class Omni6DDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size, num_workers, train_steps_per_epoch, val_steps_per_epoch):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_steps_per_epoch= train_steps_per_epoch
        self.val_steps_per_epoch= val_steps_per_epoch

    def train_dataloader(self):
        dyn_ds = DynamicSkipIterable(
            self.train_dataset,
            num_samples=self.train_steps_per_epoch * self.batch_size
        )
        return DataLoader(
            dyn_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,   # 保证每个 epoch 都整除 batch_size
        )
    def val_dataloader(self):
        dyn_ds = DynamicSkipIterable(
            self.val_dataset,
            num_samples=self.val_steps_per_epoch * self.batch_size
        )
        return DataLoader(
            dyn_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=True,  
        )

class Omni6DDataset(BaseDataset):
    def __init__(self, 
            common_conf,
            data_root,
            model_meta_root,
            scene_name=None,
            **kwargs):
        
        super().__init__(common_conf=common_conf)

        self.img_size = common_conf.img_size
        self.batch_size = common_conf.batch_size
        self.training = common_conf.training
        self.to_tensor = TF.ToTensor() 
        self.depth_scale = 1e3 
        self.view_outplane_thr = common_conf.view_outplane_thr # degree, threshold to distinguish nearby and distant views
        self.view_inplane_thr = common_conf.view_inplane_thr # degree, threshold to distinguish in-plane rotation difference
        self.crop_rel_pad = common_conf.crop_rel_pad
        self.num_views = common_conf.num_views
        self.n_sample_point = common_conf.n_sample_point

        self.data_root = data_root
        if scene_name is None:
            self.scene_name = ['ikea', 'matterport3d', 'scannet++']
        else:
            self.scene_name = [scene_name]

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

        model_meta_path = os.path.join(model_meta_root, 'obj_meta.json')
        with open(model_meta_path, 'r') as f:
            self.model_meta = json.load(f)

        self.color_tpath = "{data_root}/{scene_patch}/{scene_name}/{scene_id:04d}/{frame_id:04d}_color.png"
        self.depth_tpath = "{data_root}/{scene_patch}/{scene_name}/{scene_id:04d}/{frame_id:04d}_depth_1.exr"
        self.mask_tpath = "{data_root}/{scene_patch}/{scene_name}/{scene_id:04d}/{frame_id:04d}_mask.exr"
        self.meta_tpath = "{data_root}/{scene_patch}/{scene_name}/{scene_id:04d}/{frame_id:04d}_meta.json"

        self.total_data = []
        self.annotation = defaultdict(list)
        self.obj_ids = []
        data_patches = sorted(os.listdir(os.path.join(data_root)))
        for data_patch in data_patches:
            data_patch_dir = os.path.join(self.data_root, data_patch)
            for scene_name in self.scene_name:
                scene_ids = sorted(os.listdir(os.path.join(data_patch_dir, scene_name)))
                for scene_id in scene_ids:
                    scene_dir = os.path.join(data_patch_dir, scene_name, scene_id)
                    frames = sorted([f for f in os.listdir(scene_dir) if f.endswith('_color.png')])
                    for frame in frames:
                        frame_id = frame.split('_')[0]  # Assuming the frame name is like '0001_color.png'
                        paths = self.get_frame_paths(data_patch, scene_name, scene_id, frame_id)
                        if paths is None:
                            continue # skip frames with missing files
                        with open(paths["meta"], 'r') as f:
                            meta_data = json.load(f)
                        for obj_key in meta_data['objects']:
                            obj_meta_data = meta_data['objects'][obj_key]

                            # Skip if the object is transparent
                            if 'transparent' in obj_meta_data['material']:
                                continue

                            obj_id = obj_meta_data['meta']['oid']
                            inst_id = int(obj_key.split('_', 1)[0])
                            obj_label = obj_meta_data['meta']['class_name']
                            obj_label_id = obj_meta_data['meta']['class_label']

                            extri_quaternion = obj_meta_data['quaternion_wxyz']
                            rotation = Rotation.from_quat([extri_quaternion[1], 
                                                    extri_quaternion[2], 
                                                    extri_quaternion[3], 
                                                    extri_quaternion[0]])
                            extri_R = rotation.as_matrix()
                            extri_T = np.array(obj_meta_data['translation'])
                            intri = meta_data['camera']['intrinsics']

                            data_idx = len(self.total_data)
                            self.total_data.append({
                                "data_idx": data_idx,
                                "image_path": paths["color"],
                                "depth_path": paths["depth"],
                                "mask_path": paths["mask"],
                                "extri_R": extri_R,
                                "extri_T": extri_T,
                                "intri": intri,
                                "inst_id": inst_id,
                                "obj_name": obj_id,
                                "obj_label": obj_label,
                                "obj_label_id": obj_label_id,
                                })
                            self.annotation[obj_id].append({
                                "data_idx": data_idx,
                                "extri_R": extri_R,
                            })
                            # Add the object ID to the list if not already present
                            if obj_id not in self.obj_ids:
                                self.obj_ids.append(obj_id)

        self.len_train = len(self.total_data)     
        self.indices = self.create_index_list(self.len_train, self.batch_size)              
        status = "Training" if self.training else "Test"
        logging.info(f"{status}: Omni6D Data size: {self.len_train}")
        logging.info(f"{status}: Omni6D Data length of training set: {len(self)}")
    
    def get_frame_paths(self, scene_patch: str, scene_name: str, scene_id: int, frame_id: int) -> dict:
        """
        Returns a dict with file paths for color, depth, mask, and meta
        for the given scene_patch, scene_id, and frame_id.
        """
        params = {
            "data_root":   self.data_root,
            "scene_patch": scene_patch,
            "scene_name":  scene_name,
            "scene_id":    int(scene_id),
            "frame_id":    int(frame_id),
        }

        paths = {
            "color": self.color_tpath.format(**params),
            "depth": self.depth_tpath.format(**params),
            "mask":  self.mask_tpath.format(**params),
            "meta":  self.meta_tpath.format(**params),
        }

        # check that every file exists
        if not all(os.path.exists(p) for p in paths.values()):
            return None

        return paths
    
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
        avg_scale = np.clip(avg_scale, a_min=eps, a_max=5e5)

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
        image_path = first_frame_data['image_path']
        depth_path = first_frame_data['depth_path']
        mask_path = first_frame_data['mask_path']
        instance_id = first_frame_data['inst_id']
        instance_name = first_frame_data['obj_name']

        # load rgb / depth / mask
        image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB (0-255) 
        depth_map = imread(depth_path)
        mask = imread(mask_path)[:, :, 0] * 255
        
        # get the mask based on instance_id
        instance_mask = np.equal(mask, instance_id)
        mask = np.logical_and(instance_mask, depth_map > 0)
        if np.sum(instance_mask) < mask.shape[0] * mask.shape[1] * 0.001:
            return None
        
        # get the reference views
        reference_cands = self.annotation[instance_name]
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
            filepath = anno['image_path']
            image_path = anno['image_path']
            depth_path = anno['depth_path']
            mask_path = anno['mask_path']

            obj_label = anno['obj_label']
            obj_label_id = anno['obj_label_id']
            instance_name = anno['obj_name']
            instance_id = anno['inst_id']

            # load rgb / depth / mask
            image = cv2.imread(image_path)[:, :, ::-1]  # BGR to RGB (0-255) 
            depth_map = imread(depth_path)
            mask = imread(mask_path)[:, :, 0] * 255
            
            # get the mask based on instance_id
            instance_mask = np.equal(mask, instance_id)
            mask = np.logical_and(instance_mask, depth_map > 0)

            if np.sum(mask) < mask.shape[0] * mask.shape[1] * 0.001:
                return None

            # apply mask on image and depth
            depth_map[~mask] = 0
            depth_map *= self.depth_scale
            if self.training and self.image_aug is not None:
                # Convert boolean mask → uint8 (0 or 255)
                mask_u8 = (mask > 0).astype(np.uint8) * 255
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                mask_dilated_u8 = cv2.dilate(mask_u8, kernel, iterations=3)
                mask_dilated = mask_dilated_u8 > 0
                # apply
                image[~mask_dilated] = 0
            else:
                # Apply the mask to image
                image[mask] = 0

            # load the camera parameters
            intrinsic = anno['intri']
            translation = anno['extri_T']
            rotation = anno['extri_R']
            extri = np.eye(4, dtype=np.float32)
            extri[:3, :3] = rotation
            extri[:3, 3] = translation * self.depth_scale

            # Update the intrinsics matrix
            img_resize_scale = image.shape[0] / intrinsic['height']
            assert image.shape[1] / intrinsic['width'] == img_resize_scale
            cam_fx, cam_fy, cam_cx, cam_cy = intrinsic['fx'], intrinsic['fy'], intrinsic['cx'], intrinsic['cy']
            cam_K = np.array([[cam_fx, 0, cam_cx],
                            [0, cam_fy, cam_cy],
                            [0, 0, 1]], dtype=np.float32)
            cam_K *= img_resize_scale
            cam_K[2, 2] = 1

            # Centeral Crop the image
            # ----------------------------------
            orig_mask_modal = mask.astype(np.uint8)
            orig_image_np_hwc = image.copy()
            orig_depth_map = depth_map.copy()
            orig_camera_c2w = CameraModel(
                width=image.shape[1],
                height=image.shape[0],
                f=(cam_K[0, 0], cam_K[1, 1]),
                c=(cam_K[0, 2], cam_K[1, 2]),
                T_world_from_eye=np.linalg.inv(extri))
            
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
            if idx == 0:
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
        # # ---------------------
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
    np.random.seed(42)
    random.seed(42)
    data_root = '/media/tum/data1/Omni6DPose/data/Omni6DPose/SOPE'
    model_meta_root = '/media/tum/data1/Omni6DPose/data/Omni6DPose/Meta'
    # scene_name = 'ikea' #{'ikea', 'matterport3d', 'scannet++'}

    start = time.time()
    # Initialize the dataset
    dataset = Omni6DDataset(
        common_conf=config.common_config,
        data_root=data_root,
        model_meta_root=model_meta_root,
        # scene_name=scene_name,
    )   
    print("Data loading time:", time.time() - start)

    start = time.time()
    for idx in range(20):
        batch = dataset.get_data(seq_index=idx, img_per_seq=8, aspect_ratio=1.0)
    print("Data fetching time:", time.time() - start)