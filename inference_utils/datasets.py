import os
import os.path as osp
import json
import numpy as np
import json
import cv2
from torch.utils.data import Dataset
from omegaconf.dictconfig import DictConfig
import pycocotools.mask as cocomask
from inference_utils.utils import center_crop, crop_anchor
from bop_toolkit_lib.misc import get_symmetry_transformations
from typing import Tuple
from plyfile import PlyData
import trimesh

class BOP_Dataset(Dataset):
    def __init__(self, args : DictConfig):

        self.dataset_base = args.dataset.root
        self.dataset_name = args.dataset.name
        self.output_base = args.output_base
        os.makedirs(self.output_base, exist_ok=True)

        self.obj_ids = args.dataset.obj_ids
        self.obj_models, self.obj_diams, self.obj_symms = self.get_obj_data(self.dataset_base)
        self.target_size = args.dataset.target_size # (518, 518)
        self.depth_scale = args.dataset.depth_scale 

        self.test_dir = osp.join(self.dataset_base, 'test')
        self.test_target_path = osp.join(self.dataset_base, self.dataset_name, 'test_targets_bop19.json')
        with open(self.test_target_path, 'r') as f:
            self.test_target = json.load(f)
        
        self.mask_type = args.mask_type
        if self.mask_type == 'cnos':
            self.cnos_mask_path = osp.join(self.dataset_base, f'cnos_mask/cnos-fastsam_{self.dataset_name}.json')
            with open(self.cnos_mask_path, 'r') as f:
                self.cnos_masks = json.load(f)
        
        # load model for visualization and evaluation
        self.model_dir = osp.join(self.dataset_base, 'models')
        with open(osp.join(self.model_dir, 'models_info.json'), 'r') as f:
            self.model_info = json.load(f)

        # collect the data
        self.test_data = []
        for test_item in self.test_target:
            scene_id = test_item['scene_id']
            image_id = test_item['im_id']
            obj_id = test_item['obj_id']

            obj_name = f'obj_{obj_id:06d}'
            model_path = osp.join(self.model_dir, f'{obj_name}.ply')

            # get the scene camera
            scene_camera_path = osp.join(self.test_dir, f'{scene_id:06d}', 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)
            # get the scene gt (for th gt mask)
            scene_gt_path = osp.join(self.test_dir, f'{scene_id:06d}', f'scene_gt.json')
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)
            # get the scene gt info
            scene_gt_info_path = osp.join(self.test_dir, f'{scene_id:06d}', f'scene_gt_info.json')
            with open(scene_gt_info_path, 'r') as f:
                scene_gt_info = json.load(f)

            if self.dataset_name not in {'ycbv', 'lm', 'lmo'}:
                raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

            reference_id = None
            frame_ids = sorted(scene_gt.keys(), key=int)
            for frame_id in frame_ids:
                test_item_gt = scene_gt[str(frame_id)]
                test_item_gt_info = scene_gt_info[str(frame_id)]
                inst_id = next((idx for idx, item in enumerate(test_item_gt) if item['obj_id'] == obj_id), None)
                if inst_id is None:
                    continue

                visib = test_item_gt_info[inst_id]['visib_fract']
                if visib >= 0.5:
                    reference_id = int(frame_id)
                    break

            if reference_id is None:
                raise ValueError(f"No visible reference frame found for scene {scene_id}, obj {obj_id}")

            anchor_image_path = osp.join(self.test_dir, f'{scene_id:06d}', 'rgb', f'{reference_id:06d}.png')
            anchor_depth_path = osp.join(self.test_dir, f'{scene_id:06d}', 'depth', f'{reference_id:06d}.png')
            anchor_mask_path = osp.join(self.test_dir, f'{scene_id:06d}', 'mask_visib', f'{reference_id:06d}_{inst_id:06d}.png')

            # the anchor pose is used to align the convention
            anchor_pose_gt = test_item_gt[inst_id]
            anchor_extri = np.eye(4)
            anchor_extri[:3,:3] = np.array(anchor_pose_gt["cam_R_m2c"]).reshape(3,3)
            anchor_extri[:3,3] = np.array(anchor_pose_gt["cam_t_m2c"]).reshape(3)
            anchor_intri = np.array(scene_camera[str(reference_id)]['cam_K']).reshape(3,3)

            # get the query image
            test_item_gt = scene_gt[str(image_id)]
            inst_id = [idx for idx, item in enumerate(test_item_gt) if item['obj_id'] == obj_id][0]
            query_image_path = osp.join(self.test_dir, f'{scene_id:06d}', 'rgb', f'{image_id:06d}.png')
            query_depth_path = osp.join(self.test_dir, f'{scene_id:06d}', 'depth', f'{image_id:06d}.png')
            query_intri = np.array(scene_camera[str(image_id)]['cam_K']).reshape(3,3)

            # load query mask
            if self.mask_type == 'cnos':
                det_masks = [item for item in self.cnos_masks if item['scene_id'] == scene_id and item['image_id'] == image_id and item['category_id'] == obj_id]
                # get the mask with highest score
                if len(det_masks) == 0:
                    query_mask = None
                    continue
                query_mask = max(det_masks, key=lambda x: x['score'])
            elif self.mask_type == 'gt':
                query_mask = osp.join(self.test_dir, f'{scene_id:06d}', 'mask_visib', f'{image_id:06d}_{inst_id:06d}.png')
            else:
                raise ValueError(f"Unsupported mask type: {self.mask_type}")

            # load gt pose (for eval)
            query_gt_pose = np.eye(4)
            query_gt_pose[:3, :3] = np.array(test_item_gt[inst_id]["cam_R_m2c"]).reshape(3, 3)
            query_gt_pose[:3, 3] = np.array(test_item_gt[inst_id]["cam_t_m2c"]).reshape(3)

            # save the data item
            self.test_data.append(
                {
                    'scene_id': scene_id,
                    'image_id': image_id,
                    'obj_id': obj_id,
                    'model_path': model_path,
                    'anchor_image_path': anchor_image_path,
                    'anchor_depth_path': anchor_depth_path,
                    'anchor_mask_path': anchor_mask_path,
                    'anchor_extri': anchor_extri, # for vis and eval
                    'anchor_intri': anchor_intri, # for vis and eval
                    'query_image_path': query_image_path,
                    'query_depth_path': query_depth_path,
                    'query_mask': query_mask,
                    'query_intri': query_intri, # for vis and eval
                    'query_gt_pose': query_gt_pose, # for vis and eval
                }
            )

    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        test_data = self.test_data[idx]

        # Load anchor image
        anchor_image_path = test_data['anchor_image_path']
        anchor_depth_path = test_data['anchor_depth_path']
        anchor_mask_path = test_data['anchor_mask_path']

        anchor_image = cv2.imread(anchor_image_path)
        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
        anchor_mask = cv2.imread(anchor_mask_path, cv2.IMREAD_GRAYSCALE) / 255
        anchor_mask = anchor_mask.astype(bool)
        anchor_depth = cv2.imread(anchor_depth_path, cv2.IMREAD_UNCHANGED)

        anchor_mask = anchor_mask & (anchor_depth > 0)
        anchor_image = anchor_image * anchor_mask[..., None]
        anchor_depth = anchor_depth * anchor_mask * self.depth_scale # depth scale

        # Load query image
        query_image_path = test_data['query_image_path']
        query_depth_path = test_data['query_depth_path']
        query_mask = test_data['query_mask']

        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_depth = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)

        if self.mask_type == 'cnos':
            seg = query_mask['segmentation']
            h,w = seg['size']
            rle = cocomask.frPyObjects(seg, h, w)
            query_mask = cocomask.decode(rle)
        elif self.mask_type == 'gt':
            query_mask = cv2.imread(query_mask, cv2.IMREAD_GRAYSCALE) / 255
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
        
        query_mask = query_mask.astype(bool)
        query_mask = query_mask & (query_depth > 0)
        query_image = query_image * query_mask[..., None]
        query_depth = query_depth * query_mask * self.depth_scale # depth scale

        data_batch = {
            'anchor_image': anchor_image,
            'anchor_depth': anchor_depth,
            'anchor_mask': anchor_mask,
            'anchor_extri': test_data['anchor_extri'],
            'anchor_intri': test_data['anchor_intri'],
            'query_image': query_image,
            'query_depth': query_depth,
            'query_mask': query_mask,
            'query_intri': test_data['query_intri'],
            'query_gt_pose': test_data['query_gt_pose'],
            'obj_id': test_data['obj_id'],
            'scene_id': test_data['scene_id'],
            'image_id': test_data['image_id'],
            'model_path': test_data['model_path'],
            'query_image_path': test_data['query_image_path'],
        }

        return data_batch
    
    def get_obj_info(self, obj_id) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of a given object name
        '''
        return self.obj_models[int(obj_id)], self.obj_diams[int(obj_id)], self.obj_symms[int(obj_id)]

    def get_object_info(self) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of all objects
        '''
        return self.obj_models, self.obj_diams, self.obj_symms

    def get_obj_data(self, root: str) -> Tuple[dict,dict,dict]:

        '''
        Returns complete information about a dataset point clouds
        '''

        obj_files = [file for file in os.listdir(osp.join(root, 'models')) if '.ply' in file]
        obj_models, obj_diams, obj_symm = dict(), dict(), dict()

        with open(osp.join(root,'models','models_info.json')) as f:
            models_info = json.load(f)

        for obj_file in obj_files:

            obj_id = int(os.path.splitext(obj_file[4:])[0])
            model_info = models_info[str(obj_id)]
            
            obj_models[obj_id] = self.get_obj_rendering(root, obj_id)
            obj_diams[obj_id] = model_info['diameter']
            obj_symm[obj_id] = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
            
        return (obj_models, obj_diams, obj_symm)
    
    def get_obj_rendering(self, root: str, obj_id: int) -> dict:
        '''
        returns object usable for vispy rendering
        argument obj_model is expected to have the following fields:
        - pts : (N,3) xyz points in mm
        - normals: (N,3) normals
        - faces: (M,3) polygon faces needed for rendering 
        '''

        pcd = PlyData.read(os.path.join(root,'models','obj_{:06d}.ply'.format(obj_id)))
        # these are already in mm    
        xs = pcd['vertex']['x']
        ys = pcd['vertex']['y']
        zs = pcd['vertex']['z']
        nxs = pcd['vertex']['nx']
        nys = pcd['vertex']['ny']
        nzs = pcd['vertex']['nz']

        raw_vertexs = np.asarray(pcd['face']['vertex_indices'])
        faces = np.stack([vert for vert in raw_vertexs],axis=0)

        xyz = np.stack((xs,ys,zs), axis=1)
        normals = np.stack((nxs,nys,nzs), axis=1)
        
        return {
            'pts': xyz,
            'normals': normals,
            'faces': faces
        }
        
class LMO_Dataset(Dataset):
    def __init__(self, args : DictConfig):

        self.dataset_base = args.dataset.root
        self.dataset_name = args.dataset.name
        self.output_base = args.output_base
        os.makedirs(self.output_base, exist_ok=True)

        self.obj_ids = args.dataset.obj_ids
        self.obj_models, self.obj_diams, self.obj_symms = self.get_obj_data(self.dataset_base)
        self.target_size = args.dataset.target_size # (518, 518)
        self.depth_scale = args.dataset.depth_scale 

        self.test_dir = osp.join(self.dataset_base, 'test')
        self.test_target_path = osp.join(self.dataset_base, self.dataset_name, 'test_targets_bop19.json')
        with open(self.test_target_path, 'r') as f:
            self.test_target = json.load(f)

        self.reference_dir = osp.join(self.dataset_base.replace('lmo', 'lm'), 'test')
        # self.reference_ids = [
        #     {"obj_id": 1, "scene_id": 1, "im_id": 693},
        #     {"obj_id": 5, "scene_id": 5, "im_id": 775},
        #     {"obj_id": 6, "scene_id": 6, "im_id": 949},
        #     {"obj_id": 8, "scene_id": 8, "im_id": 994},
        #     {"obj_id": 9, "scene_id": 9, "im_id": 1228},
        #     {"obj_id": 10, "scene_id": 10, "im_id": 289},
        #     {"obj_id": 11, "scene_id": 11, "im_id": 1069},
        #     {"obj_id": 12, "scene_id": 12, "im_id": 647},
        # ]
        self.reference_ids = [
            {"obj_id": 1, "scene_id": 1, "im_id": 693},
            {"obj_id": 5, "scene_id": 5, "im_id": 775},
            {"obj_id": 6, "scene_id": 6, "im_id": 949},
            {"obj_id": 8, "scene_id": 8, "im_id": 994},
            {"obj_id": 9, "scene_id": 9, "im_id": 1036}, #
            {"obj_id": 10, "scene_id": 10, "im_id": 289},
            {"obj_id": 11, "scene_id": 11, "im_id": 1069},
            {"obj_id": 12, "scene_id": 12, "im_id": 1011}, #
        ]
        self.obj_to_scene_frame = {
            ref["obj_id"]: (ref["scene_id"], ref["im_id"])
            for ref in self.reference_ids
        }
        
        self.mask_type = args.mask_type
        if self.mask_type == 'cnos':
            self.cnos_mask_path = osp.join(self.dataset_base, f'cnos_mask/cnos-fastsam_{self.dataset_name}.json')
            with open(self.cnos_mask_path, 'r') as f:
                self.cnos_masks = json.load(f)
        
        # load model for visualization and evaluation
        self.model_dir = osp.join(self.dataset_base, 'models')
        with open(osp.join(self.model_dir, 'models_info.json'), 'r') as f:
            self.model_info = json.load(f)

        # collect the data
        self.test_data = []
        for test_item in self.test_target:
            scene_id = test_item['scene_id']
            image_id = test_item['im_id']
            obj_id = test_item['obj_id']

            if obj_id not in self.obj_ids:
                continue

            obj_name = f'obj_{obj_id:06d}'
            model_path = osp.join(self.model_dir, f'{obj_name}.ply')

            # get the scene camera
            scene_camera_path = osp.join(self.test_dir, f'{scene_id:06d}', 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)
            # get the scene gt (for th gt mask)
            scene_gt_path = osp.join(self.test_dir, f'{scene_id:06d}', f'scene_gt.json')
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)

            # get the anchor image (first image of the scene) - YCBV
            if self.dataset_name == 'lmo':
                ref_scene_id, ref_im_id = self.obj_to_scene_frame[obj_id]
                # get the scene camera
                ref_scene_camera_path = osp.join(self.reference_dir, f'{ref_scene_id:06d}', 'scene_camera.json')
                with open(ref_scene_camera_path, 'r') as f:
                    ref_scene_camera = json.load(f)
                # get the scene gt (for th gt mask)
                ref_scene_gt_path = osp.join(self.reference_dir, f'{ref_scene_id:06d}', f'scene_gt.json')
                with open(ref_scene_gt_path, 'r') as f:
                    ref_scene_gt = json.load(f)

                ref_test_item_gt = ref_scene_gt[str(ref_im_id)]
                inst_id = next(idx for idx, item in enumerate(ref_test_item_gt) if item['obj_id'] == obj_id)
                
                anchor_image_path = osp.join(self.reference_dir, f'{ref_scene_id:06d}', 'rgb', f'{ref_im_id:06d}.png')
                anchor_depth_path = osp.join(self.reference_dir, f'{ref_scene_id:06d}', 'depth', f'{ref_im_id:06d}.png')
                anchor_mask_path = osp.join(self.reference_dir, f'{ref_scene_id:06d}', 'mask_visib', f'{ref_im_id:06d}_{inst_id:06d}.png')

                # the anchor pose is used to align the convention
                anchor_pose_gt = ref_test_item_gt[inst_id]
                anchor_extri = np.eye(4)
                anchor_extri[:3,:3] = np.array(anchor_pose_gt["cam_R_m2c"]).reshape(3,3)
                anchor_extri[:3,3] = np.array(anchor_pose_gt["cam_t_m2c"]).reshape(3)
                anchor_intri = np.array(ref_scene_camera[str(ref_im_id)]['cam_K']).reshape(3,3)
            else:
                raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

            # get the query image
            test_item_gt = scene_gt[str(image_id)]
            inst_id = [idx for idx, item in enumerate(test_item_gt) if item['obj_id'] == obj_id][0]
            query_image_path = osp.join(self.test_dir, f'{scene_id:06d}', 'rgb', f'{image_id:06d}.png')
            query_depth_path = osp.join(self.test_dir, f'{scene_id:06d}', 'depth', f'{image_id:06d}.png')
            query_intri = np.array(scene_camera[str(image_id)]['cam_K']).reshape(3,3)

            # load query mask
            if self.mask_type == 'cnos':
                det_masks = [item for item in self.cnos_masks if item['scene_id'] == scene_id and item['image_id'] == image_id and item['category_id'] == obj_id]
                # get the mask with highest score
                if len(det_masks) == 0:
                    query_mask = None
                    continue
                query_mask = max(det_masks, key=lambda x: x['score'])
            elif self.mask_type == 'gt':
                query_mask = osp.join(self.test_dir, f'{scene_id:06d}', 'mask_visib', f'{image_id:06d}_{inst_id:06d}.png')
            else:
                raise ValueError(f"Unsupported mask type: {self.mask_type}")

            # load gt pose (for eval)
            query_gt_pose = np.eye(4)
            query_gt_pose[:3, :3] = np.array(test_item_gt[inst_id]["cam_R_m2c"]).reshape(3, 3)
            query_gt_pose[:3, 3] = np.array(test_item_gt[inst_id]["cam_t_m2c"]).reshape(3)

            # save the data item
            self.test_data.append(
                {
                    'scene_id': scene_id,
                    'image_id': image_id,
                    'obj_id': obj_id,
                    'model_path': model_path,
                    'anchor_image_path': anchor_image_path,
                    'anchor_depth_path': anchor_depth_path,
                    'anchor_mask_path': anchor_mask_path,
                    'anchor_extri': anchor_extri, # for vis and eval
                    'anchor_intri': anchor_intri, # for vis and eval
                    'query_image_path': query_image_path,
                    'query_depth_path': query_depth_path,
                    'query_mask': query_mask,
                    'query_intri': query_intri, # for vis and eval
                    'query_gt_pose': query_gt_pose, # for vis and eval
                }
            )

    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        test_data = self.test_data[idx]

        # Load anchor image
        anchor_image_path = test_data['anchor_image_path']
        anchor_depth_path = test_data['anchor_depth_path']
        anchor_mask_path = test_data['anchor_mask_path']

        anchor_image = cv2.imread(anchor_image_path)
        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
        anchor_mask = cv2.imread(anchor_mask_path, cv2.IMREAD_GRAYSCALE) / 255
        anchor_mask = anchor_mask.astype(bool)
        anchor_depth = cv2.imread(anchor_depth_path, cv2.IMREAD_UNCHANGED)

        anchor_mask = anchor_mask & (anchor_depth > 0)
        anchor_image = anchor_image * anchor_mask[..., None]
        anchor_depth = anchor_depth * anchor_mask * self.depth_scale # depth scale

        # Load query image
        query_image_path = test_data['query_image_path']
        query_depth_path = test_data['query_depth_path']
        query_mask = test_data['query_mask']

        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_depth = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)

        if self.mask_type == 'cnos':
            seg = query_mask['segmentation']
            h,w = seg['size']
            rle = cocomask.frPyObjects(seg, h, w)
            query_mask = cocomask.decode(rle)
        elif self.mask_type == 'gt':
            query_mask = cv2.imread(query_mask, cv2.IMREAD_GRAYSCALE) / 255
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
        
        query_mask = query_mask.astype(bool)
        query_mask = query_mask & (query_depth > 0)
        query_image = query_image * query_mask[..., None]
        query_depth = query_depth * query_mask * self.depth_scale # depth scale

        data_batch = {
            'anchor_image': anchor_image,
            'anchor_depth': anchor_depth,
            'anchor_mask': anchor_mask,
            'anchor_extri': test_data['anchor_extri'],
            'anchor_intri': test_data['anchor_intri'],
            'query_image': query_image,
            'query_depth': query_depth,
            'query_mask': query_mask,
            'query_intri': test_data['query_intri'],
            'query_gt_pose': test_data['query_gt_pose'],
            'obj_id': test_data['obj_id'],
            'scene_id': test_data['scene_id'],
            'image_id': test_data['image_id'],
            'model_path': test_data['model_path'],
            'query_image_path': test_data['query_image_path'],
        }

        return data_batch
    
    def get_obj_info(self, obj_id) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of a given object name
        '''
        return self.obj_models[int(obj_id)], self.obj_diams[int(obj_id)], self.obj_symms[int(obj_id)]

    def get_object_info(self) -> Tuple:
        '''
        Returns the info (object model, object diameter, object symmetry) of all objects
        '''
        return self.obj_models, self.obj_diams, self.obj_symms

    def get_obj_data(self, root: str) -> Tuple[dict,dict,dict]:

        '''
        Returns complete information about a dataset point clouds
        '''

        obj_files = [file for file in os.listdir(osp.join(root, 'models')) if '.ply' in file]
        obj_models, obj_diams, obj_symm = dict(), dict(), dict()

        with open(osp.join(root,'models','models_info.json')) as f:
            models_info = json.load(f)

        for obj_file in obj_files:

            obj_id = int(os.path.splitext(obj_file[4:])[0])
            model_info = models_info[str(obj_id)]
            
            obj_models[obj_id] = self.get_obj_rendering(root, obj_id)
            obj_diams[obj_id] = model_info['diameter']
            obj_symm[obj_id] = get_symmetry_transformations(model_info, max_sym_disc_step=0.05)
            
        return (obj_models, obj_diams, obj_symm)
    
    def get_obj_rendering(self, root: str, obj_id: int) -> dict:
        '''
        returns object usable for vispy rendering
        argument obj_model is expected to have the following fields:
        - pts : (N,3) xyz points in mm
        - normals: (N,3) normals
        - faces: (M,3) polygon faces needed for rendering 
        '''

        pcd = PlyData.read(os.path.join(root,'models','obj_{:06d}.ply'.format(obj_id)))
        # these are already in mm    
        xs = pcd['vertex']['x']
        ys = pcd['vertex']['y']
        zs = pcd['vertex']['z']
        nxs = pcd['vertex']['nx']
        nys = pcd['vertex']['ny']
        nzs = pcd['vertex']['nz']

        raw_vertexs = np.asarray(pcd['face']['vertex_indices'])
        faces = np.stack([vert for vert in raw_vertexs],axis=0)

        xyz = np.stack((xs,ys,zs), axis=1)
        normals = np.stack((nxs,nys,nzs), axis=1)
        
        return {
            'pts': xyz,
            'normals': normals,
            'faces': faces
        }


class YCBInEOAT_Dataset(Dataset):
    def __init__(self, args : DictConfig):

        self.dataset_base = args.dataset.root
        self.dataset_name = args.dataset.name
        self.output_base = args.output_base
        os.makedirs(self.output_base, exist_ok=True)

        self.obj_ids = args.dataset.obj_ids
        self.target_size = args.dataset.target_size # (518, 518)
        self.depth_scale = args.dataset.depth_scale 
        self.running_stride = 10

        self.query_dir = osp.join(self.dataset_base, self.dataset_name)
        self.video_frames = os.listdir(self.query_dir)
        self.anchor_dir = osp.join(self.dataset_base, 'dexycb_reference_view_ours')
        self.obj_names = os.listdir(self.anchor_dir)
        self.obj_models = self.get_obj_data(self.anchor_dir)
        self.obj_model_base = osp.join(self.dataset_base, 'YCB_models_with_ply', 'CADmodels')

        # collect the data
        self.test_data = []
        for video_frame in self.video_frames:
            video_frame_dir = osp.join(self.query_dir, video_frame)
            object_name = next(
                (obj_name for obj_name in self.obj_names
                if any(word in video_frame for word in obj_name.split('_')[1:])),
                None
            )
            # print(f"Processing video frame: {video_frame}, object name: {object_name}")
            model_path = osp.join(self.obj_model_base, object_name, f'textured.obj')
            obj_id = int(os.path.splitext(os.path.basename(object_name))[0].split('_')[0]) 
            # get the anchor image
            anchor_dir = osp.join(self.anchor_dir, object_name)
            anchor_image_path = osp.join(anchor_dir, 'color.png')
            anchor_depth_path = osp.join(anchor_dir, 'depth.png')
            anchor_mask_path = osp.join(anchor_dir, 'mask.png')
            anchor_intri_path = osp.join(anchor_dir, 'K.txt')
            anchor_pose_gt_path = osp.join(anchor_dir, f'{object_name}_gt_pose.txt')

            # the anchor pose is used to align the convention
            anchor_extri = np.loadtxt(anchor_pose_gt_path).reshape(4,4)
            anchor_extri[:3, 3] *= 1000 # convert to mm
            anchor_intri = np.loadtxt(anchor_intri_path).reshape(3,3)

            # get the query image
            query_rgb_files = os.listdir(osp.join(video_frame_dir, 'rgb'))
            query_pose_files = os.listdir(osp.join(video_frame_dir, 'annotated_poses'))
            query_camera_K = np.loadtxt(osp.join(video_frame_dir, 'cam_K.txt')).reshape(3,3)
            for idx in range(0, len(query_rgb_files), self.running_stride):
                file = query_rgb_files[idx]
                pose = query_pose_files[idx]
                query_image_path = osp.join(video_frame_dir, 'rgb', file)
                query_depth_path = osp.join(video_frame_dir, 'depth', file)
                query_mask_path = osp.join(video_frame_dir, 'gt_mask', file)
                # load gt pose (for eval)
                query_gt_pose = np.loadtxt(osp.join(video_frame_dir, 'annotated_poses', pose)).reshape(4, 4)

                # save the data item
                self.test_data.append(
                    {
                        'scene_id': video_frame,
                        'image_id': file.split('.')[0],
                        'obj_id': obj_id,
                        'model_path': model_path,
                        'anchor_image_path': anchor_image_path,
                        'anchor_depth_path': anchor_depth_path,
                        'anchor_mask_path': anchor_mask_path,
                        'anchor_extri': anchor_extri, # for vis and eval
                        'anchor_intri': anchor_intri, # for vis and eval
                        'query_image_path': query_image_path,
                        'query_depth_path': query_depth_path,
                        'query_mask_path': query_mask_path,
                        'query_intri': query_camera_K, # for vis and eval
                        'query_gt_pose': query_gt_pose, # for vis and eval
                    }
                )

    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        test_data = self.test_data[idx]

        # Load anchor image
        anchor_image_path = test_data['anchor_image_path']
        anchor_depth_path = test_data['anchor_depth_path']
        anchor_mask_path = test_data['anchor_mask_path']

        anchor_image = cv2.imread(anchor_image_path)
        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
        anchor_mask = cv2.imread(anchor_mask_path, cv2.IMREAD_GRAYSCALE) / 255
        anchor_mask = anchor_mask.astype(bool)
        anchor_depth = cv2.imread(anchor_depth_path, cv2.IMREAD_UNCHANGED)

        anchor_mask = anchor_mask & (anchor_depth > 0)
        anchor_image = anchor_image * anchor_mask[..., None]
        anchor_depth = anchor_depth * anchor_mask * self.depth_scale # depth scale

        # Load query image
        query_image_path = test_data['query_image_path']
        query_depth_path = test_data['query_depth_path']
        query_mask_path = test_data['query_mask_path']

        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_depth = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)
        query_mask = cv2.imread(query_mask_path, cv2.IMREAD_GRAYSCALE) / 255
        
        query_mask = query_mask.astype(bool)
        query_mask = query_mask & (query_depth > 0)
        query_image = query_image * query_mask[..., None]
        query_depth = query_depth * query_mask * self.depth_scale # depth scale

        data_batch = {
            'anchor_image': anchor_image,
            'anchor_depth': anchor_depth,
            'anchor_mask': anchor_mask,
            'anchor_extri': test_data['anchor_extri'],
            'anchor_intri': test_data['anchor_intri'],
            'query_image': query_image,
            'query_depth': query_depth,
            'query_mask': query_mask,
            'query_intri': test_data['query_intri'],
            'query_gt_pose': test_data['query_gt_pose'],
            'obj_id': test_data['obj_id'],
            'scene_id': test_data['scene_id'],
            'image_id': test_data['image_id'],
            'model_path': test_data['model_path'],
            'query_image_path': test_data['query_image_path'],
        }

        return data_batch
    
    def get_obj_info(self, obj_id):
        '''
        Returns the info object model of a given object name
        '''
        return self.obj_models[int(obj_id)]

    def get_object_info(self):
        '''
        Returns the info object model of all objects
        '''
        return self.obj_models

    def get_obj_data(self, root: str):

        '''
        Returns complete information about a dataset point clouds
        '''

        obj_dirs = os.listdir(root)
        obj_models= dict()
        obj_files = []
        for obj_dir in obj_dirs:
            obj_dir_path = osp.join(root, obj_dir)
            obj_files.append(osp.join(obj_dir_path, f'mesh_{obj_dir}.obj'))

        for obj_file in obj_files:
            obj_id = int(os.path.splitext(os.path.basename(obj_file))[0].split('_')[1])           
            obj_models[obj_id] = self.get_obj_rendering(obj_file)
            
        return obj_models
        
    def get_obj_rendering(self, obj_file: str) -> dict:
        '''
        Loads a 3D .obj model file and returns data for vispy rendering:
        - pts: (N,3) xyz points in mm
        - normals: (N,3) vertex normals
        - faces: (M,3) triangle indices
        '''

        mesh = trimesh.load(obj_file, force='mesh')

        # Vertices: assumed to be in meters; convert to mm if needed
        xyz = mesh.vertices  # (N, 3)
        normals = mesh.vertex_normals  # (N, 3)
        faces = mesh.faces  # (M, 3)

        # Optional: scale from meters to mm if needed
        # xyz *= 1000.0

        return {
            'pts': xyz,
            'normals': normals,
            'faces': faces
        }


class HOPE_Dataset(Dataset):
    def __init__(self, args : DictConfig):

        self.dataset_base = args.dataset.root
        self.dataset_name = args.dataset.name
        self.output_base = args.output_base
        os.makedirs(self.output_base, exist_ok=True)

        self.target_size = args.dataset.target_size # (518, 518)
        self.depth_scale = args.dataset.depth_scale 

        self.obj_ids = args.dataset.obj_ids
        self.onboarding_base = osp.join(self.dataset_base, args.onboarding)
        self.onboarding_stride = args.onboarding_stride
        self.get_anchor_images()

        self.test_dir = osp.join(self.dataset_base, 'test')
        self.test_target_path = osp.join(self.dataset_base, self.dataset_name, 'test_targets_bop24.json')
        with open(self.test_target_path, 'r') as f:
            self.test_target = json.load(f)
        
        self.mask_type = args.mask_type
        self.mask_score_thr = args.mask_score_thr
        if self.mask_type == 'cnos':
            self.cnos_mask_path = osp.join(self.dataset_base, f'cnos-fastsam_{self.dataset_name}.json')
            with open(self.cnos_mask_path, 'r') as f:
                self.cnos_masks = json.load(f)
        
        # load model for visualization and evaluation
        self.model_dir = osp.join(self.dataset_base, 'models')
        with open(osp.join(self.model_dir, 'models_info.json'), 'r') as f:
            self.model_info = json.load(f)

        # collect the data
        self.test_data = []
        for test_item in self.test_target:
            scene_id = test_item['scene_id']
            image_id = test_item['im_id']

            if scene_id != 1:
                continue

            # get the scene camera
            scene_camera_path = osp.join(self.test_dir, f'{scene_id:06d}', 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            # get the query image
            query_image_path = osp.join(self.test_dir, f'{scene_id:06d}', 'rgb', f'{image_id:06d}.png')
            query_depth_path = osp.join(self.test_dir, f'{scene_id:06d}', 'depth', f'{image_id:06d}.png')
            query_intri = np.array(scene_camera[str(image_id)]['cam_K']).reshape(3,3)

            # load query mask
            if self.mask_type == 'cnos':
                det_masks = [item for item in self.cnos_masks if item['scene_id'] == scene_id and item['image_id'] == image_id]
                if len(det_masks) == 0:
                    continue
                # only keep the masks with high score for inference
                det_masks_sorted = sorted(det_masks, key=lambda x: x['score'], reverse=True)
                det_masks_kept = [mask for mask in det_masks_sorted if mask['score'] >= self.mask_score_thr]
                for idx, query_mask in enumerate(det_masks_kept):
                    obj_id = query_mask['category_id']
                    obj_name = f'obj_{obj_id:06d}'
                    model_path = osp.join(self.model_dir, f'{obj_name}.ply')
                    self.test_data.append(
                        {
                            'scene_id': scene_id,
                            'image_id': image_id,
                            'obj_id': obj_id,
                            'inst_id': idx,
                            'model_path': model_path,
                            'query_image_path': query_image_path,
                            'query_depth_path': query_depth_path,
                            'query_mask': query_mask,
                            'query_intri': query_intri,
                        }
                    )
                
            else:
                raise ValueError(f"Unsupported mask type: {self.mask_type}")

    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        test_data = self.test_data[idx]

        # Load query image
        query_image_path = test_data['query_image_path']
        query_depth_path = test_data['query_depth_path']
        query_mask = test_data['query_mask']

        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
        query_depth = cv2.imread(query_depth_path, cv2.IMREAD_UNCHANGED)

        if self.mask_type == 'cnos':
            seg = query_mask['segmentation']
            mask_score = query_mask['score']
            h,w = seg['size']
            rle = cocomask.frPyObjects(seg, h, w)
            query_mask = cocomask.decode(rle)
        elif self.mask_type == 'gt':
            query_mask = cv2.imread(query_mask, cv2.IMREAD_GRAYSCALE) / 255
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
        
        query_mask = query_mask.astype(bool)
        query_mask = query_mask & (query_depth > 0)
        query_image = query_image * query_mask[..., None]
        query_depth = query_depth * query_mask * self.depth_scale # depth scale

        data_batch = {
            'query_image': query_image,
            'query_depth': query_depth,
            'query_mask': query_mask,
            'query_intri': test_data['query_intri'],
            'obj_id': test_data['obj_id'],
            'scene_id': test_data['scene_id'],
            'image_id': test_data['image_id'],
            'inst_id': test_data['inst_id'],
            'mask_score': mask_score,
            'model_path': test_data['model_path'],
            'query_image_path': test_data['query_image_path'],
        }

        return data_batch
    
    def get_anchor_images(self):
        self.anchor_info = {} 
        for obj_id in self.obj_ids:
            obj_onboarding_dir = osp.join(self.onboarding_base, f'obj_{obj_id:06d}')
            # check the total frames in onboarding directory
            if not osp.exists(obj_onboarding_dir):
                print(f"Onboarding directory for object {obj_id} does not exist: {obj_onboarding_dir}")
                continue
            obj_onboarding_frames = os.listdir(osp.join(obj_onboarding_dir, 'rgb'))
            if len(obj_onboarding_frames) == 0:
                print(f"No frames found in onboarding directory for object {obj_id}: {obj_onboarding_dir}")
                continue

            # get the scene camera and scene gt
            scene_camera_path = osp.join(obj_onboarding_dir, 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            scene_gt_path = osp.join(obj_onboarding_dir, 'scene_gt.json')
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)

            anchor_images = []
            anchor_depths = []
            anchor_masks = []
            for idx in range(0, len(obj_onboarding_frames), self.onboarding_stride):
                # get the anchor image 
                anchor_image_path = osp.join(obj_onboarding_dir, 'rgb', f'{idx:06d}.jpg')
                anchor_depth_path = osp.join(obj_onboarding_dir, 'depth', f'{idx:06d}.png')
                anchor_mask_path = osp.join(obj_onboarding_dir, 'mask_visib', f'{idx:06d}_000000.png')

                # Load anchor image
                anchor_image = cv2.imread(anchor_image_path)
                anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
                anchor_mask = cv2.imread(anchor_mask_path, cv2.IMREAD_GRAYSCALE) / 255
                anchor_mask = anchor_mask.astype(bool)
                anchor_depth = cv2.imread(anchor_depth_path, cv2.IMREAD_UNCHANGED)

                anchor_mask = anchor_mask & (anchor_depth > 0)
                anchor_image = anchor_image * anchor_mask[..., None]
                anchor_depth = anchor_depth * anchor_mask 

                # the anchor pose is used to align the convention
                if idx == 0:
                    anchor_pose_gt = scene_gt[str(idx)][0]
                    anchor_extri = np.eye(4)
                    anchor_extri[:3,:3] = np.array(anchor_pose_gt["cam_R_m2c"]).reshape(3,3)
                    anchor_extri[:3,3] = np.array(anchor_pose_gt["cam_t_m2c"]).reshape(3)
                    anchor_intri = np.array(scene_camera[str(idx)]['cam_K']).reshape(3,3)

                    # process the anchor image and mask
                    anchor_image, anchor_mask, anchor_depth, anchor_intri, anchor_extri = center_crop(anchor_image, 
                                                                                                    anchor_mask, 
                                                                                                    anchor_depth, 
                                                                                                    anchor_intri, 
                                                                                                    anchor_extri, 
                                                                                                    target_image_shape=self.target_size)
                else:
                    anchor_image, anchor_mask, anchor_depth = crop_anchor(anchor_image, 
                                                                        anchor_mask, 
                                                                        anchor_depth,
                                                                        target_size=self.target_size)

                anchor_images.append(anchor_image)
                anchor_depths.append(anchor_depth)
                anchor_masks.append(anchor_mask)

                # break # only one reference view
            
            anchor_images = np.stack(anchor_images, axis=0)
            anchor_depths = np.stack(anchor_depths, axis=0)
            anchor_masks = np.stack(anchor_masks, axis=0)

            self.anchor_info[obj_id] = {
                "anchor_images": anchor_images,
                "anchor_depths": anchor_depths,
                "anchor_masks": anchor_masks,
                "extrinsics": anchor_extri,
                "intrinsics": anchor_intri,
            }

    def get_object_anchor_info(self):
        '''
        Returns the anchor images, depths, masks, extrinsics and intrinsics for a given object id
        '''
        return self.anchor_info
    

class HANDAL_Dataset(Dataset):
    def __init__(self, args : DictConfig):

        self.dataset_base = args.dataset.root
        self.dataset_name = args.dataset.name
        self.output_base = args.output_base
        os.makedirs(self.output_base, exist_ok=True)

        self.target_size = args.dataset.target_size # (518, 518)
        self.depth_scale = args.dataset.depth_scale 

        self.obj_ids = args.dataset.obj_ids
        self.onboarding_base = osp.join(self.dataset_base, args.onboarding)
        self.onboarding_stride = args.onboarding_stride
        self.get_anchor_images()

        self.test_dir = osp.join(self.dataset_base, 'test')
        self.test_target_path = osp.join(self.dataset_base, self.dataset_name, 'test_targets_bop24.json')
        with open(self.test_target_path, 'r') as f:
            self.test_target = json.load(f)
        
        self.mask_type = args.mask_type
        self.mask_score_thr = args.mask_score_thr
        if self.mask_type == 'cnos':
            self.cnos_mask_path = osp.join(self.dataset_base, f'cnos-fastsam_{self.dataset_name}.json')
            with open(self.cnos_mask_path, 'r') as f:
                self.cnos_masks = json.load(f)
        
        # load model for visualization and evaluation
        self.model_dir = osp.join(self.dataset_base, 'models_eval')
        with open(osp.join(self.model_dir, 'models_info.json'), 'r') as f:
            self.model_info = json.load(f)

        # collect the data
        self.test_data = []
        for test_item in self.test_target:
            scene_id = test_item['scene_id']
            image_id = test_item['im_id']

            # get the scene camera
            scene_camera_path = osp.join(self.test_dir, f'{scene_id:06d}', 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            # get the query image
            query_image_path = osp.join(self.test_dir, f'{scene_id:06d}', 'rgb', f'{image_id:06d}.jpg')
            query_intri = np.array(scene_camera[str(image_id)]['cam_K']).reshape(3,3)

            # load query mask
            if self.mask_type == 'cnos':
                det_masks = [item for item in self.cnos_masks if item['scene_id'] == scene_id and item['image_id'] == image_id]
                if len(det_masks) == 0:
                    continue
                # only keep the masks with high score for inference
                det_masks_sorted = sorted(det_masks, key=lambda x: x['score'], reverse=True)
                det_masks_kept = [mask for mask in det_masks_sorted if mask['score'] >= self.mask_score_thr]
                for idx, query_mask in enumerate(det_masks_kept):
                    obj_id = query_mask['category_id']
                    obj_name = f'obj_{obj_id:06d}'
                    model_path = osp.join(self.model_dir, f'{obj_name}.ply')
                    self.test_data.append(
                        {
                            'scene_id': scene_id,
                            'image_id': image_id,
                            'obj_id': obj_id,
                            'inst_id': idx,
                            'model_path': model_path,
                            'query_image_path': query_image_path,
                            'query_mask': query_mask,
                            'query_intri': query_intri,
                        }
                    )
                
            else:
                raise ValueError(f"Unsupported mask type: {self.mask_type}")

    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self, idx):
        test_data = self.test_data[idx]

        # Load query image
        query_image_path = test_data['query_image_path']
        query_mask = test_data['query_mask']

        query_image = cv2.imread(query_image_path)
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

        if self.mask_type == 'cnos':
            seg = query_mask['segmentation']
            mask_score = query_mask['score']
            h,w = seg['size']
            rle = cocomask.frPyObjects(seg, h, w)
            query_mask = cocomask.decode(rle)
        elif self.mask_type == 'gt':
            query_mask = cv2.imread(query_mask, cv2.IMREAD_GRAYSCALE) / 255
        else:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
        
        query_mask = query_mask.astype(bool)
        query_image = query_image * query_mask[..., None]

        data_batch = {
            'query_image': query_image,
            'query_mask': query_mask,
            'query_intri': test_data['query_intri'],
            'obj_id': test_data['obj_id'],
            'scene_id': test_data['scene_id'],
            'image_id': test_data['image_id'],
            'inst_id': test_data['inst_id'],
            'mask_score': mask_score,
            'model_path': test_data['model_path'],
            'query_image_path': test_data['query_image_path'],
        }

        return data_batch
    
    def get_anchor_images(self):
        self.anchor_info = {} 
        for obj_id in self.obj_ids:
            obj_onboarding_dir = osp.join(self.onboarding_base, f'obj_{obj_id:06d}')
            # check the total frames in onboarding directory
            if not osp.exists(obj_onboarding_dir):
                print(f"Onboarding directory for object {obj_id} does not exist: {obj_onboarding_dir}")
                continue
            obj_onboarding_frames = os.listdir(osp.join(obj_onboarding_dir, 'rgb'))
            if len(obj_onboarding_frames) == 0:
                print(f"No frames found in onboarding directory for object {obj_id}: {obj_onboarding_dir}")
                continue

            # get the scene camera and scene gt
            scene_camera_path = osp.join(obj_onboarding_dir, 'scene_camera.json')
            with open(scene_camera_path, 'r') as f:
                scene_camera = json.load(f)

            scene_gt_path = osp.join(obj_onboarding_dir, 'scene_gt.json')
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)

            anchor_images = []
            anchor_depths = []
            anchor_masks = []
            for idx in range(0, len(obj_onboarding_frames), self.onboarding_stride):
                # get the anchor image 
                anchor_image_path = osp.join(obj_onboarding_dir, 'rgb', f'{idx:06d}.png')
                anchor_depth_path = osp.join(obj_onboarding_dir, 'depth', f'{idx:06d}.png')
                anchor_mask_path = osp.join(obj_onboarding_dir, 'mask_visib', f'{idx:06d}_000000.png')

                # Load anchor image
                anchor_image = cv2.imread(anchor_image_path)
                anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
                anchor_mask = cv2.imread(anchor_mask_path, cv2.IMREAD_GRAYSCALE) / 255
                anchor_mask = anchor_mask.astype(bool)
                anchor_depth = cv2.imread(anchor_depth_path, cv2.IMREAD_UNCHANGED)

                anchor_mask = anchor_mask & (anchor_depth > 0)
                anchor_image = anchor_image * anchor_mask[..., None]
                anchor_depth = anchor_depth * anchor_mask 

                # the anchor pose is used to align the convention
                if idx == 0:
                    anchor_pose_gt = scene_gt[str(idx)][0]
                    anchor_extri = np.eye(4)
                    anchor_extri[:3,:3] = np.array(anchor_pose_gt["cam_R_m2c"]).reshape(3,3)
                    anchor_extri[:3,3] = np.array(anchor_pose_gt["cam_t_m2c"]).reshape(3)
                    anchor_intri = np.array(scene_camera[str(idx)]['cam_K']).reshape(3,3)

                    # process the anchor image and mask
                    anchor_image, anchor_mask, anchor_depth, anchor_intri, anchor_extri = center_crop(anchor_image, 
                                                                                                    anchor_mask, 
                                                                                                    anchor_depth, 
                                                                                                    anchor_intri, 
                                                                                                    anchor_extri, 
                                                                                                    target_image_shape=self.target_size)
                else:
                    anchor_image, anchor_mask, anchor_depth = crop_anchor(anchor_image, 
                                                                        anchor_mask, 
                                                                        anchor_depth,
                                                                        target_size=self.target_size)
                
                # # shrink mask to avoid the interpolation error
                # kernel = np.ones((3, 3), np.uint8)                  
                # anchor_mask = cv2.erode(anchor_mask.astype(np.uint8), kernel, iterations=3).astype(bool)
                # anchor_depth = anchor_depth * anchor_mask

                anchor_images.append(anchor_image)
                anchor_depths.append(anchor_depth)
                anchor_masks.append(anchor_mask)

                break # only one reference view
            
            anchor_images = np.stack(anchor_images, axis=0)
            anchor_depths = np.stack(anchor_depths, axis=0)
            anchor_masks = np.stack(anchor_masks, axis=0)

            self.anchor_info[obj_id] = {
                "anchor_images": anchor_images,
                "anchor_depths": anchor_depths,
                "anchor_masks": anchor_masks,
                "extrinsics": anchor_extri,
                "intrinsics": anchor_intri,
            }

    def get_object_anchor_info(self):
        '''
        Returns the anchor images, depths, masks, extrinsics and intrinsics for a given object id
        '''
        return self.anchor_info
        
