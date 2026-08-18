import torch
import numpy as np
import cv2
import math
from torchvision import transforms as TF
from PIL import Image
from training.data.datasets_utils import misc
from training.data.datasets_utils.structs import CameraModel

to_tensor = TF.ToTensor()

def load_image(image_path, target_size=(518, 518)):
    """
    Load an image from a file path.
    """

    image = Image.open(image_path)
    image = image.convert("RGB")  

    # Resize with new dimensions (width, height)
    image = image.resize(target_size, Image.Resampling.BICUBIC)
    image = to_tensor(image)

    return image

def expand_box(x_min, x_max, y_min, y_max, scale, img_w, img_h):
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    half_w = bbox_w * scale / 2
    half_h = bbox_h * scale / 2

    x1 = int(max(center_x - half_w, 0))
    x2 = int(min(center_x + half_w, img_w))
    y1 = int(max(center_y - half_h, 0))
    y2 = int(min(center_y + half_h, img_h))

    return x1, y1, x2, y2


def center_crop(image, mask, depth_map, intri, extri, target_image_shape=(518, 518), crop_rel_pad=0):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 3:  # [3, H, W] → [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    if isinstance(intri, torch.Tensor):
        intri = intri.detach().cpu().numpy()
    if isinstance(extri, torch.Tensor):
        extri = extri.detach().cpu().numpy()

    orig_mask_modal = mask.astype(np.uint8)
    orig_image_np_hwc = image.copy()
    orig_depth_map = depth_map.copy()

    # T_world_from_eye must be a rigid transform
    T_world_from_eye = np.linalg.inv(extri)
    R = T_world_from_eye[:3, :3]
    U, _, Vt = np.linalg.svd(R)
    R_fixed = U @ Vt
    T_world_from_eye[:3, :3] = R_fixed
    T_world_from_eye[3] = [0, 0, 0, 1]

    orig_camera_c2w = CameraModel(
        width=image.shape[1],
        height=image.shape[0],
        f=(intri[0, 0], intri[1, 1]),
        c=(intri[0, 2], intri[1, 2]),
        T_world_from_eye=T_world_from_eye)
    
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
        viewport_rel_pad=crop_rel_pad,
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
    
    return image, mask, depth_map, intri, extri


def crop(cam_K, image_path, mask_path, target_size, padding_scale=1.5, depth_path=None):
    """
    Center‐crop based on mask bbox, then resize with uniform scale + pad to target_size,
    and adjust intrinsics so there’s no non‐uniform distortion.

    Args:
        cam_K (np.ndarray): 3x3 intrinsic matrix
        image_path (str): path to RGB image
        mask_path (str): path to grayscale mask image
        target_size (tuple): (width, height)
        padding_scale (float): factor to expand bbox before cropping

    Returns:
        resized_tensor (torch.Tensor): normalized, padded tensor image
        padded_mask (np.ndarray): binary mask at target_size
        K_new (np.ndarray): adjusted intrinsic matrix
    """
    # --- load ---
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if depth_path is not None:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # --- find bbox + padded box coords ---
    ys, xs = np.where(mask>0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # expand box
    w_box = x_max - x_min
    h_box = y_max - y_min
    cx, cy = x_min + w_box/2, y_min + h_box/2
    half_w = w_box/2 * padding_scale
    half_h = h_box/2 * padding_scale

    x1 = int(max(0, cx - half_w))
    x2 = int(min(image.shape[1], cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(image.shape[0], cy + half_h))

    # --- crop ---
    cropped_img  = image[y1:y2, x1:x2]
    cropped_mask = mask [y1:y2, x1:x2]
    if depth_path is not None:
        cropped_depth = depth_map[y1:y2, x1:x2]

    # --- compute uniform scale + resized dims ---
    tgt_w, tgt_h = target_size
    crop_w, crop_h = x2 - x1, y2 - y1
    scale = min(tgt_w / crop_w, tgt_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # resize
    img_resized  = cv2.resize(cropped_img,  (new_w, new_h))
    mask_resized= cv2.resize(cropped_mask,(new_w, new_h))
    if depth_path is not None:
        depth_resized = cv2.resize(cropped_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- pad to target_size (centered) ---
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    mask_padded = cv2.copyMakeBorder(
        mask_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )
    if depth_path is not None:
        depth_padded = cv2.copyMakeBorder(
            depth_resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

    # mask the img
    img_padded = img_padded * (mask_padded[:, :, None] > 0)
    if depth_path is not None:
        depth_padded = depth_padded * (mask_padded > 0)

    # --- adjust intrinsics ---
    K_new = cam_K.copy()
    # uniform scale on focal lengths
    K_new[0,0] *= scale
    K_new[1,1] *= scale
    # principal point: original cx,cy → shifted by crop and pad
    cx_new = (cam_K[0,2] - x1) * scale + pad_left
    cy_new = (cam_K[1,2] - y1) * scale + pad_top
    K_new[0,2] = cx_new
    K_new[1,2] = cy_new

    # --- to tensor ---
    pil = Image.fromarray(img_padded)
    tensor = to_tensor(pil)

    if depth_path is None:
        depth_padded = None

    return tensor, mask_padded, K_new, depth_padded

def crop_anchor(image, mask, depth_map, target_size, padding_scale=1.5):
    """
    Same as previous crop, but accepts tensors or numpy arrays directly.

    Args:
        image (np.ndarray or torch.Tensor): RGB image [H,W,3] or [3,H,W]
        mask (np.ndarray or torch.Tensor): grayscale mask [H,W]
        target_size (tuple): (width, height)
        padding_scale (float): bbox padding
        depth_map: [H,W] depth as np.ndarray or torch.Tensor

    Returns:
        image_tensor: torch.Tensor [3, H, W]
        padded_mask: np.ndarray [H, W]
        depth_padded: np.ndarray [H, W]
    """
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 3:  # [3, H, W] → [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()

    image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    mask = mask.astype(np.uint8)

    # --- find bbox + padded box coords ---
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w_box = x_max - x_min
    h_box = y_max - y_min
    cx, cy = x_min + w_box / 2, y_min + h_box / 2
    half_w = w_box / 2 * padding_scale
    half_h = h_box / 2 * padding_scale

    x1 = int(max(0, cx - half_w))
    x2 = int(min(image.shape[1], cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(image.shape[0], cy + half_h))

    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    cropped_depth = depth_map[y1:y2, x1:x2]

    # --- resize ---
    tgt_w, tgt_h = target_size
    crop_w, crop_h = x2 - x1, y2 - y1
    scale = min(tgt_w / crop_w, tgt_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    img_resized = cv2.resize(cropped_img, (new_w, new_h))
    mask_resized = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    depth_resized = cv2.resize(cropped_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- pad ---
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    depth_padded = cv2.copyMakeBorder(depth_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    # --- apply mask ---
    img_padded = img_padded * (mask_padded[:, :, None] > 0)
    depth_padded = depth_padded * (mask_padded > 0)

    return img_padded, mask_padded, depth_padded


def crop_input(image, mask, depth_map, cam_K, target_size, padding_scale=1.5):
    """
    Crops an image and its mask/depth map based on the mask region, with safe handling for empty or degenerate masks.
    """

    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 3:  # [3, H, W] → [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    if isinstance(cam_K, torch.Tensor):
        cam_K = cam_K.detach().cpu().numpy()

    image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    mask = mask.astype(np.uint8)

    H, W = image.shape[:2]

    valid_flag = True

    # --- find bbox ---
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        valid_flag = False
        # Default: central crop of target aspect
        tgt_w, tgt_h = target_size
        cx, cy = W / 2, H / 2
        aspect = tgt_w / tgt_h
        if W / H > aspect:
            crop_h = H / padding_scale
            crop_w = crop_h * aspect
        else:
            crop_w = W / padding_scale
            crop_h = crop_w / aspect
        x1 = int(max(0, cx - crop_w / 2))
        x2 = int(min(W, cx + crop_w / 2))
        y1 = int(max(0, cy - crop_h / 2))
        y2 = int(min(H, cy + crop_h / 2))
    else:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        w_box = max(1, x_max - x_min)  # avoid 0 width
        h_box = max(1, y_max - y_min)  # avoid 0 height
        cx, cy = x_min + w_box / 2, y_min + h_box / 2
        half_w = w_box / 2 * padding_scale
        half_h = h_box / 2 * padding_scale

        x1 = int(max(0, cx - half_w))
        x2 = int(min(W, cx + half_w))
        y1 = int(max(0, cy - half_h))
        y2 = int(min(H, cy + half_h))

    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    cropped_depth = depth_map[y1:y2, x1:x2] if depth_map is not None else None  

    # --- resize ---
    tgt_w, tgt_h = target_size
    scale = min(tgt_w / crop_w, tgt_h / crop_h)
    new_w = max(1, int(crop_w * scale))
    new_h = max(1, int(crop_h * scale))

    img_resized = cv2.resize(cropped_img, (new_w, new_h))
    mask_resized = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    depth_resized = None
    if cropped_depth is not None:
        depth_resized = cv2.resize(cropped_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- pad ---
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    depth_padded = None
    if depth_resized is not None:
        depth_padded = cv2.copyMakeBorder(depth_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    # --- apply mask ---
    img_padded = img_padded * (mask_padded[:, :, None] > 0)
    if depth_padded is not None:
        depth_padded = depth_padded * (mask_padded > 0)

    # --- adjust intrinsics ---
    K_new = cam_K.copy()
    K_new[0, 0] *= scale
    K_new[1, 1] *= scale
    K_new[0, 2] = (cam_K[0, 2] - x1) * scale + pad_left
    K_new[1, 2] = (cam_K[1, 2] - y1) * scale + pad_top

    # --- to tensor ---
    pil = Image.fromarray(img_padded)
    img_padded = to_tensor(pil)

    return img_padded, mask_padded, depth_padded, K_new, valid_flag

def crop_input_rgb(image, mask, cam_K, target_size, padding_scale=1.5):
    """
    Same as previous crop, but accepts tensors or numpy arrays directly.

    Args:
        cam_K (np.ndarray): 3x3 intrinsic matrix
        image (np.ndarray or torch.Tensor): RGB image [H,W,3] or [3,H,W]
        mask (np.ndarray or torch.Tensor): grayscale mask [H,W]
        target_size (tuple): (width, height)
        padding_scale (float): bbox padding

    Returns:
        image_tensor: torch.Tensor [3, H, W]
        padded_mask: np.ndarray [H, W]
        K_new: np.ndarray [3, 3]
    """
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if image.shape[0] == 3:  # [3, H, W] → [H, W, 3]
            image = np.transpose(image, (1, 2, 0))
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    if isinstance(cam_K, torch.Tensor):
        cam_K = cam_K.detach().cpu().numpy()

    image = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
    mask = mask.astype(np.uint8)

    # --- find bbox + padded box coords ---
    ys, xs = np.where(mask > 0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    w_box = x_max - x_min
    h_box = y_max - y_min
    cx, cy = x_min + w_box / 2, y_min + h_box / 2
    half_w = w_box / 2 * padding_scale
    half_h = h_box / 2 * padding_scale

    x1 = int(max(0, cx - half_w))
    x2 = int(min(image.shape[1], cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(image.shape[0], cy + half_h))

    cropped_img = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    # --- resize ---
    tgt_w, tgt_h = target_size
    crop_w, crop_h = x2 - x1, y2 - y1
    scale = min(tgt_w / crop_w, tgt_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    img_resized = cv2.resize(cropped_img, (new_w, new_h))
    mask_resized = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- pad ---
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(img_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    mask_padded = cv2.copyMakeBorder(mask_resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)

    # --- apply mask ---
    img_padded = img_padded * (mask_padded[:, :, None] > 0)

    # --- adjust intrinsics ---
    K_new = cam_K.copy()
    K_new[0, 0] *= scale
    K_new[1, 1] *= scale
    K_new[0, 2] = (cam_K[0, 2] - x1) * scale + pad_left
    K_new[1, 2] = (cam_K[1, 2] - y1) * scale + pad_top

    # --- to tensor ---
    pil = Image.fromarray(img_padded)
    img_padded = to_tensor(pil)

    return img_padded, mask_padded, K_new

def crop_cnos(cam_K, image_path, mask, target_size, padding_scale=1.5, depth_path=None):
    """
    Center‐crop based on mask bbox, then resize with uniform scale + pad to target_size,
    and adjust intrinsics so there’s no non‐uniform distortion.

    Args:
        cam_K (np.ndarray): 3x3 intrinsic matrix
        image_path (str): path to RGB image
        target_size (tuple): (width, height)
        padding_scale (float): factor to expand bbox before cropping

    Returns:
        resized_tensor (torch.Tensor): normalized, padded tensor image
        padded_mask (np.ndarray): binary mask at target_size
        K_new (np.ndarray): adjusted intrinsic matrix
    """
    # --- load ---
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if depth_path is not None:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # --- find bbox + padded box coords ---
    ys, xs = np.where(mask>0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # expand box
    w_box = x_max - x_min
    h_box = y_max - y_min
    cx, cy = x_min + w_box/2, y_min + h_box/2
    half_w = w_box/2 * padding_scale
    half_h = h_box/2 * padding_scale

    x1 = int(max(0, cx - half_w))
    x2 = int(min(image.shape[1], cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(image.shape[0], cy + half_h))

    # --- crop ---
    cropped_img  = image[y1:y2, x1:x2]
    cropped_mask = mask [y1:y2, x1:x2]
    if depth_path is not None:
        cropped_depth = depth_map[y1:y2, x1:x2]

    # --- compute uniform scale + resized dims ---
    tgt_w, tgt_h = target_size
    crop_w, crop_h = x2 - x1, y2 - y1
    scale = min(tgt_w / crop_w, tgt_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # resize
    img_resized  = cv2.resize(cropped_img,  (new_w, new_h))
    mask_resized= cv2.resize(cropped_mask,(new_w, new_h))
    if depth_path is not None:
        depth_resized = cv2.resize(cropped_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- pad to target_size (centered) ---
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    mask_padded = cv2.copyMakeBorder(
        mask_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )
    if depth_path is not None:
        depth_padded = cv2.copyMakeBorder(
            depth_resized, pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT, value=0
        )

    # mask the img
    img_padded = img_padded * (mask_padded[:, :, None] > 0)
    if depth_path is not None:
        depth_padded = depth_padded * (mask_padded > 0)

    # --- adjust intrinsics ---
    K_new = cam_K.copy()
    # uniform scale on focal lengths
    K_new[0,0] *= scale
    K_new[1,1] *= scale
    # principal point: original cx,cy → shifted by crop and pad
    cx_new = (cam_K[0,2] - x1) * scale + pad_left
    cy_new = (cam_K[1,2] - y1) * scale + pad_top
    K_new[0,2] = cx_new
    K_new[1,2] = cy_new

    # --- to tensor ---
    pil = Image.fromarray(img_padded)
    tensor = to_tensor(pil)

    if depth_path is None:
        depth_padded = None

    return tensor, mask_padded, K_new, depth_padded

def crop_template(image_path, mask_path, xyz_path, target_size, padding_scale=1.5):
    """
    Center‐crop based on mask bbox, then resize with uniform scale + pad to target_size,
    and adjust intrinsics so there’s no non‐uniform distortion.

    Args:
        cam_K (np.ndarray): 3x3 intrinsic matrix
        image_path (str): path to RGB image
        mask_path (str): path to grayscale mask image
        target_size (tuple): (width, height)
        padding_scale (float): factor to expand bbox before cropping

    Returns:
        resized_tensor (torch.Tensor): normalized, padded tensor image
        padded_mask (np.ndarray): binary mask at target_size
        K_new (np.ndarray): adjusted intrinsic matrix
    """
    # --- load ---
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    xyz_map = np.load(xyz_path)

    # --- find bbox + padded box coords ---
    ys, xs = np.where(mask>0)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # expand box
    w_box = x_max - x_min
    h_box = y_max - y_min
    cx, cy = x_min + w_box/2, y_min + h_box/2
    half_w = w_box/2 * padding_scale
    half_h = h_box/2 * padding_scale

    x1 = int(max(0, cx - half_w))
    x2 = int(min(image.shape[1], cx + half_w))
    y1 = int(max(0, cy - half_h))
    y2 = int(min(image.shape[0], cy + half_h))

    # --- crop ---
    cropped_img  = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    cropped_xyz = xyz_map[y1:y2, x1:x2]

    # --- compute uniform scale + resized dims ---
    tgt_w, tgt_h = target_size
    crop_w, crop_h = x2 - x1, y2 - y1
    scale = min(tgt_w / crop_w, tgt_h / crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    # resize
    img_resized  = cv2.resize(cropped_img,  (new_w, new_h))
    mask_resized= cv2.resize(cropped_mask,(new_w, new_h))
    xyz_resized = cv2.resize(cropped_xyz, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # --- pad to target_size (centered) ---
    pad_w = tgt_w - new_w
    pad_h = tgt_h - new_h
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0,0,0]
    )
    mask_padded = cv2.copyMakeBorder(
        mask_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=0
    )
    xyz_padded = cv2.copyMakeBorder(
        xyz_resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[0,0,0]
    )

    # mask the img and xyz 
    img_padded = img_padded * (mask_padded[:, :, None] > 0)
    xyz_padded = xyz_padded * (mask_padded[:, :, None] > 0)
    # --- to tensor ---
    pil = Image.fromarray(img_padded)
    tensor = to_tensor(pil)

    return tensor, mask_padded, xyz_padded