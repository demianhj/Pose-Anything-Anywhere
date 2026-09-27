import math
import os
import os.path as osp
import pickle
import json
from dataclasses import dataclass
from typing import Dict, List

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_dir
from matplotlib import colormaps as cmx
from torch import nn

from inference_utils.pose_estimation import robust_umeyama
from inference_utils.utils import (
    backproject_depth_to_points,
    estimate_intrinsics_from_pointmap,
    to_tensor,
    transform_points,
    umeyama_pose,
)

def depth_to_cam_points(
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    stride: int = 1,
) -> np.ndarray:
    """
    Convert a depth map to 3D camera coordinates, with optional stride downsampling.

    Args:
        depth_map (np.ndarray): (H, W) depth map.
        intrinsic (np.ndarray): (3, 3) camera intrinsic matrix.
        stride (int): Downsampling stride. Default=1 (no downsample).

    Returns:
        cam_coords (np.ndarray): (H/stride, W/stride, 3) camera-space points.
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # Downsample grid
    u, v = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    depth_sampled = depth_map[v, u]

    # Backproject to 3D
    z = depth_sampled.astype(np.float32)
    x = (u - cu) * z / fu
    y = (v - cv) * z / fv

    cam_coords = np.stack([x, y, z], axis=-1)  # shape: (H/stride, W/stride, 3)

    return cam_coords

def depth_to_cam_coords_points(
    depth_map: np.ndarray,
    intrinsic: np.ndarray,
    stride: int = 1,
):
    """Convert a depth map to valid 3D camera-space points."""
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "Intrinsic matrix must be 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "Intrinsic matrix must have zero skew"

    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    u, v = np.meshgrid(np.arange(0, W, stride), np.arange(0, H, stride))
    depth_sampled = depth_map[v, u]

    valid_mask = np.isfinite(depth_sampled) & (depth_sampled > 1e-6)
    if not np.any(valid_mask):
        return (
            np.zeros((0, 3), np.float32),
            np.zeros((0,), np.int32),
            np.zeros((0,), np.int32),
        )

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth_sampled[valid_mask]

    x = (u_valid - cu) * z_valid / fu
    y = (v_valid - cv) * z_valid / fv
    pts_cam = np.stack([x, y, z_valid], axis=-1).astype(np.float32)

    return pts_cam, u_valid, v_valid


def depth_to_world_coords_points(
    depth_map: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps: float = 1e-8,
    stride: int = 1,
):
    """Convert a depth map to valid world-space points and a dense world point map."""
    if depth_map is None or not np.any(depth_map > eps):
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((depth_map.shape[0] // stride, depth_map.shape[1] // stride, 3), dtype=np.float32),
            np.zeros((depth_map.shape[0] // stride, depth_map.shape[1] // stride), dtype=bool),
        )

    pts_cam, u_valid, v_valid = depth_to_cam_coords_points(depth_map, intrinsic, stride=stride)

    if extrinsic.shape == (3, 4):
        extrinsic_h = np.eye(4, dtype=np.float32)
        extrinsic_h[:3, :4] = extrinsic
    elif extrinsic.shape == (4, 4):
        extrinsic_h = extrinsic.copy()
    else:
        raise ValueError("Extrinsic must be shape (3,4) or (4,4)")

    R_w2c = extrinsic_h[:3, :3]
    t_w2c = extrinsic_h[:3, 3]
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    pts_world = (pts_cam @ R_c2w.T) + t_c2w

    Hs, Ws = depth_map.shape[0] // stride, depth_map.shape[1] // stride
    pointmap_world = np.zeros((Hs, Ws, 3), dtype=np.float32)
    valid_mask = np.zeros((Hs, Ws), dtype=bool)

    u_sub = (u_valid // stride).astype(int)
    v_sub = (v_valid // stride).astype(int)
    pointmap_world[v_sub, u_sub] = pts_world
    valid_mask[v_sub, u_sub] = True

    return pts_world, pointmap_world, valid_mask


def unproject_depth_map_to_point_map(
    depth_map: np.ndarray,
    extrinsics_cam: np.ndarray,
    intrinsics_cam: np.ndarray,
    stride: int = 1,
):
    """Unproject a batch of depth maps to world-space point lists and dense maps."""
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.detach().cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.detach().cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.detach().cpu().numpy()

    if depth_map.ndim == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map[..., 0]

    world_points_list, world_pointmaps = [], []
    for frame_idx in range(depth_map.shape[0]):
        pts_world, world_map, _ = depth_to_world_coords_points(
            depth_map[frame_idx],
            extrinsics_cam[frame_idx],
            intrinsics_cam[frame_idx],
            stride=stride,
        )
        world_points_list.append(pts_world)
        world_pointmaps.append(world_map)

    return world_points_list, world_pointmaps


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.original = original
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original(x) + self.dropout(self.lora_B(self.lora_A(x))) * self.scaling


def lora_to_global_attention(model, r=8, alpha=32, dropout=0.05):
    for blk in model.aggregator.frame_blocks + model.aggregator.global_blocks:
        attn = blk.attn
        for name in ("qkv", "proj"):
            orig: nn.Linear = getattr(attn, name)
            setattr(attn, name, LoRALinear(orig, r=r, alpha=alpha, dropout=dropout))

    for param_name, param in model.named_parameters():
        param.requires_grad = ("lora_" in param_name) or param_name.startswith("point_head")

    return model


@dataclass
class FrameGeometry:
    frame_id: int
    point_map: np.ndarray
    depth_map: np.ndarray
    mask: np.ndarray
    K: np.ndarray
    T_cam2canonical: np.ndarray
    pts_world: np.ndarray
    pts_rgb: np.ndarray
    pts_conf: np.ndarray


@dataclass
class LocalMapGeometry:
    center_id: int
    neighbor_ids: List[int]
    frames: Dict[int, FrameGeometry]
    relative_poses: Dict[tuple, np.ndarray]


def find_overlapping_frames(local_maps):
    overlaps = []
    for i, lm_i in enumerate(local_maps):
        for j, lm_j in enumerate(local_maps):
            if i >= j:
                continue
            shared = set(lm_i.frames.keys()) & set(lm_j.frames.keys())
            if shared:
                overlaps.append((i, j, list(shared)))
    return overlaps


def register_local_maps(local_maps):
    transforms = {}
    overlaps = find_overlapping_frames(local_maps)

    for i, j, shared_frames in overlaps:
        pts_i, pts_j = [], []
        for fid in shared_frames:
            pts_i.append(local_maps[i].frames[fid].pts_world)
            pts_j.append(local_maps[j].frames[fid].pts_world)
        pts_i = np.concatenate(pts_i, axis=0)
        pts_j = np.concatenate(pts_j, axis=0)

        _, s, R, t = umeyama_pose(pts_j, pts_i, with_scaling=True)

        T_ij = np.eye(4)
        T_ij[:3, :3] = s * R
        T_ij[:3, 3] = t

        transforms[(i, j)] = {"T": T_ij, "scale": s}
        transforms[(j, i)] = {"T": np.linalg.inv(T_ij), "scale": 1.0 / s}
        print(f"🌐 Local Map {i} ↔ {j}: {len(shared_frames)} shared frames, estimated transform.")
    return transforms


def set_axes_equal(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    span = limits[:, 1] - limits[:, 0]
    centers = limits.mean(axis=1)
    radius = 0.5 * max(span)
    for ctr, ax_fn in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        ax_fn([ctr - radius, ctr + radius])


def set_view_from_camera(ax, T_cam, target=None, dis=0.2, invert_z=True):
    cam_center = T_cam[:3, 3]
    z_axis = T_cam[:3, 2]
    if invert_z:
        z_axis = -z_axis

    if target is None:
        target = cam_center + dis * z_axis

    view_dir = target - cam_center
    r = np.linalg.norm(view_dir)
    if r < 1e-6:
        elev, azim = 20, 40
    else:
        x, y, z = view_dir / r
        elev = np.degrees(np.arcsin(z))
        azim = np.degrees(np.arctan2(y, x))

    ax.view_init(elev=elev + 180, azim=azim + 90)
    return elev, azim


def draw_camera_axes(ax, T, scale=0.05, color="b", alpha=1.0, label=None):
    o = T[:3, 3]
    R = T[:3, :3]

    ax.quiver(o[0], o[1], o[2], R[0, 0] * scale, R[1, 0] * scale, R[2, 0] * scale, color="r", alpha=alpha)
    ax.quiver(o[0], o[1], o[2], R[0, 1] * scale, R[1, 1] * scale, R[2, 1] * scale, color="g", alpha=alpha)
    ax.quiver(o[0], o[1], o[2], R[0, 2] * scale, R[1, 2] * scale, R[2, 2] * scale, color="b", alpha=alpha)
    ax.scatter(o[0], o[1], o[2], s=20, color=color, alpha=alpha, edgecolors="k", linewidth=0.5)
    if label is not None:
        ax.text(o[0], o[1], o[2], label, color=color, fontsize=8)


def save_figure(fig, save_path, dpi=240, message="💾 saved to"):
    ensure_parent(save_path)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"{message} {save_path}")


def _reachable_global_transforms(local_map_transforms, anchor_id=0):
    T_global = {anchor_id: np.eye(4)}
    visited = {anchor_id}
    queue = [anchor_id]
    while queue:
        i = queue.pop(0)
        Ti = T_global[i]
        for (src, tgt), info in local_map_transforms.items():
            if src != i or tgt in visited:
                continue
            T_ij = info["T"] if isinstance(info, dict) else info
            T_global[tgt] = Ti @ T_ij
            visited.add(tgt)
            queue.append(tgt)
    return T_global, visited


def assemble_and_visualize_submaps(
    local_maps,
    local_map_transforms,
    anchor_id=0,
    max_points_plot=200_000,
    show=True,
    save_path=None,
):
    T_global, visited = _reachable_global_transforms(local_map_transforms, anchor_id)
    print(f"✅ Connected submaps from anchor {anchor_id}: {sorted(list(visited))}")

    pts_all, colors_all = [], []
    cmap = cmx.get_cmap("tab20")

    for k, i in enumerate(sorted(list(visited))):
        T_w_li = T_global[i]
        fallback_color = np.array(cmap(k % 20))[:3]
        sub_pts, sub_colors = [], []

        for f in local_maps[i].frames.values():
            if f.pts_world is None or len(f.pts_world) == 0:
                continue
            sub_pts.append(f.pts_world)
            if hasattr(f, "pts_rgb") and f.pts_rgb is not None:
                sub_colors.append(f.pts_rgb)
            else:
                sub_colors.append(np.tile(fallback_color, (len(f.pts_world), 1)))

        if not sub_pts:
            continue

        sub_pts = np.concatenate(sub_pts, axis=0)
        sub_colors = np.concatenate(sub_colors, axis=0)
        pts_all.append(transform_points(T_w_li, sub_pts))
        colors_all.append(sub_colors)

    if not pts_all:
        print("⚠️ No points to plot.")
        return np.zeros((0, 3)), np.zeros((0, 3)), T_global

    P = np.concatenate(pts_all, axis=0)
    C = np.concatenate(colors_all, axis=0)

    if len(P) > max_points_plot:
        idx = np.random.choice(len(P), max_points_plot, replace=False)
        P = P[idx]
        C = C[idx]
        print(f"⚙️ Downsampled for plotting: {max_points_plot} points")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=5.0, c=C, marker=".")

    center = P.mean(axis=0)
    ax.scatter(center[0], center[1], center[2], c="r", s=20, label="object center")

    T_cam0 = list(T_global.values())[0]
    set_axes_equal(ax)
    set_view_from_camera(ax, T_cam0, target=center)
    ax.legend(markerscale=3)

    ax.set_title(f"Registered Submaps from Camera0 View (anchor={anchor_id})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return P, C, T_global


def assemble_and_visualize_geometry(
    local_maps,
    local_map_transforms,
    anchor_id=0,
    max_points=20_000,
    scale=0.1,
    show=True,
    save_path=None,
):
    print(f"🔹 Assembling global geometry (anchor={anchor_id})")
    T_global, visited = _reachable_global_transforms(local_map_transforms, anchor_id)
    print(f"✅ Connected {len(visited)}/{len(local_maps)} submaps from anchor {anchor_id}")

    all_pts = []
    for i in visited:
        local_map = local_maps[i]
        if not local_map.frames:
            continue
        T_w_li = T_global[i]
        key_fid = local_map.center_id if local_map.center_id in local_map.frames else sorted(local_map.frames.keys())[0]
        pts_local = local_map.frames[key_fid].pts_world
        all_pts.append((T_w_li[:3, :3] @ pts_local.T + T_w_li[:3, 3:4]).T)

    if not all_pts:
        print("⚠️ No valid points found.")
        return

    all_pts = np.concatenate(all_pts, axis=0)
    if len(all_pts) > max_points:
        idx = np.random.choice(len(all_pts), max_points, replace=False)
        all_pts = all_pts[idx]
        print(f"⚙️ Downsampled global plot to {max_points} points")

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], c="0.6", s=0.4, marker=".")

    for i, T in T_global.items():
        if i == anchor_id:
            draw_camera_axes(ax, T, scale=scale, color="red", alpha=1.0, label=f"Anchor {i}")
        else:
            draw_camera_axes(ax, T, scale=scale, color="royalblue", alpha=0.8, label=f"Cam {i}")

    center = all_pts.mean(axis=0)
    T_cam0 = list(T_global.values())[0]
    set_axes_equal(ax)
    set_view_from_camera(ax, T_cam0, target=center, dis=1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Global Registered Geometry (anchor={anchor_id})")
    ax.legend(markerscale=4)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return all_pts, T_global


def assemble_global_geometry(local_maps, local_map_transforms, anchor_id=0, max_points=20_000):
    print(f"🔹 Assembling global geometry (anchor={anchor_id})")
    T_global, visited = _reachable_global_transforms(local_map_transforms, anchor_id)
    connected_ids = sorted(list(visited))
    print(f"✅ Connected {len(visited)}/{len(local_maps)} submaps from anchor {anchor_id}")

    all_pts, all_colors = [], []
    framewise_points = {}

    for i in connected_ids:
        local_map = local_maps[i]
        if not local_map.frames:
            continue
        T_w_li = T_global[i]
        key_fid = local_map.center_id if local_map.center_id in local_map.frames else sorted(local_map.frames.keys())[0]
        frame = local_map.frames[key_fid]
        pts_global = (T_w_li[:3, :3] @ frame.pts_world.T + T_w_li[:3, 3:4]).T

        framewise_points[key_fid] = {
            "pts": pts_global,
            "rgb": getattr(frame, "pts_rgb", None),
            "pts_conf": getattr(frame, "pts_conf", None),
            "T_global": T_w_li,
        }

        all_pts.append(pts_global)
        if frame.pts_rgb is not None:
            all_colors.append(frame.pts_rgb)

    if not all_pts:
        print("⚠️ No valid points found.")
        return None, None, {}, T_global, connected_ids

    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0) if all_colors else None

    if len(all_pts) > max_points:
        idx = np.random.choice(len(all_pts), max_points, replace=False)
        all_pts = all_pts[idx]
        all_colors = all_colors[idx] if all_colors is not None else None
        print(f"⚙️ Downsampled global cloud to {max_points} points")

    return all_pts, all_colors, framewise_points, T_global, connected_ids


def visualize_global_geometry(all_pts, T_global, anchor_id=0, scale=0.1, show=True, save_path=None):
    if all_pts is None or len(all_pts) == 0:
        print("⚠️ Empty geometry — nothing to visualize.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], c="0.6", s=0.4, marker=".")

    for i, T in T_global.items():
        if i == anchor_id:
            draw_camera_axes(ax, T, scale=scale, color="red", alpha=1.0, label=f"Anchor {i}")
        else:
            draw_camera_axes(ax, T, scale=scale, color="royalblue", alpha=0.8, label=f"Cam {i}")

    center = all_pts.mean(axis=0)
    T_cam0 = list(T_global.values())[0]
    set_axes_equal(ax)
    set_view_from_camera(ax, T_cam0, target=center, dis=1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Global Registered Geometry (anchor={anchor_id})")
    ax.legend(markerscale=4)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, message="💾 saved visualization to")
    if show:
        plt.show()
    else:
        plt.close(fig)


def visualize_ref_pred(
    ref_pc: np.ndarray,
    other_pts_list: list[np.ndarray],
    T_global: dict = None,
    anchor_id: int = 0,
    scale: float = 0.1,
    show: bool = True,
    save_path: str = None,
):
    if ref_pc is None or len(ref_pc) == 0:
        print("⚠️ No reference points to visualize.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(ref_pc[:, 0], ref_pc[:, 1], ref_pc[:, 2], c="red", s=0.6, marker=".", label="Reference")

    for idx, pts in enumerate(other_pts_list):
        if pts is None or len(pts) == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="royalblue", s=0.4, marker=".", alpha=0.6, label=f"Frame {idx}")

    if T_global is not None:
        for i, T in T_global.items():
            if i == anchor_id:
                draw_camera_axes(ax, T, scale=scale, color="red", alpha=1.0, label=f"Anchor {i}")
            else:
                draw_camera_axes(ax, T, scale=scale, color="navy", alpha=0.8, label=f"Cam {i}")

    all_pts = np.concatenate([ref_pc] + [p for p in other_pts_list if p is not None], axis=0)
    center = all_pts.mean(axis=0)

    if T_global:
        T_cam0 = list(T_global.values())[0]
        set_view_from_camera(ax, T_cam0, target=center, dis=1.5)
    else:
        ax.view_init(elev=20, azim=60)

    set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Reference vs. Transformed Frames")
    ax.legend(markerscale=4)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path, message="💾 saved visualization to")
    if show:
        plt.show()
    else:
        plt.close(fig)


def save_ply(path, pts, colors=None):
    os.makedirs(osp.dirname(path), exist_ok=True)
    pts = np.asarray(pts).reshape(-1, 3)
    n = len(pts)
    if colors is None:
        colors = np.ones((n, 3), dtype=np.uint8) * 200
    else:
        colors = (np.clip(colors, 0, 1) * 255).astype(np.uint8).reshape(-1, 3)

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(pts, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_float(value, default=None):
    if value is None:
        return default
    if isinstance(value, str) and value.lower() in {"inf", "+inf", "infinity", "+infinity"}:
        return np.inf
    return float(value)


def format_path(root, pattern, **kwargs):
    return osp.join(root, pattern.format(**kwargs))


def output_root(cfg):
    return cfg["outputs"]["root"].format(dataset_name=cfg["dataset_name"])


def output_dir(cfg, key):
    out_cfg = cfg["outputs"]
    path = out_cfg[key].format(dataset_name=cfg["dataset_name"])
    if osp.isabs(path):
        return path
    return osp.join(output_root(cfg), path)


def ensure_parent(path):
    os.makedirs(osp.dirname(path), exist_ok=True)


def load_reference_data(obj_id, cfg, stride, scene_id=None):
    ref_cfg = cfg["reference"]
    ref_base = ref_cfg["base_dir"]
    object_dir = ref_cfg["object_dir_pattern"].format(obj_id=obj_id, scene_id=scene_id)
    ref_dir = osp.join(ref_base, object_dir)

    camera_path = osp.join(ref_dir, ref_cfg["camera_file"])
    npz = np.load(camera_path, allow_pickle=True)
    intrinsics = npz[ref_cfg["intrinsics_key"]]
    extrinsics = npz[ref_cfg["extrinsics_key"]]

    ref_images, ref_depths, ref_masks = [], [], []
    for i in range(ref_cfg["num_templates"]):
        ref_image_path = format_path(ref_dir, ref_cfg["rgb_pattern"], idx=i)
        ref_depth_path = format_path(ref_dir, ref_cfg["depth_pattern"], idx=i)
        ref_mask_path = format_path(ref_dir, ref_cfg["mask_pattern"], idx=i)

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_depth = cv2.imread(ref_depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        ref_mask = cv2.imread(ref_mask_path, cv2.IMREAD_UNCHANGED).astype(bool)

        ref_image = ref_image * ref_mask[..., None]
        ref_depth = ref_depth * ref_mask

        ref_images.append(to_tensor(ref_image))
        ref_depths.append(ref_depth)
        ref_masks.append(ref_mask)

    ref_images = torch.stack(ref_images, dim=0)
    ref_masks = np.stack(ref_masks, axis=0)
    ref_depths = np.stack(ref_depths, axis=0)
    ref_pcs, _ = unproject_depth_map_to_point_map(ref_depths, extrinsics, intrinsics, stride=stride)

    return object_dir, ref_images, ref_depths, ref_masks, ref_pcs


def select_top_refs(num_matches, order, top_k, min_match_threshold):
    num_matches_arr = np.array(num_matches)
    valid_idx = np.where(num_matches_arr >= min_match_threshold)[0]
    if len(valid_idx) == 0:
        return []
    top_ref_idx = valid_idx[np.argsort(num_matches_arr[valid_idx])[::-1][:top_k]]
    return [order[1:][k] for k in top_ref_idx], top_ref_idx


def build_local_map(
    q,
    selected_ref_ids,
    order,
    images_ordered,
    masks,
    point_map,
    depth_map,
    depth_conf,
    K,
    cfg,
):
    inf_cfg = cfg["inference"]
    local_frames = {}
    all_fids = [q] + selected_ref_ids
    z_max = resolve_float(inf_cfg.get("z_max"), np.inf)

    for i in all_fids:
        pts_depth, (uu, vv) = backproject_depth_to_points(
            depth_map=depth_map[order.index(i)],
            K=K,
            mask=masks[order.index(i)],
            conf_map=(depth_conf[order.index(i)] if inf_cfg["use_conf"] else None),
            stride=inf_cfg["depth_stride"],
            z_min=resolve_float(inf_cfg["z_min"]),
            z_max=z_max,
        )
        if pts_depth.shape[0] < inf_cfg["min_pose_points"]:
            print(f"Frame {i}: too few points, skip")
            continue

        rgb_img = images_ordered[order.index(i)].permute(1, 2, 0).cpu().numpy()
        pts_rgb = rgb_img[vv, uu]
        pts_conf = depth_conf[order.index(i)][vv, uu]
        pts_3d = point_map[order.index(i)][vv, uu]
        T, _, _, _ = umeyama_pose(pts_depth, pts_3d, with_scaling=True)

        local_frames[i] = FrameGeometry(
            frame_id=i,
            point_map=point_map[order.index(i)],
            depth_map=depth_map[order.index(i)],
            mask=masks[order.index(i)],
            K=K,
            T_cam2canonical=T,
            pts_world=transform_points(T, pts_depth),
            pts_rgb=pts_rgb,
            pts_conf=pts_conf,
        )

    relative_poses = {}
    available_fids = list(local_frames.keys())
    for a in available_fids:
        for b in available_fids:
            if a == b:
                continue
            Ta = local_frames[a].T_cam2canonical
            Tb = local_frames[b].T_cam2canonical
            relative_poses[(a, b)] = np.linalg.inv(Ta) @ Tb

    return LocalMapGeometry(
        center_id=q,
        neighbor_ids=[fid for fid in selected_ref_ids if fid in local_frames],
        frames=local_frames,
        relative_poses=relative_poses,
    )


def save_local_map_plot(local_map, cmap, output_path):
    if not local_map.frames:
        return

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for fid, fgeo in local_map.frames.items():
        pts = fgeo.pts_world
        color = cmap(fid % cmap.N)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=[color], label=f"Frame {fid}")
    ax.legend()
    ax.set_title(f"Local Map Center={local_map.center_id}")
    save_figure(fig, output_path, dpi=200)
    plt.close(fig)


def save_match_plot(images_ordered, filtered_query_xys, filtered_pred_xys, output_path):
    from training.vis_utils import visualize_correpondence_multi_query

    vis_query_image = images_ordered[0].permute(1, 2, 0).cpu().numpy()
    vis_ref_images = images_ordered[1:].permute(0, 2, 3, 1).cpu().numpy()

    ensure_parent(output_path)
    visualize_correpondence_multi_query(
        query_image=vis_query_image,
        pred_images=vis_ref_images,
        filtered_query_xys=filtered_query_xys,
        filtered_pred_xys=filtered_pred_xys,
        save_path=output_path,
    )


def estimate_matches(VGGT_model, aggregated_tokens_list, images, ps_idx, point_map, masks, cfg, device, context=None):
    inf_cfg = cfg["inference"]
    masked_pts3d = point_map * masks[:, :, :, None]
    sampled_choose = []
    context = context or {}

    for i in range(masks.shape[0]):
        choose = masks[i].astype(np.float32).flatten().nonzero()[0]
        if len(choose) == 0:
            if context:
                print(
                    f"⚠️ Scene {context.get('scene_id')}, Obj {context.get('obj_id')}, "
                    f"Frame {i}: mask empty."
                )
            sampled_choose.append(np.zeros((inf_cfg["n_sample_point"],), dtype=np.int64))
        elif len(choose) <= inf_cfg["n_sample_point"]:
            choose_idx = np.random.choice(np.arange(len(choose)), inf_cfg["n_sample_point"])
            sampled_choose.append(choose[choose_idx])
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), inf_cfg["n_sample_point"], replace=False)
            sampled_choose.append(choose[choose_idx])

    masked_pts3d = torch.from_numpy(masked_pts3d[None]).float().to(device)
    sampled_choose = torch.from_numpy(np.array(sampled_choose)).long().to(device)[None]
    matching_res = VGGT_model.track_head(aggregated_tokens_list, images, ps_idx, masked_pts3d, sampled_choose)

    filtered_query_xys, filtered_pred_xys, num_matches = [], [], []
    H, W = masks.shape[-2:]
    for ref_idx in range(masks.shape[0] - 1):
        pred_match = matching_res[ref_idx]["pred_label"].cpu().numpy()
        valid_mask = pred_match[0] > 0
        valid_indices = np.flatnonzero(valid_mask)
        target_pos = pred_match[0, valid_indices].astype(np.int64)
        target_pool_size = sampled_choose[0][ref_idx + 1].cpu().numpy().shape[0]
        valid = (target_pos >= 0) & (target_pos < target_pool_size)

        if valid.sum() < inf_cfg["min_valid_matches"]:
            print(f"Not enough valid matches for pose estimation: {valid.sum()}")
            filtered_query_xys.append(np.array([[-1, -1]]))
            filtered_pred_xys.append(np.array([[-1, -1]]))
            num_matches.append(-1)
            continue

        valid_indices = valid_indices[valid]
        target_pos = target_pos[valid]
        indices_1 = sampled_choose[0][0].cpu().numpy()[valid_indices]
        indices_2 = sampled_choose[0][ref_idx + 1].cpu().numpy()[target_pos]
        num_matches.append(len(indices_2))

        ys_1, xs_1 = np.divmod(indices_1, W)
        ys_2, xs_2 = np.divmod(indices_2, W)
        filtered_query_xys.append(np.stack([xs_1, ys_1], axis=-1))
        filtered_pred_xys.append(np.stack([xs_2, ys_2], axis=-1))

    return filtered_query_xys, filtered_pred_xys, num_matches


def run_object_sfm(obj_id, VGGT_model, cfg, device, dtype, scene_id=None):
    inf_cfg = cfg["inference"]
    vis_cfg = cfg["visualization"]
    out_cfg = cfg["outputs"]

    object_dir, ref_images, _, ref_masks, ref_pcs = load_reference_data(
        obj_id,
        cfg,
        stride=inf_cfg["depth_stride"],
        scene_id=scene_id,
    )
    cmap = cm.get_cmap(vis_cfg["local_map_colormap"], vis_cfg["local_map_color_count"])

    local_maps = []
    for q in range(cfg["reference"]["num_templates"]):
        order = [q] + [k for k in range(cfg["reference"]["num_templates"]) if k != q]
        print(f"🔹 Frame {q}: query={q}, order={order}")

        images_ordered = torch.stack([ref_images[i] for i in order], dim=0)
        masks_ordered = np.stack([ref_masks[i] for i in order], axis=0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=dtype):
                images = images_ordered[None].to(device)
                aggregated_tokens_list, ps_idx = VGGT_model.aggregator(images)

            point_map, _ = VGGT_model.point_head(aggregated_tokens_list, images, ps_idx)
            point_map = point_map.squeeze(0).cpu().numpy()

            depth_map, depth_conf = VGGT_model.depth_head(aggregated_tokens_list, images, ps_idx)
            depth_map = depth_map.squeeze(0).cpu().numpy()
            depth_conf = depth_conf.squeeze(0).cpu().numpy()

            K, _ = estimate_intrinsics_from_pointmap(
                point_map[0],
                masks_ordered[0],
                stride=inf_cfg["intrinsics_stride"],
                allow_skew=inf_cfg["allow_intrinsics_skew"],
            )

            filtered_query_xys, filtered_pred_xys, num_matches = estimate_matches(
                VGGT_model,
                aggregated_tokens_list,
                images,
                ps_idx,
                point_map,
                masks_ordered,
                cfg,
                device,
                context={"scene_id": scene_id, "obj_id": obj_id},
            )

            selected = select_top_refs(num_matches, order, inf_cfg["top_k"], inf_cfg["min_match_threshold"])
            if not selected:
                local_maps.append(LocalMapGeometry(center_id=q, neighbor_ids=[], frames={}, relative_poses={}))
                print(f"⚠️ Frame {q}: no valid refs above {inf_cfg['min_match_threshold']} matches, skipping")
                continue

            selected_ref_ids, top_ref_idx = selected
            print(f"Query {q}: top-{inf_cfg['top_k']} refs = {selected_ref_ids}  counts={np.array(num_matches)[top_ref_idx]}")

            local_map = build_local_map(
                q,
                selected_ref_ids,
                order,
                images_ordered,
                masks_ordered,
                point_map,
                depth_map,
                depth_conf,
                K,
                cfg,
            )

            local_map_path = format_path(
                osp.join(output_dir(cfg, "local_map_dir"), object_dir),
                out_cfg["local_map_pattern"],
                scene_id=scene_id,
                obj_id=obj_id,
                query_id=q,
            )
            save_local_map_plot(local_map, cmap, local_map_path)

            match_path = format_path(
                osp.join(output_dir(cfg, "match_dir"), object_dir),
                out_cfg["match_pattern"],
                scene_id=scene_id,
                obj_id=obj_id,
                query_id=q,
            )
            save_match_plot(images_ordered, filtered_query_xys, filtered_pred_xys, match_path)

        local_maps.append(local_map)

    print("start global registration")
    local_map_transforms = register_local_maps(local_maps)

    assemble_and_visualize_submaps(
        local_maps,
        local_map_transforms,
        anchor_id=inf_cfg["anchor_id"],
        max_points_plot=vis_cfg["global_merge_max_points"],
        show=vis_cfg["show"],
        save_path=format_path(
            output_dir(cfg, "global_merge_dir"),
            out_cfg["global_merge_pattern"],
            scene_id=scene_id,
            obj_id=obj_id,
        ),
    )

    assemble_and_visualize_geometry(
        local_maps,
        local_map_transforms,
        anchor_id=inf_cfg["anchor_id"],
        max_points=vis_cfg["global_cam_max_points"],
        scale=vis_cfg["camera_axis_scale"],
        show=vis_cfg["show"],
        save_path=format_path(
            output_dir(cfg, "global_cam_dir"),
            out_cfg["global_cam_pattern"],
            scene_id=scene_id,
            obj_id=obj_id,
        ),
    )

    all_pts, _, framewise_points, T_global, connected_ids = assemble_global_geometry(
        local_maps,
        local_map_transforms,
        anchor_id=inf_cfg["anchor_id"],
        max_points=vis_cfg["global_geometry_max_points"],
    )
    visualize_global_geometry(
        all_pts,
        T_global,
        anchor_id=inf_cfg["anchor_id"],
        scale=vis_cfg["camera_axis_scale"],
        show=vis_cfg["show"],
        save_path=format_path(
            output_dir(cfg, "global_cam_dir"),
            out_cfg["global_cam_pattern"],
            scene_id=scene_id,
            obj_id=obj_id,
        ),
    )

    if inf_cfg["anchor_id"] not in framewise_points:
        print(f"⚠️ Anchor frame {inf_cfg['anchor_id']} missing from framewise_points; skipping object {obj_id}")
        return

    ref_anchor_pc = ref_pcs[inf_cfg["anchor_id"]]
    pts_anchor_global = framewise_points[inf_cfg["anchor_id"]]["pts"]
    pts_conf_anchor_global = framewise_points[inf_cfg["anchor_id"]]["pts_conf"]
    s, R, t = robust_umeyama(
        pts_anchor_global,
        ref_anchor_pc,
        conf_src=pts_conf_anchor_global,
        with_scaling=True,
    )

    T_global_to_ref = np.eye(4)
    T_global_to_ref[:3, :3] = s * R
    T_global_to_ref[:3, 3] = t

    framewise_points_ref = {}
    for fid, data in framewise_points.items():
        pts_g = data["pts"]
        pts_ref = (T_global_to_ref[:3, :3] @ pts_g.T + T_global_to_ref[:3, 3:4]).T
        framewise_points_ref[fid] = {
            "pts": pts_ref,
            "rgb": data.get("rgb", None),
            "T_global": T_global_to_ref @ data["T_global"],
        }

    other_pts_list = [data["pts"] for data in framewise_points_ref.values()]
    visualize_ref_pred(
        ref_pc=ref_anchor_pc,
        other_pts_list=other_pts_list,
        save_path=format_path(
            output_dir(cfg, "ref_coords_dir"),
            out_cfg["ref_coords_pattern"],
            scene_id=scene_id,
            obj_id=obj_id,
        ),
        show=vis_cfg["show"],
    )

    T_global_ref = {}
    sR = T_global_to_ref[:3, :3]
    t_ref = T_global_to_ref[:3, 3]
    R_ref = sR / np.linalg.norm(sR[:, 0])
    s = np.linalg.norm(sR[:, 0])

    for fid, T_g in T_global.items():
        R_g = T_g[:3, :3]
        t_g = T_g[:3, 3]

        T_new = np.eye(4)
        T_new[:3, :3] = R_ref @ R_g
        T_new[:3, 3] = s * (R_ref @ t_g) + t_ref
        T_global_ref[fid] = T_new

    global_map_data = {
        "global_pts": other_pts_list,
        "T_global": T_global_ref,
        "ids": connected_ids,
        "scale": s,
    }
    global_pts = np.concatenate(other_pts_list, axis=0)

    ply_path = format_path(output_dir(cfg, "ply_dir"), out_cfg["ply_pattern"], scene_id=scene_id, obj_id=obj_id)
    save_ply(ply_path, global_pts)
    print(connected_ids)

    pkl_path = format_path(output_dir(cfg, "pkl_dir"), out_cfg["pkl_pattern"], scene_id=scene_id, obj_id=obj_id)
    ensure_parent(pkl_path)
    with open(pkl_path, "wb") as f:
        pickle.dump(global_map_data, f)
    print(f"Saved global map to {pkl_path}")


def discover_scene_objects(cfg):
    scene_cfg = cfg.get("scenes")
    if not scene_cfg:
        return [(None, cfg["reference"]["obj_ids"])]

    scene_root = scene_cfg["root"]
    scene_ids = scene_cfg.get("scene_ids")
    if scene_ids is None:
        scene_ids = sorted([d for d in os.listdir(scene_root) if d.isdigit()])

    scene_objects = []
    for scene_id in scene_ids:
        scene_id = str(scene_id).zfill(scene_cfg.get("scene_id_width", 6))
        gt_path = osp.join(scene_root, scene_id, scene_cfg.get("gt_file", "scene_gt.json"))
        if not osp.exists(gt_path):
            print(f"Missing gt file for scene {scene_id}")
            continue

        with open(gt_path, "r") as f:
            scene_gt = json.load(f)

        obj_ids = sorted({
            ann["obj_id"]
            for frame_objs in scene_gt.values()
            for ann in frame_objs
        })
        if scene_cfg.get("obj_ids") is not None:
            allowed = set(scene_cfg["obj_ids"])
            obj_ids = [obj_id for obj_id in obj_ids if obj_id in allowed]

        print(f"Scene {scene_id}: found {len(obj_ids)} objects {obj_ids}")
        scene_objects.append((scene_id, obj_ids))

    return scene_objects


def run_configured_sfm(model, cfg, device, dtype):
    for scene_id, obj_ids in discover_scene_objects(cfg):
        if scene_id is not None:
            print(f"\n Processing Scene {scene_id}")
        for obj_id in obj_ids:
            if scene_id is None:
                print(f"Object {obj_id}")
            else:
                print(f"Scene {scene_id}, Object {obj_id}")
            run_object_sfm(obj_id, model, cfg, device, dtype, scene_id=scene_id)


def build_model(cfg, model_config_override=None):
    from training.lightning_pany import PANY

    model_config_name = model_config_override or cfg["model_config"]
    training_config_path = cfg["training_config_path"]
    if not osp.isabs(training_config_path):
        training_config_path = osp.abspath(training_config_path)

    with initialize_config_dir(version_base=None, config_dir=training_config_path):
        model_config = compose(config_name=model_config_name)

    model = PANY(model_config)
    model = lora_to_global_attention(model, **cfg["lora"])

    ckpt = torch.load(cfg["ckpt_path"], map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model
