import os
import os.path as osp
import pickle
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import trimesh
from hydra import compose, initialize_config_dir

from inference_utils.model import load_model
from inference_utils.multi_view_sfm import depth_to_cam_points, unproject_depth_map_to_point_map
from inference_utils.oryon_utils.metrics import compute_add, compute_adds
from inference_utils.oryon_utils.pcd import get_diameter
from inference_utils.pose_estimation import robust_umeyama
from inference_utils.utils import backproject_depth_to_points, to_tensor
from inference_utils.visualization import (
    calculate_2d_projections,
    draw_3d_bbox,
    get_3d_bbox,
)


def resolve_device_and_dtype(device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16

    return device, dtype


def build_inference_model(config_name, ckpt_path, training_config_path="training"):
    if not osp.isabs(training_config_path):
        training_config_path = osp.abspath(training_config_path)

    with initialize_config_dir(version_base=None, config_dir=training_config_path):
        config = compose(config_name=config_name)

    return load_model(config, ckpt_path)


def load_global_map(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded global map from {pkl_path}")
    print(f"   {len(data['global_pts'])} submaps | {len(data['T_global'])} transforms")
    return data["global_pts"], data["T_global"], data["ids"]


def load_unposed_references(obj_ids, pkl_pattern, ref_base):
    references = defaultdict(list)

    for obj_id in obj_ids:
        ref_anchor_pcs, _, sel_ids = load_global_map(pkl_pattern.format(obj_id=obj_id))
        references[obj_id].append(
            load_unposed_reference_object(
                ref_dir=osp.join(ref_base, f"obj_{obj_id:06d}"),
                ref_anchor_pcs=ref_anchor_pcs,
                sel_ids=sel_ids,
            )
        )

    return references


def load_scene_unposed_references(pkl_root, ref_base, object_dir_pattern="{scene_id}/obj_{obj_id:06d}"):
    references = defaultdict(lambda: defaultdict(list))

    for scene_id in sorted(os.listdir(pkl_root)):
        scene_dir = osp.join(pkl_root, scene_id)
        if not osp.isdir(scene_dir):
            continue

        for pkl_file in sorted(os.listdir(scene_dir)):
            if not pkl_file.endswith(".pkl"):
                continue

            obj_id = int(osp.splitext(pkl_file)[0])
            ref_anchor_pcs, _, sel_ids = load_global_map(osp.join(scene_dir, pkl_file))
            ref_dir = osp.join(ref_base, object_dir_pattern.format(scene_id=scene_id, obj_id=obj_id))
            references[scene_id][obj_id].append(
                load_unposed_reference_object(
                    ref_dir=ref_dir,
                    ref_anchor_pcs=ref_anchor_pcs,
                    sel_ids=sel_ids,
                )
            )
            print(f"Loaded Scene {scene_id}, Object {obj_id:02d}")

    print(f"Total scenes loaded: {len(references)}")
    return references


def load_unposed_reference_object(ref_dir, ref_anchor_pcs, sel_ids):
    camera_path = osp.join(ref_dir, "camera.npz")
    npz = np.load(camera_path, allow_pickle=True)
    intrinsics = npz["intrinsics"]
    extrinsics = npz["extrinsics"]

    ref_images, ref_depths, ref_masks = [], [], []
    for i in sel_ids:
        ref_image = cv2.imread(osp.join(ref_dir, f"rgb_{i:02d}.png"))
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_depth = cv2.imread(osp.join(ref_dir, f"depth_{i:02d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32)
        ref_mask = cv2.imread(osp.join(ref_dir, f"mask_{i:02d}.png"), cv2.IMREAD_UNCHANGED).astype(bool)

        ref_images.append(to_tensor(ref_image))
        ref_depths.append(ref_depth)
        ref_masks.append(ref_mask)

    ref_images = torch.stack(ref_images, dim=0)
    ref_masks = np.stack(ref_masks, axis=0)
    ref_depths = np.stack(ref_depths, axis=0)
    sel_ids = np.asarray(sel_ids, dtype=int)
    can_index_intrinsics = sel_ids.size > 0 and sel_ids.max() < len(intrinsics)
    can_index_extrinsics = sel_ids.size > 0 and sel_ids.max() < len(extrinsics)
    selected_intrinsics = intrinsics[sel_ids] if can_index_intrinsics else intrinsics
    selected_extrinsics = extrinsics[sel_ids] if can_index_extrinsics else extrinsics

    _, ref_anchor_pcs_first = unproject_depth_map_to_point_map(ref_depths, selected_extrinsics, selected_intrinsics)
    return {
        "images": ref_images,
        "masks": ref_masks,
        "depths": ref_depths,
        "ref_anchor_pcs": ref_anchor_pcs,
        "ref_anchor_pcs_first": ref_anchor_pcs_first[0],
    }


def compute_sim3_ref_to_anchor_with_conf(
    ref_pcs,
    ref_confs,
    ref_anchor_pcs,
    ref_ids,
    with_scaling=True,
    verbose=True,
):
    dst_all = np.concatenate([ref_anchor_pcs[aid] for aid in ref_ids], axis=0)
    src_all = np.concatenate(ref_pcs, axis=0)
    conf_all = np.concatenate(ref_confs, axis=0)

    assert src_all.shape[0] == dst_all.shape[0] == conf_all.shape[0], "Point number mismatch"

    try:
        scale, R, t = robust_umeyama(
            p_src=src_all,
            p_dst=dst_all,
            conf_src=conf_all,
            with_scaling=with_scaling,
        )
    except Exception as exc:
        print(f"Weighted Sim3 estimation failed: {exc}")
        return None, None, None, None

    T_pred2anchor = np.eye(4)
    T_pred2anchor[:3, :3] = scale * R
    T_pred2anchor[:3, 3] = t

    if verbose:
        print(f"Weighted Sim3 estimated | scale={scale:.4f}")

    return T_pred2anchor, scale, R, t


def estimate_reference_alignment(
    point_map,
    point_conf,
    depth_map,
    depth_conf,
    masks,
    ref_anchor_pcs,
    selected_ref_ids,
    K,
    fallback_ref_pc,
    use_conf=False,
    stride=2,
    z_min=1e-6,
    z_max=np.inf,
):
    if len(selected_ref_ids) == 0:
        return estimate_first_reference_alignment(point_map, point_conf, masks, fallback_ref_pc)

    ref_pcs, ref_confs, used_ref_ids = [], [], []
    for ref_id in selected_ref_ids:
        frame_idx = ref_id + 1
        pts_depth, (uu, vv) = backproject_depth_to_points(
            depth_map=depth_map[frame_idx],
            K=K,
            mask=masks[frame_idx],
            conf_map=(depth_conf[frame_idx] if use_conf else None),
            stride=stride,
            z_min=z_min,
            z_max=z_max,
        )
        if pts_depth.shape[0] < 10:
            print(f"Frame {frame_idx}: too few points, skip")
            continue

        ref_pcs.append(point_map[frame_idx][vv, uu])
        ref_confs.append(point_conf[frame_idx][vv, uu])
        used_ref_ids.append(ref_id)

    if not ref_pcs:
        return estimate_first_reference_alignment(point_map, point_conf, masks, fallback_ref_pc)

    _, scale, R, t = compute_sim3_ref_to_anchor_with_conf(
        ref_pcs=ref_pcs,
        ref_confs=ref_confs,
        ref_anchor_pcs=ref_anchor_pcs,
        ref_ids=used_ref_ids,
        with_scaling=True,
    )
    return scale, R, t


def estimate_first_reference_alignment(point_map, point_conf, masks, ref_anchor_pc):
    ref_idx = 1
    ref_mask = masks[ref_idx]
    src = point_map[ref_idx][ref_mask > 0]
    dst = ref_anchor_pc[ref_mask > 0]
    conf = point_conf[ref_idx][ref_mask > 0]
    return robust_umeyama(src, dst, conf, conf, with_scaling=True)


def estimate_query_pose(pred_query_pc, point_conf, query_mask, query_depth, query_camera, scale, R, t):
    pred_query_aligned = scale * (pred_query_pc @ R.T) + t
    pred_query_aligned = pred_query_aligned * query_mask[:, :, None]

    query_depth = query_depth * query_mask
    query_pc = depth_to_cam_points(query_depth, query_camera)[query_mask].reshape(-1, 3)
    pred_query_aligned = pred_query_aligned[query_mask].reshape(-1, 3)
    conf_query = point_conf[0][query_mask > 0]

    non_zero_mask = np.linalg.norm(query_pc, axis=1) > 0
    query_pc = query_pc[non_zero_mask]
    pred_query_aligned = pred_query_aligned[non_zero_mask]
    conf_query = conf_query[non_zero_mask]

    _, R_pose, t_pose = robust_umeyama(
        pred_query_aligned,
        query_pc,
        conf_query,
        conf_query,
        with_scaling=False,
    )

    pred_pose = np.eye(4)
    pred_pose[:3, :3] = R_pose
    pred_pose[:3, 3] = t_pose
    return pred_pose


def pose_to_bop_result(scene_id, image_id, obj_id, pred_pose, score=1.0, runtime=0.0):
    return {
        "scene_id": scene_id,
        "im_id": image_id,
        "obj_id": obj_id,
        "score": score,
        "R": " ".join(map(str, pred_pose[:3, :3].reshape(-1).tolist())),
        "t": " ".join(map(str, pred_pose[:3, 3].tolist())),
        "time": runtime,
    }


def get_auc(rec, max_val=0.1):
    if len(rec) == 0:
        return 0

    rec = np.sort(np.array(rec))
    n = len(rec)
    prec = np.arange(1, n + 1) / float(n)
    rec = rec.reshape(-1)
    prec = prec.reshape(-1)
    index = np.where(rec < max_val)[0]
    rec = rec[index]
    prec = prec[index]
    if len(rec) == 0:
        return 0

    mrec = np.array([0, *list(rec), max_val])
    mpre = np.array([0, *list(prec), prec[-1]])
    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])

    i = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_val


def evaluate_pose(metrics, obj_id, obj_model, obj_sym, pred_pose, query_gt_pose, store_auc=False):
    add_diam = get_diameter(obj_model["pts"])
    add = compute_add(obj_model["pts"], pred_pose, query_gt_pose)
    adds = compute_adds(obj_model["pts"], pred_pose, query_gt_pose) if obj_sym.shape[0] > 1 else add

    metrics.setdefault(obj_id, {"ADD-0.1d": [], "ADD(S)-0.1d": []})
    metrics[obj_id]["ADD(S)-0.1d"].append(float(adds <= add_diam * 0.1))
    metrics[obj_id]["ADD-0.1d"].append(float(add <= add_diam * 0.1))
    if store_auc:
        metrics[obj_id].setdefault("ADD-AUC", []).append(float(add))
        metrics[obj_id].setdefault("ADD(S)-AUC", []).append(float(adds))

    return add, adds


def save_pose_bbox_visualization(query_image_path, model_path, pred_pose, query_camera, output_path):
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

    model = trimesh.load(model_path)
    model_points = np.array(model.vertices)
    scale = np.max(model_points, axis=0) - np.min(model_points, axis=0)
    shift = np.mean(model_points, axis=0)
    bbox_3d = get_3d_bbox(scale, shift)

    transformed_bbox_3d = pred_pose[:3, :3] @ bbox_3d + pred_pose[:3, 3][:, np.newaxis]
    projected_bbox = calculate_2d_projections(transformed_bbox_3d, query_camera)
    draw_image_bbox = draw_3d_bbox(query_image, projected_bbox, color=(0, 255, 0))

    os.makedirs(osp.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(draw_image_bbox, cv2.COLOR_RGB2BGR))


def write_pose_results(csv_path, pose_results):
    os.makedirs(osp.dirname(csv_path), exist_ok=True)
    pd.DataFrame(pose_results).to_csv(csv_path, index=False)


def write_metrics_log(log_path, metrics):
    os.makedirs(osp.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_file:
        all_add_auc = []
        all_adds_auc = []
        for obj_id in sorted(metrics.keys()):
            add_vals = metrics[obj_id]["ADD-0.1d"]
            adds_vals = metrics[obj_id]["ADD(S)-0.1d"]
            add_mean = np.mean(add_vals) if add_vals else 0.0
            adds_mean = np.mean(adds_vals) if adds_vals else 0.0

            line = f"Object {obj_id:02d} - ADD-0.1d: {add_mean:.4f}, ADD(S)-0.1d: {adds_mean:.4f}"
            if "ADD-AUC" in metrics[obj_id]:
                add_auc = get_auc(metrics[obj_id]["ADD-AUC"], max_val=0.1 * 1e3)
                adds_auc = get_auc(metrics[obj_id]["ADD(S)-AUC"], max_val=0.1 * 1e3)
                all_add_auc.append(add_auc)
                all_adds_auc.append(adds_auc)
                line += f", ADD_AUC: {add_auc:.4f}, ADD(S)_AUC: {adds_auc:.4f}"
            print(line)
            log_file.write(line + "\n")

        all_add = [value for obj_metrics in metrics.values() for value in obj_metrics["ADD-0.1d"]]
        all_adds = [value for obj_metrics in metrics.values() for value in obj_metrics["ADD(S)-0.1d"]]

        total_add_line = f"Total ADD-0.1d: {np.mean(all_add):.4f}"
        total_adds_line = f"Total ADD(S)-0.1d: {np.mean(all_adds):.4f}"

        print(total_add_line)
        print(total_adds_line)
        log_file.write(total_add_line + "\n")
        log_file.write(total_adds_line + "\n")
        if all_add_auc:
            total_add_auc_line = f"Total ADD_AUC: {np.mean(all_add_auc):.4f}"
            total_adds_auc_line = f"Total ADD(S)_AUC: {np.mean(all_adds_auc):.4f}"
            print(total_add_auc_line)
            print(total_adds_auc_line)
            log_file.write(total_add_auc_line + "\n")
            log_file.write(total_adds_auc_line + "\n")
