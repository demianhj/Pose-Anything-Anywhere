import argparse
import os
import os.path as osp
import random
import sys

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

sys.path.append(os.getcwd())
from inference_utils.datasets import BOP_Dataset
from inference_utils.multi_view import (
    build_inference_model,
    estimate_query_pose,
    estimate_reference_alignment,
    evaluate_pose,
    load_unposed_references,
    pose_to_bop_result,
    resolve_device_and_dtype,
    save_pose_bbox_visualization,
    write_metrics_log,
    write_pose_results,
)
from inference_utils.multi_view_sfm import estimate_matches, save_match_plot, select_top_refs
from inference_utils.oryon_utils.misc import format_sym_set
from inference_utils.utils import crop_input, estimate_intrinsics_from_pointmap


def main():
    parser = argparse.ArgumentParser(description="Run LM multi-view unposed inference.")
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="scripts/configs/lmo_mv_rgb.yaml",
        help="Path to the LM RGB-D inference YAML config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional override for the Hydra model config name.",
    )
    args = parser.parse_args()

    with open(args.dataset_config, "r") as f:
        raw_cfg = yaml.safe_load(f)

    ref_cfg = raw_cfg.get("reference", {})
    inf_cfg = raw_cfg.get("inference", {})
    out_cfg = raw_cfg.get("outputs", {})

    dataset_cfg = OmegaConf.create(raw_cfg)
    dataset = BOP_Dataset(dataset_cfg)
    obj_models, _, obj_symms = dataset.get_object_info()

    dataset_base = raw_cfg["dataset"]["root"]
    model_dir = osp.join(dataset_base, "models")
    output_base = raw_cfg["output_base"]
    os.makedirs(output_base, exist_ok=True)

    obj_ids = raw_cfg["dataset"]["obj_ids"]
    target_size = tuple(raw_cfg["dataset"].get("target_size", [518, 518]))
    top_k = inf_cfg.get("top_k", 2)
    min_match_threshold = inf_cfg.get("min_match_threshold", 500)
    seed = inf_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    model = build_inference_model(
        config_name=args.config or raw_cfg.get("model_config", "pany_model"),
        ckpt_path=raw_cfg.get("ckpt_path", "ckpt/pany.ckpt"),
        training_config_path=raw_cfg.get("training_config_path", "training"),
    )
    device, dtype = resolve_device_and_dtype(raw_cfg.get("device", "auto"))
    model = model.to(device).eval()

    references = load_unposed_references(
        obj_ids=obj_ids,
        pkl_pattern=ref_cfg.get("pkl_pattern", "data/ref_data/lm/pkl/{obj_id}.pkl"),
        ref_base=ref_cfg.get("base_dir", "data/multi-view-references/lm"),
    )

    match_cfg = {
        "inference": {
            "n_sample_point": inf_cfg.get("n_sample_point", 2048),
            "min_valid_matches": inf_cfg.get("min_valid_matches", 6),
        }
    }

    pose_results = []
    metrics = {}

    for batch in dataset:
        obj_id = int(batch["obj_id"])
        image_id = int(batch["image_id"])
        scene_id = int(batch["scene_id"])
        query_gt_pose = batch["query_gt_pose"]
        query_camera = batch["query_intri"]

        query_image, query_mask, query_depth, updated_cam_K, _ = crop_input(
            batch["query_image"],
            batch["query_mask"],
            batch["query_depth"],
            query_camera,
            target_size=target_size,
        )
        query_mask = query_mask.astype(bool)

        ref_data = references[obj_id][0]
        ref_images = ref_data["images"]
        ref_masks = ref_data["masks"]
        ref_anchor_pcs = ref_data["ref_anchor_pcs"]
        ref_anchor_pcs_first = ref_data["ref_anchor_pcs_first"]

        ref_images = ref_images * torch.from_numpy(ref_masks[:, None, :, :]).float()
        images_ordered = torch.cat([query_image.unsqueeze(0), ref_images], dim=0)
        masks = np.concatenate([query_mask[None], ref_masks], axis=0)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device == "cuda", dtype=dtype):
                images = images_ordered[None].to(device)
                aggregated_tokens_list, ps_idx = model.aggregator(images)

            point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
            point_map = point_map.squeeze(0).cpu().numpy()
            point_conf = point_conf.squeeze(0).cpu().numpy()

            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)
            depth_map = depth_map.squeeze(0).cpu().numpy()
            depth_conf = depth_conf.squeeze(0).cpu().numpy()

            K, _ = estimate_intrinsics_from_pointmap(
                point_map[0],
                masks[0],
                stride=inf_cfg.get("intrinsics_stride", 8),
                allow_skew=inf_cfg.get("allow_intrinsics_skew", False),
            )

            filtered_query_xys, filtered_pred_xys, num_matches = estimate_matches(
                model,
                aggregated_tokens_list,
                images,
                ps_idx,
                point_map,
                masks,
                match_cfg,
                device,
                context={"scene_id": scene_id, "obj_id": obj_id},
            )

        obj_name = f"obj_{obj_id:06d}"
        output_dir = osp.join(output_base, obj_name)
        match_path = osp.join(
            output_dir,
            out_cfg.get("match_pattern", "match_vis_{scene_id:06d}_{image_id:06d}_{obj_id:06d}.png").format(
                scene_id=scene_id,
                image_id=image_id,
                obj_id=obj_id,
            ),
        )
        save_match_plot(images_ordered, filtered_query_xys, filtered_pred_xys, match_path)

        ref_order = [-1] + list(range(len(ref_images)))
        selected = select_top_refs(num_matches, ref_order, top_k, min_match_threshold)
        if selected:
            selected_ref_ids, top_ref_idx = selected
            print(
                f"Frame {image_id}_{obj_id}: top-{top_k} refs = "
                f"{selected_ref_ids} counts={np.array(num_matches)[top_ref_idx]}"
            )
        else:
            selected_ref_ids = []
            print(f"Frame {image_id}_{obj_id}: no valid refs above {min_match_threshold} matches; using fallback ref")

        scale, R, t = estimate_reference_alignment(
            point_map=point_map,
            point_conf=point_conf,
            depth_map=depth_map,
            depth_conf=depth_conf,
            masks=masks,
            ref_anchor_pcs=ref_anchor_pcs,
            selected_ref_ids=selected_ref_ids,
            K=K,
            fallback_ref_pc=ref_anchor_pcs_first,
            use_conf=inf_cfg.get("use_conf", False),
            stride=inf_cfg.get("depth_stride", 2),
            z_min=inf_cfg.get("z_min", 1e-6),
            z_max=inf_cfg.get("z_max") if inf_cfg.get("z_max") is not None else np.inf,
        )
        if scale is None:
            print(f"Frame {image_id}_{obj_id}: reference alignment failed, skipping")
            continue

        pred_pose = estimate_query_pose(
            pred_query_pc=point_map[0],
            point_conf=point_conf,
            query_mask=query_mask,
            query_depth=query_depth,
            query_camera=updated_cam_K,
            scale=scale,
            R=R,
            t=t,
        )
        pose_results.append(pose_to_bop_result(scene_id, image_id, obj_id, pred_pose))

        obj_model = obj_models[obj_id]
        obj_sym = format_sym_set(obj_symms[obj_id])
        add, adds = evaluate_pose(metrics, obj_id, obj_model, obj_sym, pred_pose, query_gt_pose)
        print(f"Frame {image_id} - Obj {obj_id} - ADD: {add:.2f} mm, ADD-S: {adds:.2f} mm")

        save_pose_bbox_visualization(
            query_image_path=batch["query_image_path"],
            model_path=osp.join(model_dir, f"{obj_name}.ply"),
            pred_pose=pred_pose,
            query_camera=query_camera,
            output_path=osp.join(
                output_dir,
                out_cfg.get("pose_vis_pattern", "{scene_id:06d}_{image_id:06d}_{obj_id:06d}.png").format(
                    scene_id=scene_id,
                    image_id=image_id,
                    obj_id=obj_id,
                ),
            ),
        )

    write_pose_results(osp.join(output_base, out_cfg.get("pose_results_file", "pose_results.csv")), pose_results)
    write_metrics_log(osp.join(output_base, out_cfg.get("metrics_log_file", "log_posed.txt")), metrics)


if __name__ == "__main__":
    main()
