import argparse
import os
import sys

import torch

sys.path.append(os.getcwd())
from inference_utils.multi_view_sfm import build_model, load_yaml_config, run_object_sfm


def main():
    parser = argparse.ArgumentParser(description="Build LM reference SFM maps from multi-view templates.")
    parser.add_argument(
        "--sfm-config",
        type=str,
        default="scripts/configs/lm_reference_sfm.yaml",
        help="Path to the reference SFM YAML config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional override for the Hydra model config name.",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.sfm_config)
    model = build_model(cfg, model_config_override=args.config)

    device = cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device).eval()
    dtype = torch.float16
    if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
        dtype = torch.bfloat16

    for obj_id in cfg["reference"]["obj_ids"]:
        run_object_sfm(obj_id, model, cfg, device, dtype)


if __name__ == "__main__":
    main()
