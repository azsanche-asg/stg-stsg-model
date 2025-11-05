import json
import os

import numpy as np
import yaml
from tqdm import tqdm

from .features import depth_edges, edge_map, load_inputs, seg_boundary
from .proposals import suggest_floors, suggest_repeats_per_floor
from .scorer import search_best


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/v1_facades.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)

    files = sorted(
        [fname for fname in os.listdir(args.images) if fname.endswith(".png") and "_mask" not in fname and "_depth" not in fname]
    )

    for fname in tqdm(files, desc="ST-SG v1"):
        stem = os.path.splitext(fname)[0]
        img, gray, depth, mask = load_inputs(args.images, stem, cfg["io"]["aux_suffix"])
        height, width = gray.shape[:2]

        feature_map = edge_map(gray)
        if depth is not None and cfg["features"].get("use_depth_edges", True):
            depth_edge_map = depth_edges(depth)
            if depth_edge_map is not None:
                feature_map = np.maximum(feature_map, depth_edge_map)
        if mask is not None and cfg["features"].get("use_seg_masks", True):
            mask_boundary = seg_boundary(mask)
            if mask_boundary is not None:
                feature_map = np.maximum(feature_map, mask_boundary)

        floors_list = suggest_floors(
            gray,
            depth=depth,
            mask=mask,
            fmin=cfg["search"]["floors"][0],
            fmax=cfg["search"]["floors"][1],
        )
        if not floors_list:
            floors_list = [(3, [(0, height // 3), (height // 3, 2 * height // 3), (2 * height // 3, height)])]

        _ = suggest_repeats_per_floor(
            img,
            floors_list[0][1],
            rmin=cfg["search"]["repeats"][0],
            rmax=cfg["search"]["repeats"][1],
        )

        topk = search_best(
            height,
            width,
            floors_list,
            gray,
            feature_map,
            cfg["loss"]["lambda_rec"],
            cfg["loss"]["lambda_mdl"],
            cfg["loss"]["mdl_beta_depth"],
            rmin=cfg["search"]["repeats"][0],
            rmax=cfg["search"]["repeats"][1],
            beam_width=cfg["search"]["beam_width"],
        )

        topk.sort(key=lambda item: item[0])
        _, floors_best, repeats_best = topk[0]

        grammar = {
            "rules": [f"Split_y_{idx}" for idx in range(floors_best)] + [f"Repeat_x_{repeats_best}"],
            "repeats": [int(floors_best), int(repeats_best)],
            "depth": 2,
            "persist_ids": [],
            "motion": [],
        }
        with open(os.path.join(args.out, f"{stem}_pred.json"), "w", encoding="utf-8") as handle:
            json.dump(grammar, handle, indent=2)


if __name__ == "__main__":
    main()
