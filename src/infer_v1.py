import json
import os

import numpy as np
import yaml
from tqdm import tqdm

from .experiment_tracker import create_run_folder
from .features import depth_edges, edge_map, load_inputs, seg_boundary
from .proposals import suggest_floors, suggest_repeats_per_floor
from .scorer import search_best


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/v1_facades.yaml")
    parser.add_argument('--no_repeat', action='store_true', help='disable horizontal repeat reasoning')
    parser.add_argument('--no_split', action='store_true', help='disable vertical split reasoning')
    parser.add_argument('--no_renderer', action='store_true', help='disable feature-space reconstruction scoring')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    # Create experiment run folder for predictions
    run_dir, timestamp = create_run_folder(
        base_out=os.path.join(os.path.dirname(os.path.abspath(args.out)), 'experiments'),
        config_path=args.config,
    )
    tag = []
    if args.no_repeat:
        tag.append('noRepeat')
    if args.no_split:
        tag.append('noSplit')
    if args.no_renderer:
        tag.append('noRenderer')
    if tag:
        run_dir = run_dir + '_' + '_'.join(tag)
        os.makedirs(run_dir, exist_ok=True)
    print(f'\n🧪 Starting new experiment run at {run_dir}\n')
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.out):
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

        if args.no_split:
            floors_list = [(1, [(0, height)])]
        else:
            floors_list = suggest_floors(
                gray,
                depth=depth,
                mask=mask,
                fmin=cfg["search"]["floors"][0],
                fmax=cfg["search"]["floors"][1],
            )
        if not floors_list:
            floors_list = [(3, [(0, height // 3), (height // 3, 2 * height // 3), (2 * height // 3, height)])]

        if args.no_repeat:
            rep = 1
            repeat_min = repeat_max = 1
        else:
            repeat_min = cfg["search"]["repeats"][0]
            repeat_max = cfg["search"]["repeats"][1]
            rep = suggest_repeats_per_floor(
                img,
                floors_list[0][1],
                rmin=repeat_min,
                rmax=repeat_max,
            )

        if args.no_renderer:
            feature_map = np.zeros_like(feature_map)

        topk = search_best(
            height,
            width,
            floors_list,
            gray,
            feature_map,
            cfg["loss"]["lambda_rec"],
            cfg["loss"]["lambda_mdl"],
            cfg["loss"]["mdl_beta_depth"],
            rmin=repeat_min,
            rmax=repeat_max,
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

    print(f'\n✅ Inference complete. Predictions saved to {args.out}\n')
    print(f'🗂️  Run folder registered at: {run_dir}')


if __name__ == "__main__":
    main()
