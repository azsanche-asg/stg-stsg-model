import argparse
import json
import os

import cv2
from tqdm import tqdm

from src.induce import induce_stsg_from_image
from src.utils import stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Folder with facade PNGs (from procedural generator)")
    parser.add_argument("--out", required=True, help="Folder where _pred.json will be written")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    files = sorted(fname for fname in os.listdir(args.images) if fname.endswith(".png"))
    for fname in tqdm(files, desc="ST-SG Inference"):
        if "_mask" in fname or "_depth" in fname:
            continue
        image_path = os.path.join(args.images, fname)
        img = cv2.imread(image_path)
        grammar = induce_stsg_from_image(img)
        out_name = f"{stem(fname)}_pred.json"
        out_path = os.path.join(args.out, out_name)
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(grammar, handle, indent=2)


if __name__ == "__main__":
    main()
