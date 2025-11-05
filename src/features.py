import cv2
import numpy as np
import os


def load_inputs(root_dir, stem, aux_suffix):
    image_path = os.path.join(root_dir, f"{stem}.png")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"image missing: {stem}.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    depth = None
    depth_path = os.path.join(root_dir, f"{stem}{aux_suffix['depth']}")
    if os.path.exists(depth_path):
        depth = np.load(depth_path).astype(np.float32)

    mask = None
    mask_path = os.path.join(root_dir, f"{stem}{aux_suffix['mask']}")
    if os.path.exists(mask_path):
        mask = np.load(mask_path).astype(np.int32)

    return img, img_gray, depth, mask


def edge_map(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    return edges.astype(np.float32) / 255.0


def depth_edges(depth):
    if depth is None:
        return None
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(dx * dx + dy * dy)
    magnitude = magnitude / (magnitude.max() + 1e-6)
    return magnitude


def seg_boundary(mask):
    if mask is None:
        return None
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], np.float32)
    boundary = cv2.filter2D((mask > 0).astype(np.float32), -1, kernel) != 0
    return boundary.astype(np.float32)


def patch_feats(img):
    feats = img.astype(np.float32) / 255.0
    for channel in range(3):
        mean = feats[:, :, channel].mean()
        std = feats[:, :, channel].std() + 1e-6
        feats[:, :, channel] = (feats[:, :, channel] - mean) / std
    return feats
