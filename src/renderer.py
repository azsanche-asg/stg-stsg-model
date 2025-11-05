import cv2
import numpy as np


def soft_bands_mask(height, width, floors):
    band_height = max(1, height // max(1, floors))
    mask = np.zeros((height, width), np.float32)
    for idx in range(1, floors):
        y = idx * band_height
        y0, y1 = max(0, y - 2), min(height, y + 2)
        mask[y0:y1, :] = 1.0
    return cv2.GaussianBlur(mask, (5, 5), 0)


def soft_grid_mask(height, width, floors, repeats):
    band_height = max(1, height // max(1, floors))
    band_width = max(1, width // max(1, repeats))
    mask = np.zeros((height, width), np.float32)
    for idx in range(1, floors):
        y = idx * band_height
        y0, y1 = max(0, y - 2), min(height, y + 2)
        mask[y0:y1, :] = 1.0
    for idx in range(1, repeats):
        x = idx * band_width
        x0, x1 = max(0, x - 2), min(width, x + 2)
        mask[:, x0:x1] = np.maximum(mask[:, x0:x1], 1.0)
    return cv2.GaussianBlur(mask, (5, 5), 0)


def recon_loss(feature_map, mask_soft):
    diff = (feature_map - mask_soft) ** 2
    return float(diff.mean())
