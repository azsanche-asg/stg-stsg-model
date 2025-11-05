import cv2
import numpy as np

from .features import depth_edges, edge_map, seg_boundary


def suggest_floors(img_gray, depth=None, mask=None, fmin=2, fmax=6):
    height, width = img_gray.shape[:2]
    edges = edge_map(img_gray)
    if depth is not None:
        depth_edge_map = depth_edges(depth)
        if depth_edge_map is not None:
            edges = np.maximum(edges, depth_edge_map)
    if mask is not None:
        mask_boundary = seg_boundary(mask)
        if mask_boundary is not None:
            edges = np.maximum(edges, mask_boundary)

    row_strength = edges.mean(axis=1)
    autoc = np.correlate(row_strength - row_strength.mean(), row_strength - row_strength.mean(), mode="full")
    autoc = autoc[autoc.size // 2 :]

    candidates = []
    for floors in range(fmin, fmax + 1):
        band_height = max(1, height // floors)
        score = 0.0
        for idx in range(1, floors):
            y = idx * band_height
            y0, y1 = max(0, y - 1), min(height, y + 1)
            score += row_strength[y0 : y1 + 1].mean()
        candidates.append((floors, score))
    candidates.sort(key=lambda item: item[1], reverse=True)
    floors_list = [item[0] for item in candidates[: min(6, len(candidates))]]

    bands_list = []
    for floors in floors_list:
        band_height = max(1, height // floors)
        bands = [(idx * band_height, height if idx == floors - 1 else (idx + 1) * band_height) for idx in range(floors)]
        bands_list.append(bands)
    return list(zip(floors_list, bands_list))


def suggest_repeats_per_floor(img, bands, rmin=3, rmax=12):
    height, width = img.shape[:2]
    per_band_counts = []
    for y0, y1 in bands:
        band = img[y0:y1, :, :]
        col_proj = band.mean(axis=0).mean(axis=1) if band.ndim == 3 else band.mean(axis=0)
        col_proj = cv2.GaussianBlur(col_proj[:, None], (7, 1), 0)[:, 0]
        autoc = np.correlate(col_proj - col_proj.mean(), col_proj - col_proj.mean(), mode="full")
        autoc = autoc[autoc.size // 2 :]
        idx = np.argsort(autoc[1 : width // 2])[-5:] + 1
        counts = [max(rmin, min(width // max(period, 1), rmax)) for period in idx]
        per_band_counts.append(int(np.median(counts)) if len(counts) > 0 else rmin)
    repeat = int(np.median(per_band_counts)) if len(per_band_counts) > 0 else rmin
    return repeat
