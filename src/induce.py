import cv2
import numpy as np


def _smooth(values, kernel_size=7):
    k = max(3, kernel_size | 1)
    return cv2.GaussianBlur(values.astype(np.float32), (k, 1), 0, borderType=cv2.BORDER_REFLECT101)


def _row_projection(gray):
    return gray.mean(axis=1)


def _col_projection(gray, y0, y1):
    return gray[y0:y1, :].mean(axis=0)


def _autocorr(signal):
    signal = signal - signal.mean()
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[corr.size // 2 :]
    if corr.max() > 1e-8:
        corr = corr / corr.max()
    return corr


def detect_floors(gray, max_floors=6):
    rows = _row_projection(gray)
    rows = _smooth(rows[:, None], kernel_size=15)[:, 0]
    autoc = _autocorr(rows)
    period_candidates = np.argsort(autoc[5:100])[-3:][::-1] + 5
    if len(period_candidates) == 0:
        return [(0, gray.shape[0])], 1
    period = int(np.median(period_candidates))
    floors = max(1, min(int(round(gray.shape[0] / max(period, 1))), max_floors))
    height = gray.shape[0]
    band_height = max(1, height // floors)
    bands = []
    for idx in range(floors):
        y0 = idx * band_height
        y1 = gray.shape[0] if idx == floors - 1 else (idx + 1) * band_height
        bands.append((y0, y1))
    return bands, floors


def detect_repeat_in_band(gray, band, search=30):
    y0, y1 = band
    col_proj = _col_projection(gray, y0, y1)
    autoc = _autocorr(col_proj)
    if autoc.size < 3:
        return 1
    candidates = np.argsort(autoc[1:search])[-5:] + 1
    width = gray.shape[1]
    counts = [max(1, min(width // max(period, 1), 12)) for period in candidates]
    count = int(np.median(counts))
    return count


def induce_stsg_from_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bands, num_floors = detect_floors(gray)
    per_floor_counts = [detect_repeat_in_band(gray, band) for band in bands]
    repeat = int(np.median(per_floor_counts)) if per_floor_counts else 1
    rules = []
    for idx in range(num_floors):
        rules.append(f"Split_y_{idx}")
        rules.append(f"Repeat_x_{repeat}")
    grammar = {
        "rules": rules,
        "repeats": [num_floors, repeat],
        "depth": 2,
        "persist_ids": [],
        "motion": [],
    }
    return grammar
