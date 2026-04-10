# pipeline/mask_utils.py
from __future__ import annotations

import cv2
import numpy as np


def get_cross_line_masks(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40, 80, 80), (80, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    return green_mask, blue_mask


# ... existing code ...
def rebuild_horizontal_line(mask: np.ndarray) -> np.ndarray:
    """
    Rebuild a horizontal line using the raw mask extent.
    Thickness is computed from the detected vertical spread of the line.
    """
    out = np.zeros_like(mask)

    rows = np.where(mask.any(axis=1))[0]
    if len(rows) == 0:
        return out

    y_top = int(rows.min())
    y_bottom = int(rows.max())
    line_thickness = y_bottom - y_top + 1

    xs_all = []
    for y in rows:
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        xs_all.append(xs)

    if not xs_all:
        return out

    xs_all = np.concatenate(xs_all)
    x_left = int(xs_all.min())
    x_right = int(xs_all.max())

    out[y_top:y_bottom + 1, x_left:x_right + 1] = 255
    return out


def rebuild_vertical_line(mask: np.ndarray) -> np.ndarray:
    """
    Rebuild a vertical line using the raw mask extent.
    Thickness is computed from the detected horizontal spread of the line.
    """
    out = np.zeros_like(mask)

    cols = np.where(mask.any(axis=0))[0]
    if len(cols) == 0:
        return out

    x_left = int(cols.min())
    x_right = int(cols.max())
    line_thickness = x_right - x_left + 1

    ys_all = []
    for x in cols:
        ys = np.where(mask[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        ys_all.append(ys)

    if not ys_all:
        return out

    ys_all = np.concatenate(ys_all)
    y_top = int(ys_all.min())
    y_bottom = int(ys_all.max())

    out[y_top:y_bottom + 1, x_left:x_right + 1] = 255
    return out


def rebuild_cross_masks(horizontal_mask: np.ndarray, vertical_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        rebuild_horizontal_line(horizontal_mask),
        rebuild_vertical_line(vertical_mask),
    )
# ... existing code ...