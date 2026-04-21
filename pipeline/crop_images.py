import cv2
import numpy as np
import csv
from pathlib import Path
# ── Cropping ──────────────────────────────────────────────────────────────────

def find_red_frame_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0,   120, 70), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = image.shape[0] * image.shape[1]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 0.02 * img_area:
            boxes.append((x, y, w, h))
    return boxes


def crop_rightmost_red_frame(image: np.ndarray) -> np.ndarray:
    boxes = find_red_frame_boxes(image)
    if not boxes:
        raise ValueError("No red-framed panel found in image.")
    x, y, w, h = max(boxes, key=lambda b: b[0])
    return image[y:y + h, x:x + w]


def crop_to_cross_extent(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40,  80, 80), (80,  255, 255))
    blue_mask  = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))

    green_cols = np.where(green_mask.any(axis=0))[0]
    blue_rows  = np.where(blue_mask.any(axis=1))[0]

    if len(green_cols) == 0 or len(blue_rows) == 0:
        raise ValueError("Could not detect green/blue cross lines.")

    x_left, x_right = int(green_cols.min()), int(green_cols.max())
    y_top, y_bottom = int(blue_rows.min()),  int(blue_rows.max())

    inner_crop = crop[y_top:y_bottom, x_left:x_right].copy()
    if inner_crop.size == 0:
        raise ValueError(f"Empty crop: y={y_top}:{y_bottom}, x={x_left}:{x_right}")
    return inner_crop


# ── Helpers ───────────────────────────────────────────────────────────────────

def center_crop_np(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    y = (img.shape[0] - target_h) // 2
    x = (img.shape[1] - target_w) // 2
    return img[y:y+target_h, x:x+target_w]