import cv2
import numpy as np
from scipy.signal import find_peaks


def find_hue_clusters(hue_hist: np.ndarray, n_clusters: int = 2):
    kernel = np.ones(5) / 5
    smoothed = np.convolve(hue_hist, kernel, mode="same")
    peaks, _ = find_peaks(smoothed, prominence=smoothed.max() * 0.05)
    if len(peaks) == 0:
        raise RuntimeError("No hue peaks found — image may be entirely grayscale.")
    top = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:n_clusters]
    return sorted(top)


def hue_window_for_cluster(hue_values: np.ndarray, center_bin: int,
                            search_radius: int = 30) -> tuple[int, int]:
    dist = np.abs(hue_values.astype(int) - center_bin)
    dist = np.minimum(dist, 180 - dist)
    nearby = hue_values[dist <= search_radius]

    if len(nearby) < 10:
        return center_bin - search_radius // 2, center_bin + search_radius // 2

    mean_hue = float(np.mean(nearby))
    std_hue = float(np.std(nearby))
    minimum_half_width = 8
    half = max(std_hue * 1.5, minimum_half_width)
    lo = int(round(mean_hue - half)) % 180
    hi = int(round(mean_hue + half)) % 180
    return lo, hi


def hue_mask(hsv: np.ndarray, lo: int, hi: int,
             sat_thresh: float, val_thresh: float) -> np.ndarray:
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    sat_ok = s >= sat_thresh
    val_ok = v >= val_thresh

    if lo <= hi:
        hue_ok = (h >= lo) & (h <= hi)
    else:
        hue_ok = (h >= lo) | (h <= hi)

    return (hue_ok & sat_ok & val_ok).astype(np.uint8) * 255


def build_arrow_mask(img: np.ndarray,
                     n_arrows: int = 2,
                     sat_k: float = 3.0,
                     morph_close_px: int = 3) -> np.ndarray:
    """
    Takes a cropped BGR image array.
    Returns a uint8 mask (255 = arrow pixel).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    mu_s, std_s = sat.mean(), sat.std()
    sat_thresh = float(np.clip(mu_s + sat_k * std_s, 1, 254))

    mu_v, std_v = val.mean(), val.std()
    val_thresh = float(np.clip(mu_v - std_v, 1, 254))

    colorful_mask = (sat >= sat_thresh) & (val >= val_thresh)
    hue_vals = hsv[:, :, 0][colorful_mask]

    if hue_vals.size < n_arrows:
        raise RuntimeError("Too few colourful pixels found; check the image.")

    hue_hist, _ = np.histogram(hue_vals, bins=180, range=(0, 180))
    cluster_bins = find_hue_clusters(hue_hist, n_clusters=n_arrows)

    combined = np.zeros(img.shape[:2], dtype=np.uint8)
    for cb in cluster_bins:
        lo, hi = hue_window_for_cluster(hue_vals, cb)
        mask_i = hue_mask(hsv, lo, hi, sat_thresh, val_thresh)
        k = morph_close_px
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_CLOSE, kernel)
        combined = cv2.bitwise_or(combined, mask_i)

    return combined


def fill_arrow_lines(mask: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """
    Fill entire rows/columns that are predominantly arrow pixels.
    Uses fraction of masked pixels relative to full image dimension.
    """
    filled = mask.copy()

    for y in range(mask.shape[0]):
        row = mask[y, :]
        masked_cols = np.where(row == 255)[0]
        if len(masked_cols) == 0:
            continue
        frac = len(masked_cols) / mask.shape[1]
        if frac > threshold:
            filled[y, :] = 255

    for x in range(mask.shape[1]):
        col = mask[:, x]
        masked_rows = np.where(col == 255)[0]
        if len(masked_rows) == 0:
            continue
        frac = len(masked_rows) / mask.shape[0]
        if frac > threshold:
            filled[:, x] = 255

    return filled

