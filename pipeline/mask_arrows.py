import cv2
import numpy as np
from scipy.signal import find_peaks

def find_hue_clusters(hue_hist: np.ndarray, n_clusters: int = 2):
    """Find the most prominent hue peaks in a histogram."""
    kernel = np.ones(5) / 5
    smoothed = np.convolve(hue_hist, kernel, mode="same")
    peaks, _ = find_peaks(smoothed, prominence=smoothed.max() * 0.05)
    if len(peaks) == 0:
        raise RuntimeError("No hue peaks found — image may be entirely grayscale.")
    top = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:n_clusters]
    return sorted(top)


def hue_window_for_cluster(hue_values: np.ndarray, center_bin: int,
                            search_radius: int = 30) -> tuple[int, int]:
    """Estimate a hue range around a detected hue cluster center."""
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
    """Create a binary mask for pixels inside a hue range and above S/V thresholds."""
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
    Build a binary mask for likely arrow pixels in a cropped BGR image.

    The mask is based on dominant hue clusters among colorful pixels, then cleaned
    up with a small morphological close.
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
        # k = morph_close_px
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        # mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_CLOSE, kernel)
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


def plot_hue_histogram(img: np.ndarray,
                       out_path,
                       n_arrows: int = 2,
                       sat_k: float = 3.0) -> None:
    """
    Save a hue histogram for the colorful pixels in the image, marking detected peaks.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        mu_s, std_s = sat.mean(), sat.std()
        sat_thresh = float(np.clip(mu_s + sat_k * std_s, 1, 254))

        mu_v, std_v = val.mean(), val.std()
        val_thresh = float(np.clip(mu_v - std_v, 1, 254))

        colorful_mask = (sat >= sat_thresh) & (val >= val_thresh)
        hue_vals = hsv[:, :, 0][colorful_mask]

        if hue_vals.size == 0:
            raise RuntimeError("No colorful pixels found for histogram plot.")

        hue_hist, _ = np.histogram(hue_vals, bins=180, range=(0, 180))
        cluster_bins = find_hue_clusters(hue_hist, n_clusters=n_arrows)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(180), hue_hist, color="steelblue", width=1.0, edgecolor="none")
        for cb in cluster_bins:
            lo, hi = hue_window_for_cluster(hue_vals, cb)
            ax.axvline(cb, color="crimson", linewidth=2, linestyle="--", label=f"peak {cb}")
            ax.axvspan(lo, hi, color="orange", alpha=0.2)

        ax.set_title("Hue histogram of colorful pixels")
        ax.set_xlabel("Hue bin")
        ax.set_ylabel("Pixel count")
        ax.set_xlim(0, 179)
        ax.legend(loc="upper right")
        fig.tight_layout()

        out_path = str(out_path)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[OK] Hue histogram saved to: {out_path}")
    except ImportError:
        print("[SKIP] matplotlib not available — hue histogram skipped")


def plot_hue_scatter(img: np.ndarray,
                     out_path,
                     max_points: int = 20000,
                     sat_k: float = 3.0) -> None:
    """
    Save a hue-vs-saturation scatter plot for colorful pixels.
    Useful for seeing how tightly the arrow colors cluster.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        mu_s, std_s = sat.mean(), sat.std()
        sat_thresh = float(np.clip(mu_s + sat_k * std_s, 1, 254))

        mu_v, std_v = val.mean(), val.std()
        val_thresh = float(np.clip(mu_v - std_v, 1, 254))

        colorful_mask = (sat >= sat_thresh) & (val >= val_thresh)
        h = hsv[:, :, 0][colorful_mask].ravel()
        s = hsv[:, :, 1][colorful_mask].ravel()

        if h.size == 0:
            raise RuntimeError("No colorful pixels found for scatter plot.")

        if h.size > max_points:
            idx = np.random.choice(h.size, size=max_points, replace=False)
            h = h[idx]
            s = s[idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(h, s, s=4, alpha=0.25, color="darkgreen", edgecolors="none")
        ax.set_title("Hue vs saturation for colorful pixels")
        ax.set_xlabel("Hue")
        ax.set_ylabel("Saturation")
        ax.set_xlim(0, 179)
        ax.set_ylim(0, 255)
        fig.tight_layout()

        out_path = str(out_path)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"[OK] Hue scatter saved to: {out_path}")
    except ImportError:
        print("[SKIP] matplotlib not available — hue scatter skipped")



