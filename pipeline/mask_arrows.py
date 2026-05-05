import cv2
import numpy as np
from scipy.signal import find_peaks


# ── Hue cluster detection ─────────────────────────────────────────────────────
def find_hue_clusters(hue_hist: np.ndarray, min_prominence: float = 0.1) -> list[int]:
    """
    Find prominent hue peaks in a histogram.

    Instead of hardcoding the number of clusters, all peaks above a relative
    prominence threshold are returned. This handles images with 1, 2, or more
    arrow colors automatically.
    """
    kernel = np.ones(5) / 5
    smoothed = np.convolve(hue_hist, kernel, mode="same")
    peaks, _ = find_peaks(
        smoothed,
        prominence=smoothed.max() * min_prominence,
        distance=10,  # minimum hue distance between clusters
    )
    if len(peaks) == 0:
        raise RuntimeError("No hue peaks found — image may be entirely grayscale.")
    return sorted(peaks.tolist())


# ── Hue window estimation ─────────────────────────────────────────────────────
def hue_window_for_cluster(hue_values: np.ndarray, center_bin: int,
                            search_radius: int = 30) -> tuple[int, int]:
    """
    Estimate a hue range around a detected hue cluster center.

    Uses percentile-based range instead of mean±std, which is more robust
    to outliers and does not assume a normal distribution.
    """
    dist = np.abs(hue_values.astype(int) - center_bin)
    dist = np.minimum(dist, 180 - dist)
    nearby = hue_values[dist <= search_radius]

    if len(nearby) < 10:
        return (center_bin - search_radius // 2) % 180, (center_bin + search_radius // 2) % 180

    lo = int(np.percentile(nearby, 2))
    hi = int(np.percentile(nearby, 98))

    # ensure minimum window width
    if hi - lo < 10:
        mid = (lo + hi) // 2
        lo, hi = mid - 5, mid + 5

    return lo % 180, hi % 180


# ── Hue mask ──────────────────────────────────────────────────────────────────

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
        # wrap-around case e.g. red hue near 0/180
        hue_ok = (h >= lo) | (h <= hi)

    return (hue_ok & sat_ok & val_ok).astype(np.uint8) * 255


# ── Arrow mask builder ────────────────────────────────────────────────────────

def build_arrow_mask(img: np.ndarray,
                     max_coverage: float = 0.20) -> np.ndarray:
    """
    Build a binary mask for likely arrow pixels in a cropped BGR image.

    Uses Otsu thresholding on the saturation channel to adaptively separate
    colorful pixels (arrows) from the grayscale background, then clusters
    their hues to identify arrow colors. The number of arrow colors is
    detected automatically rather than hardcoded.

    A sanity check raises RuntimeError if the mask covers more than
    max_coverage of the image, which indicates a likely detection failure.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # adaptive saturation threshold via Otsu
    otsu_thresh, _ = cv2.threshold(
        sat.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    sat_thresh = float(otsu_thresh)

    # value threshold: exclude very dark pixels
    mu_v, std_v = float(val.mean()), float(val.std())
    val_thresh = float(np.clip(mu_v - std_v, 1, 254))

    colorful_mask = (sat >= sat_thresh) & (val >= val_thresh)
    hue_vals = hsv[:, :, 0][colorful_mask]

    if hue_vals.size < 20:
        raise RuntimeError("Too few colourful pixels found; check the image.")

    hue_hist, _ = np.histogram(hue_vals, bins=180, range=(0, 180))
    cluster_bins = find_hue_clusters(hue_hist)

    combined = np.zeros(img.shape[:2], dtype=np.uint8)
    for cb in cluster_bins:
        lo, hi = hue_window_for_cluster(hue_vals, cb)
        mask_i = hue_mask(hsv, lo, hi, sat_thresh, val_thresh)
        combined = cv2.bitwise_or(combined, mask_i)

    # sanity check: mask should not cover most of the image
    coverage = float(np.count_nonzero(combined)) / combined.size
    if coverage > max_coverage:
        raise RuntimeError(
            f"Arrow mask suspiciously large ({coverage:.1%}) — "
            f"check image for unexpected colorful regions."
        )

    return combined


# ── Line filling ──────────────────────────────────────────────────────────────

def fill_arrow_lines(mask: np.ndarray, threshold: float = 0.4) -> np.ndarray:
    """
    Fill entire rows/columns that are predominantly arrow pixels.

    Arrows often span full rows or columns in OCTA report images. Any row or
    column where more than `threshold` fraction of pixels are masked is filled
    entirely.
    """
    filled = mask.copy()

    for y in range(mask.shape[0]):
        row = mask[y, :]
        masked_cols = np.where(row == 255)[0]
        if len(masked_cols) == 0:
            continue
        if len(masked_cols) / mask.shape[1] > threshold:
            filled[y, :] = 255

    for x in range(mask.shape[1]):
        col = mask[:, x]
        masked_rows = np.where(col == 255)[0]
        if len(masked_rows) == 0:
            continue
        if len(masked_rows) / mask.shape[0] > threshold:
            filled[:, x] = 255

    return filled


# ── Diagnostic plots ──────────────────────────────────────────────────────────

def _compute_colorful_pixels(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Shared helper: compute HSV, colorful mask, and thresholds for diagnostic plots."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    otsu_thresh, _ = cv2.threshold(
        sat.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    sat_thresh = float(otsu_thresh)

    mu_v, std_v = float(val.mean()), float(val.std())
    val_thresh = float(np.clip(mu_v - std_v, 1, 254))

    colorful_mask = (sat >= sat_thresh) & (val >= val_thresh)
    hue_vals = hsv[:, :, 0][colorful_mask]

    return hsv, hue_vals, sat_thresh, val_thresh


def plot_hue_histogram(img: np.ndarray, out_path) -> None:
    """
    Save a hue histogram for the colorful pixels in the image, marking detected peaks.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, hue_vals, _, _ = _compute_colorful_pixels(img)

        if hue_vals.size == 0:
            raise RuntimeError("No colorful pixels found for histogram plot.")

        hue_hist, _ = np.histogram(hue_vals, bins=180, range=(0, 180))
        cluster_bins = find_hue_clusters(hue_hist)

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
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"[OK] Hue histogram saved to: {out_path}")

    except ImportError:
        print("[SKIP] matplotlib not available — hue histogram skipped")


def plot_hue_scatter(img: np.ndarray, out_path, max_points: int = 20000) -> None:
    """
    Save a hue-vs-saturation scatter plot for colorful pixels.
    Useful for seeing how tightly the arrow colors cluster.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hsv, hue_vals, sat_thresh, _ = _compute_colorful_pixels(img)

        if hue_vals.size == 0:
            raise RuntimeError("No colorful pixels found for scatter plot.")

        h = hue_vals
        s = hsv[:, :, 1][(hsv[:, :, 1] >= sat_thresh)].ravel()

        # align lengths in case of shape mismatch
        min_len = min(len(h), len(s))
        h, s = h[:min_len], s[:min_len]

        if h.size > max_points:
            idx = np.random.choice(h.size, size=max_points, replace=False)
            h, s = h[idx], s[idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(h, s, s=4, alpha=0.25, color="darkgreen", edgecolors="none")
        ax.set_title("Hue vs saturation for colorful pixels")
        ax.set_xlabel("Hue")
        ax.set_ylabel("Saturation")
        ax.set_xlim(0, 179)
        ax.set_ylim(0, 255)
        fig.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"[OK] Hue scatter saved to: {out_path}")

    except ImportError:
        print("[SKIP] matplotlib not available — hue scatter skipped")