import cv2
import numpy as np
from scipy.signal import find_peaks

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

    colorful_mask = ((sat >= sat_thresh) & (val >= val_thresh)).astype(np.uint8) * 255

    # sanity check: mask should not cover most of the image
    coverage = float(np.count_nonzero(colorful_mask)) / colorful_mask.size
    if coverage > max_coverage:
        raise RuntimeError(
            f"Arrow mask suspiciously large ({coverage:.1%}) — "
            f"check image for unexpected colorful regions."
        )

    return colorful_mask


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
def save_otsu_debug(img: np.ndarray, out_prefix) -> None:
    """
    Save diagnostic images showing Otsu thresholding on saturation.

    Outputs:
    - saturation grayscale image
    - binary saturation mask from Otsu
    - overlay of colorful pixels on original image
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Otsu threshold on saturation
    otsu_thresh, otsu_mask = cv2.threshold(
        sat.astype(np.uint8),
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # value threshold
    mu_v, std_v = float(val.mean()), float(val.std())
    val_thresh = float(np.clip(mu_v - std_v, 1, 254))

    # final colorful mask
    colorful_mask = (
        (sat >= otsu_thresh) &
        (val >= val_thresh)
    ).astype(np.uint8) * 255

    # overlay
    overlay = img.copy()
    overlay[colorful_mask == 255] = (0, 255, 255)

    # save outputs
    cv2.imwrite(str(out_prefix) + "_sat.png", sat)
    cv2.imwrite(str(out_prefix) + "_otsu_mask.png", otsu_mask)
    cv2.imwrite(str(out_prefix) + "_colorful_mask.png", colorful_mask)
    cv2.imwrite(str(out_prefix) + "_otsu_overlay.png", overlay)

    print(
        f"[OK] Otsu threshold = {otsu_thresh:.1f}, "
        f"coverage = {(colorful_mask > 0).mean():.1%}"
    )


def plot_otsu_histogram(img: np.ndarray, out_path) -> None:
    """
    Save a histogram of saturation values with Otsu threshold marked.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        otsu_thresh, _ = cv2.threshold(
            sat.astype(np.uint8),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_yscale("log")
        ax.set_xlim(0, 150)

        ax.hist(
            sat.ravel(),
            bins=256,
            range=(0, 255),
            color="steelblue",
            alpha=0.8,
        )

        ax.axvline(
            otsu_thresh,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"Otsu threshold = {otsu_thresh:.1f}",
        )

        ax.set_title("Saturation histogram with Otsu threshold")
        ax.set_xlabel("Saturation")
        ax.set_ylabel("Pixel count")

        ax.legend()

        fig.tight_layout()
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)

        print(f"[OK] Otsu histogram saved → {out_path}")

    except ImportError:
        print("[SKIP] matplotlib not available")