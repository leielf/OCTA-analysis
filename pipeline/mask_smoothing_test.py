from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

from mask_utils import (
    get_cross_line_masks,
    rebuild_cross_masks,
)
from process_images import *

# Adjust these if needed
INPUT_IMAGE = Path("medical_images/chlorochin_vfn/1/ODHR_choriocapillaris.JPG")
OUTPUT_DIR = Path("output_test_mask")


def detect_blue_green_masks(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Temporary test detector:
    - green mask = horizontal line
    - blue mask = vertical line

    Tune the HSV ranges if your image colors differ.
    """
    # hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    #
    # # Green range
    # green_lower = np.array([35, 40, 40], dtype=np.uint8)
    # green_upper = np.array([90, 255, 255], dtype=np.uint8)
    #
    # # Blue range
    # blue_lower = np.array([90, 40, 40], dtype=np.uint8)
    # blue_upper = np.array([140, 255, 255], dtype=np.uint8)
    #
    # green_mask = cv2.inRange(hsv, green_lower, green_upper)
    # blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40, 80, 80), (80, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    return green_mask, blue_mask

    return green_mask, blue_mask


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def main() -> None:
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = INPUT_IMAGE

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    red_crop = crop_rightmost_red_frame(image_bgr)

    green_raw, blue_raw = get_cross_line_masks(red_crop)

    # Rebuild clean lines from the raw masks
    green_smooth, blue_smooth = rebuild_cross_masks(
        green_raw,
        blue_raw
    )

    annotation_raw = cv2.bitwise_or(green_raw, blue_raw)
    annotation_smooth = cv2.bitwise_or(green_smooth, blue_smooth)
    final_mask = cv2.bitwise_not(annotation_smooth)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_mask(OUTPUT_DIR / "green_raw.png", green_raw)
    save_mask(OUTPUT_DIR / "blue_raw.png", blue_raw)
    save_mask(OUTPUT_DIR / "green_smooth.png", green_smooth)
    save_mask(OUTPUT_DIR / "blue_smooth.png", blue_smooth)
    save_mask(OUTPUT_DIR / "annotation_raw.png", annotation_raw)
    save_mask(OUTPUT_DIR / "annotation_smooth.png", annotation_smooth)
    save_mask(OUTPUT_DIR / "final_mask.png", final_mask)

    overlay = red_crop.copy()
    overlay[annotation_smooth > 0] = (0, 0, 255)
    save_mask(OUTPUT_DIR / "overlay.png", overlay)

    print(f"[OK] Saved results to: {OUTPUT_DIR}")
    print(f"[OK] Tested image: {image_path}")


if __name__ == "__main__":
    main()
