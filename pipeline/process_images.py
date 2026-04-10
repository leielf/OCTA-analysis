import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import csv

# from mask_utils import *


INPUT_DIR        = Path("/medical_images/chlorochin_vfn")     # folder with patients/controls subfolders
OUTPUT_DIR       = Path("/medical_images/output")    # where results are saved
ALLOWED_PREFIXES = ("ODHR", "OSHR") # only process files starting with these
TARGET_SIZE      = (256, 256)
MIN_CROP_SIZE    = 100
MIN_VALID_RATIO  = 0.60

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_red_frame_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes (x, y, w, h) of all large red-framed regions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red wraps in HSV → two ranges
    mask1 = cv2.inRange(hsv, (0,   120, 70), (10,  255, 255))
    mask2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Close small gaps in the frame border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    img_area = image.shape[0] * image.shape[1]
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # Keep only reasonably large rectangles (at least 2% of image area)
        if area > 0.02 * img_area:
            boxes.append((x, y, w, h))

    return boxes


def crop_rightmost_red_frame(image: np.ndarray) -> np.ndarray:
    """Crop the red-framed panel that is furthest to the right."""
    boxes = find_red_frame_boxes(image)
    if not boxes:
        raise ValueError("No red-framed panel found in image.")
    box = max(boxes, key=lambda b: b[0])
    x, y, w, h = box
    return image[y:y + h, x:x + w]


def get_cross_line_masks(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (green_mask, blue_mask) for the cross overlay lines.
    Each is a binary uint8 mask (255 = line pixel).
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (40,  80, 80), (80,  255, 255))
    blue_mask  = cv2.inRange(hsv, (100, 80, 80), (130, 255, 255))
    return green_mask, blue_mask


def crop_to_cross_extent(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Uses the green (horizontal) and blue (vertical) cross lines to determine
    the crop rectangle, then returns:
      - the cropped image (cross lines masked out / set to black)
      - the corresponding annotation mask (255 = valid, 0 = cross line)
    """
    green_mask, blue_mask = get_cross_line_masks(crop)

    # ── Find extents of each line ──────────────────────────────────────────
    # Green horizontal line → left/right bounds (x-axis extent)
    green_cols = np.where(green_mask.any(axis=0))[0]
    # Blue vertical line  → top/bottom bounds (y-axis extent)
    blue_rows  = np.where(blue_mask.any(axis=1))[0]

    if len(green_cols) == 0 or len(blue_rows) == 0:
        raise ValueError("Could not detect green/blue cross lines.")

    x_left   = int(green_cols.min())
    x_right  = int(green_cols.max())
    y_top    = int(blue_rows.min())
    y_bottom = int(blue_rows.max())

    # ── Crop to cross extent ───────────────────────────────────────────────
    inner_crop = crop[y_top:y_bottom, x_left:x_right].copy()

    # final_mask = build_annotation_mask(inner_crop, tolerance=6, dilate_iterations=1)

    # ── Build annotation mask (remove cross lines from cropped region) ─────
    # # Re-detect lines in the cropped region
    green_c, blue_c = get_cross_line_masks(inner_crop)
    annotation = cv2.bitwise_or(green_c, blue_c)
    #
    # # Dilate slightly to catch anti-aliased edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    annotation = cv2.dilate(annotation, kernel, iterations=1)
    #
    final_mask = cv2.bitwise_not(annotation)  # 255 = valid pixel

    return inner_crop, final_mask


def prepare_gray_image(masked_image: np.ndarray, final_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert extracted OCTA crop to normalized grayscale.
    Returns:
      - gray_float: float32 image in range [0, 1]
      - resized_mask: uint8 mask resized to TARGET_SIZE
    """
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    # resized_mask = cv2.resize(final_mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
    resized_mask = final_mask

    gray = gray.astype(np.float32)
    valid = resized_mask > 0
    if not np.any(valid):
        raise ValueError("No valid pixels after masking.")

    valid_pixels = gray[valid]
    p1, p99 = np.percentile(valid_pixels, [1, 99])

    if p99 <= p1:
        raise ValueError("Invalid intensity range for normalization.")

    gray = np.clip((gray - p1) / (p99 - p1), 0, 1)
    gray[~valid] = 0.0

    return gray, resized_mask

def get_eye_label(stem: str) -> str:
    stem_upper = stem.upper()
    if stem_upper.startswith("OD"):
        return "OD"
    if stem_upper.startswith("OS"):
        return "OS"
    return "UNK"

def process_image(jpg_path: Path, out_dir: Path) -> dict:
    """Full extraction pipeline for one report image."""
    record = {
        "file_name": jpg_path.name,
        "stem": jpg_path.stem,
        "subject": jpg_path.parent.name,
        "eye": get_eye_label(jpg_path.stem),
        "status": "failed",
        "crop_width": "",
        "crop_height": "",
        "octa_path": "",
        "mask_path": "",
        "gray_path": "",
        "reason": "",
    }

    image = cv2.imread(str(jpg_path))
    if image is None:
        record["reason"] = "Cannot read image"
        print(f"  [SKIP] Cannot read {jpg_path.name}")
        return record

    stem = jpg_path.stem

    try:
        red_crop = crop_rightmost_red_frame(image)
    except ValueError as e:
        record["reason"] = str(e)
        print(f"  [FAIL] {jpg_path.name}: {e}")
        return record

    try:
        inner_crop, final_mask = crop_to_cross_extent(red_crop)
    except ValueError as e:
        record["reason"] = str(e)
        print(f"  [FAIL] {jpg_path.name}: {e}")
        return record

    record["crop_width"] = inner_crop.shape[1]
    record["crop_height"] = inner_crop.shape[0]

    masked_image = cv2.bitwise_and(inner_crop, inner_crop, mask=final_mask)

    try:
        gray_float, resized_mask = prepare_gray_image(masked_image, final_mask)
    except ValueError as e:
        record["reason"] = str(e)
        print(f"  [FAIL] {jpg_path.name}: {e}")
        return record

    gray_u8 = (gray_float * 255).astype(np.uint8)

    out_dir.mkdir(parents=True, exist_ok=True)
    octa_path = out_dir / f"{stem}_octa.png"
    mask_path = out_dir / f"{stem}_mask.png"
    gray_path = out_dir / f"{stem}_gray.png"

    cv2.imwrite(str(octa_path), masked_image)
    cv2.imwrite(str(mask_path), final_mask)
    cv2.imwrite(str(gray_path), gray_u8)

    record["status"] = "ok"
    record["reason"] = "ok"
    record["octa_path"] = str(octa_path)
    record["mask_path"] = str(mask_path)
    record["gray_path"] = str(gray_path)

    print(f"  [OK]  {jpg_path.name}")
    return record

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # input structure:
    #     patients/
    #       patient_001/
    #         ODHR_choriocapillaris.JPG
    #         OSHR_choriocapillaris.JPG

    patients_root = INPUT_DIR
    patient_dirs = sorted([d for d in patients_root.iterdir() if d.is_dir()])

    if not patient_dirs:
        print(f"No patient subfolders found in {patients_root}")
        return

    for subject_dir in patient_dirs:
    # for subject_dir in sorted(subject_dirs):   # e.g. patient_001
        print(f"  Patient: {subject_dir.name}")

        all_jpgs = (list(subject_dir.glob("*.jpg")) +
                    list(subject_dir.glob("*.JPG")) +
                    list(subject_dir.glob("*.jpeg")))

        # Filter to only ODHR and OSHR files
        jpg_files = [
            p for p in all_jpgs
            if any(p.stem.upper().startswith(prefix) for prefix in ALLOWED_PREFIXES)
        ]

        if not jpg_files:
            print("    [SKIP] No ODHR/OSHR files found.")

        out_dir = OUTPUT_DIR / subject_dir.name
        records = []

        for jpg_path in sorted(jpg_files):
            record = process_image(jpg_path, out_dir)
            records.append(record)

        if records:
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "extraction_summary.csv"
            fieldnames = [
                "file_name",
                "stem",
                "subject",
                "eye",
                "status",
                "crop_width",
                "crop_height",
                "octa_path",
                "mask_path",
                "gray_path",
                "reason",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
            print(f"\nSaved summary to: {csv_path}")

    print("\nDone.")













def validate_extraction(inner_crop: np.ndarray, final_mask: np.ndarray) -> tuple[bool, str, float]:
    """
    Basic quality control for extracted crops.
    Returns:
      - ok
      - reason
      - valid_ratio
    """
    h, w = inner_crop.shape[:2]
    if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
        return False, f"Crop too small: {w}x{h}", 0.0

    valid_ratio = float(np.count_nonzero(final_mask) / final_mask.size)
    if valid_ratio < MIN_VALID_RATIO:
        return False, f"Valid mask ratio too low: {valid_ratio:.3f}", valid_ratio

    return True, "ok", valid_ratio


if __name__ == "__main__":
    main()
