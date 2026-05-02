import cv2
import numpy as np
import csv
from pathlib import Path
from PIL import Image
from mask_arrows import build_arrow_mask, fill_arrow_lines
from crop_images import *

INPUT_DIR        = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/chlorochin_vfn")
OUTPUT_DIR       = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output_center_arrow_cropped")
ALLOWED_PREFIXES = ("ODHR", "OSHR")
MIN_CROP_SIZE    = 100
MIN_VALID_RATIO  = 0.60


def prepare_gray_image(image: np.ndarray, final_mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid = final_mask > 0
    if not np.any(valid):
        raise ValueError("No valid pixels after masking.")
    valid_pixels = gray[valid]
    p1, p99 = np.percentile(valid_pixels, [1, 99])
    if p99 <= p1:
        raise ValueError("Invalid intensity range for normalization.")
    gray = np.clip((gray - p1) / (p99 - p1), 0, 1)
    gray[~valid] = 0.0
    return gray


def get_eye_label(stem: str) -> str:
    s = stem.upper()
    if s.startswith("OD"): return "OD"
    if s.startswith("OS"): return "OS"
    return "UNK"


# ── Stage 1: crop to cross extent ─────────────────────────────────────────────

def stage1_crop(patient_dirs: list[Path]) -> list[dict]:
    """Crop all images to cross extent, return list of successful crop records."""
    crops = []
    for subject_dir in patient_dirs:
        print(f"\nPatient: {subject_dir.name}")
        all_jpgs = (list(subject_dir.glob("*.jpg")) +
                    list(subject_dir.glob("*.JPG")) +
                    list(subject_dir.glob("*.jpeg")))
        jpg_files = [p for p in all_jpgs
                     if any(p.stem.upper().startswith(px) for px in ALLOWED_PREFIXES)]
        if not jpg_files:
            print("  [SKIP] No ODHR/OSHR files found.")
            continue

        for jpg_path in sorted(jpg_files):
            image = cv2.imread(str(jpg_path))
            if image is None:
                print(f"  [SKIP] Cannot read {jpg_path.name}")
                continue
            try:
                red_crop    = crop_rightmost_red_frame(image)
                inner_crop  = crop_to_cross_extent(red_crop)
            except ValueError as e:
                print(f"  [FAIL] {jpg_path.name}: {e}")
                continue

            h, w = inner_crop.shape[:2]
            if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
                print(f"  [FAIL] {jpg_path.name}: crop too small {w}x{h}")
                continue

            crops.append({
                "jpg_path":    jpg_path,
                "subject":     subject_dir.name,
                "inner_crop":  inner_crop,
            })
            print(f"  [CROP] {jpg_path.name}  ({w}x{h})")

    return crops


# ── Stage 2: center-crop to uniform size ──────────────────────────────────────

def compute_common_center_crop(crops: list[dict]):
    """
    Compute the common centered square half-size across all images.
    """
    half_sizes_x = []
    half_sizes_y = []

    for c in crops:
        mask = c["final_mask"]
        cx = c["record"]["cx"]
        cy = c["record"]["cy"]
        h, w = mask.shape[:2]
        half_size_x = min(cx, w - cx - 1)
        half_size_y = min(cy, h - cy - 1)
        half_sizes_x.append(half_size_x)
        half_sizes_y.append(half_size_y)

    print(f"min x = { min(half_sizes_x)}, min y = { min(half_sizes_y)}")
    print(f"max x = { max(half_sizes_x)}, max y = { max(half_sizes_y)}")
    return min(half_sizes_x), min(half_sizes_y)




# ── Stage 2: build masks only ───────────────────────────────────────────────────

def stage2_build_masks(crops: list[dict]) -> tuple[list[dict], np.ndarray | None]:
    records_by_subject: dict[str, list[dict]] = {}

    for c in crops:
        jpg_path   = c["jpg_path"]
        subject    = c["subject"]
        inner_crop = c["inner_crop"]
        stem       = jpg_path.stem

        record = {
            "file_name":   jpg_path.name,
            "stem":        stem,
            "subject":     subject,
            "eye":         get_eye_label(stem),
            "status":      "failed",
            "crop_width":  inner_crop.shape[1],
            "crop_height": inner_crop.shape[0],
            "octa_path":   "",
            "mask_path":   "",
            "gray_path":   "",
            "reason":      "",
        }

        try:
            arrow_mask = build_arrow_mask(inner_crop)
            arrow_mask = fill_arrow_lines(arrow_mask)
            cx, cy = mask_density_center(arrow_mask)
            record.update({"cx": cx, "cy":cy})
        except RuntimeError as e:
            record["reason"] = str(e)
            print(f"  [FAIL] {jpg_path.name}: {e}")
            records_by_subject.setdefault(subject, []).append(record)
            continue

        final_mask = cv2.bitwise_not(arrow_mask)

        valid_ratio = float(np.count_nonzero(final_mask) / final_mask.size)
        if valid_ratio < MIN_VALID_RATIO:
            record["reason"] = f"Valid ratio too low: {valid_ratio:.3f}"
            print(f"  [FAIL] {jpg_path.name}: valid ratio too low ({valid_ratio:.3f})")
            records_by_subject.setdefault(subject, []).append(record)
            continue

        record["status"] = "ok"
        record["reason"] = "ok"
        record["mask_path"] = "__PENDING__"

        c["final_mask"] = final_mask
        c["record"] = record

        records_by_subject.setdefault(subject, []).append(record)

    return records_by_subject


# ── Stage 3: centered crop + save outputs ─────────────────────────────────────

def stage3_center_crop_and_save(crops: list[dict],
                                records_by_subject: dict[str, list[dict]]) -> None:
    half_size_x, half_size_y = compute_common_center_crop(crops)

    shared_mask = None
    for c in crops:
        if "final_mask" not in c:
            continue

        jpg_path   = c["jpg_path"]
        subject    = c["subject"]
        inner_crop = c["inner_crop"]
        stem       = jpg_path.stem
        final_mask = c["final_mask"]
        record     = c["record"]

        try:
            centered_image = crop_square_around_mask(inner_crop, record["cx"], record["cy"], half_size_x, half_size_y)
            c["inner_crop"] = centered_image
            centered_mask = crop_square_around_mask(final_mask, record["cx"], record["cy"], half_size_x, half_size_y)
            c["final_mask"] = centered_mask
        except ValueError as e:
            record["status"] = "failed"
            record["reason"] = str(e)
            print(f"  [FAIL] {jpg_path.name}: {e}")
            continue

        final_mask = centered_mask
        shared_mask = (final_mask if shared_mask is None
                       else cv2.bitwise_and(shared_mask, final_mask))

        shared_mask = cv2.bitwise_not(fill_arrow_lines(cv2.bitwise_not(shared_mask)))

        # try:
            # gray_float = prepare_gray_image(masked_image, final_mask)
        # except ValueError as e:
        #     record["status"] = "failed"
        #     record["reason"] = str(e)
        #     print(f"  [FAIL] {jpg_path.name}: {e}")
        #     continue

        # gray_u8 = (gray_float * 255).astype(np.uint8)

        out_dir = OUTPUT_DIR / subject
        out_dir.mkdir(parents=True, exist_ok=True)

        octa_path = out_dir / f"{stem}_octa.png"
        mask_path = out_dir / f"{stem}_mask.png"
        # gray_path = out_dir / f"{stem}_gray.png"

        cv2.imwrite(str(octa_path), centered_image)
        cv2.imwrite(str(mask_path), final_mask)
        # cv2.imwrite(str(gray_path), gray_u8)

        record["octa_path"] = str(octa_path)
        record["mask_path"] = str(mask_path)
        # record["gray_path"] = str(gray_path)
        record["status"] = "ok"
        record["reason"] = "ok"
        print(f"  [OK]  {jpg_path.name}")

    if shared_mask is not None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        shared_path = OUTPUT_DIR / "shared_mask.png"
        cv2.imwrite(str(shared_path), shared_mask)
        print(f"\nShared mask saved → {shared_path}")
    else:
        print("\nNo masks built.")

    fieldnames = ["file_name", "stem", "subject", "eye", "status",
                  "crop_width", "crop_height", "octa_path",
                  "mask_path", "gray_path", "reason", "cx", "cy"]
    for subject, records in records_by_subject.items():
        out_dir = OUTPUT_DIR / subject
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "extraction_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print(f"  Saved summary → {csv_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    patient_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()])
    if not patient_dirs:
        print(f"No patient subfolders found in {INPUT_DIR}")
        return

    crops = stage1_crop(patient_dirs)
    if not crops:
        print("No images successfully cropped.")
        return

    records_by_subject = stage2_build_masks(crops)
    stage3_center_crop_and_save(crops, records_by_subject)

    print("\nDone.")

if __name__ == "__main__":
    main()
