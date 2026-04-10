import csv
from pathlib import Path

import cv2
import numpy as np


OUTPUT_ROOT = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output")
WAVELET_LEVELS = 3


def load_gray_and_mask(gray_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if gray is None:
        raise ValueError(f"Cannot read grayscale image: {gray_path}")
    if mask is None:
        raise ValueError(f"Cannot read mask image: {mask_path}")

    gray = gray.astype(np.float64) / 255.0
    mask = (mask > 0).astype(np.float64)

    if gray.shape != mask.shape:
        raise ValueError(f"Image and mask shape mismatch: {gray.shape} vs {mask.shape}")

    return gray, mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero out invalid pixels."""
    return image * mask


def pad_mirror(image: np.ndarray, pad: int, axis: int) -> np.ndarray:
    """Mirror-pad one axis by pad pixels on both sides."""
    if pad <= 0:
        return image

    if axis == 0:
        top = image[1:pad + 1][::-1, :]
        bottom = image[-pad - 1:-1][::-1, :]
        return np.vstack([top, image, bottom])

    left = image[:, 1:pad + 1][:, ::-1]
    right = image[:, -pad - 1:-1][:, ::-1]
    return np.hstack([left, image, right])


def filterh(im: np.ndarray, l: int) -> np.ndarray:
    """Low-pass Haar filtering along rows/columns, matching the MATLAB logic."""
    im = pad_mirror(im, l, axis=0)
    return 0.5 * (im[:-l, :] + im[l:, :])


def filterg(im: np.ndarray, l: int) -> np.ndarray:
    """High-pass Haar filtering along rows/columns, matching the MATLAB logic."""
    im = pad_mirror(im, l, axis=0)
    return 0.5 * (im[l:, :] - im[:-l, :])


def waveletdescr(im: np.ndarray, mask:np.ndarray, maxlevel: int = 3) -> np.ndarray:
    """
    Haar wavelet frame descriptors similar to MATLAB waveletdescr.m.

    This computes the same kind of multilevel wavelet-energy feature vector:
      - three detail-band energies per level
      - one final low-pass (residual) energy

    Feature layout:
      [level1_HH, level1_LH, level1_HL,
       level2_HH, level2_LH, level2_HL,
       ...
       levelN_HH, levelN_LH, levelN_HL,
       lowpass_energy]
    """
    im = im.astype(np.float64)
    m, n = im.shape
    npix = max(np.sum(mask > 0), 1)

    # MATLAB waveletdescr.m equivalent feature vector allocation:
    # 3 coefficients per decomposition level + 1 final low-pass energy.
    v = np.zeros(3 * maxlevel + 1, dtype=np.float64)

    # Multiresolution decomposition loop:
    # corresponds to the repeated Haar wavelet frame filtering steps
    # used in waveletdescr.m / waveletdescr_demo.m.
    for i in range(1, maxlevel + 1):
        l = 2 ** i

        # Guard against requesting more wavelet levels than the image supports.
        # This mirrors the practical limits of the MATLAB implementation.
        if l >= min(m, n):
            raise ValueError("Image too small for the requested number of wavelet levels.")

        # Horizontal / vertical filtering stage for the current wavelet level.
        # These correspond to the Haar analysis filters in the original algorithm.
        imhy = filterh(im, l)
        imgy = filterg(im, l)

        # Energy of the detail subbands at this level.
        # The squared sums match the "descriptor = subband energy" idea
        # used by waveletdescr.m.
        vgg = np.sum(filterg(imgy.T, l) ** 2) / npix
        vhg = np.sum(filterh(imgy.T, l) ** 2) / npix
        vgh = np.sum(filterg(imhy.T, l) ** 2) / npix

        # Prepare the approximation image for the next level of decomposition.
        # This is the recursive low-pass branch of the multilevel wavelet frame.
        im = filterh(imhy.T, l).T

        # Store the three subband energies for this level.
        v[3 * i - 3:3 * i] = [vgg, vhg, vgh]

    # Final low-pass / residual energy after the last decomposition level.
    # This corresponds to the last approximation band in the wavelet descriptor.
    v[-1] = np.sum(im ** 2) / npix

    return v


def compute_descriptors_for_patient(summary_csv: Path) -> list[dict[str, str]]:
    with open(summary_csv, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows: list[dict[str, str]] = []

    for row in rows:
        if row.get("status", "").strip().lower() != "ok":
            continue

        gray_path_text = "/Users/leielf/Desktop/uni/cvut/semestral project/" + row.get("gray_path", "").strip()
        mask_path_text = "/Users/leielf/Desktop/uni/cvut/semestral project/" + row.get("mask_path", "").strip()

        if not gray_path_text or not mask_path_text:
            continue

        gray_path = Path(gray_path_text)
        print(gray_path)
        # gray_path.startswith("/Users/leielf/Desktop/uni/cvut/semestral project/")
        mask_path = Path(mask_path_text)

        gray, mask = load_gray_and_mask(gray_path, mask_path)
        gray_masked = apply_mask(gray, mask)

        features = waveletdescr(gray_masked, mask, maxlevel=WAVELET_LEVELS)

        out_row = dict(row)
        for idx, value in enumerate(features[:-1], start=1):
            out_row[f"w{idx:02d}"] = f"{value:.8f}"
        out_row[f"w{len(features):02d}"] = f"{features[-1]:.8f}"

        out_rows.append(out_row)

    return out_rows


def main() -> None:
    summary_files = sorted(OUTPUT_ROOT.rglob("extraction_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No extraction_summary.csv files found under {OUTPUT_ROOT}")

    for summary_csv in summary_files:
        rows = compute_descriptors_for_patient(summary_csv)
        if not rows:
            print(f"[SKIP] No valid rows in {summary_csv}")
            continue

        base_fields = list(rows[0].keys())
        output_csv = summary_csv.parent / "wavelet_descriptors.csv"

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=base_fields)
            writer.writeheader()
            writer.writerows(rows)

        print(f"[OK] Saved descriptors to: {output_csv}")


if __name__ == "__main__":
    main()