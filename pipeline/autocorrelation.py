from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from pipeline.mask_arrows import fill_arrow_lines_individual_masks

OUTPUT_ROOT = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output")
AUTOCORR_BINS = 10


def prepare_gray_for_autocorrelation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Convert a color or grayscale OCTA image to normalized grayscale for
    masked autocorrelation.

    The normalization is computed only from pixels inside the mask.
    Pixels outside the mask are set to 0.
    """
    if image is None:
        raise ValueError("Input image is None.")

    if mask is None:
        raise ValueError("Input mask is None.")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    elif image.ndim == 2:
        gray = image.astype(np.float64)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if gray.shape != mask.shape:
        raise ValueError(f"Image and mask must have the same shape, got {gray.shape} and {mask.shape}")

    # valid = mask > 0
    #
    # if not np.any(valid):
    #     raise ValueError("Mask contains no valid pixels.")

    # valid_pixels = gray[valid]
    #
    # p1, p99 = np.percentile(valid_pixels, [1, 99])
    #
    # if p99 <= p1:
    #     raise ValueError("Invalid intensity range for normalization.")
    #
    # gray = np.clip((gray - p1) / (p99 - p1), 0.0, 1.0)
    # gray[~valid] = 0.0

    return gray

def normalize_inside_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if image.shape != mask.shape:
        raise ValueError(f"Image and mask must have the same shape, got {image.shape} and {mask.shape}")

    valid = mask > 0

    if not np.any(valid):
        raise ValueError("Mask contains no valid pixels.")

    pixels = image[valid]

    p1, p99 = np.percentile(pixels, [1, 99])

    if p99 <= p1:
        raise ValueError("Invalid intensity range inside mask.")

    normalized = np.clip((image - p1) / (p99 - p1), 0.0, 1.0)
    normalized[~valid] = 0.0

    return normalized


def autocorrelation_2d(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Compute masked normalized 2D autocorrelation using FFT.

    The mask defines valid pixels. For each spatial shift, the correlation is
    divided by the number of overlapping valid pixels. The result is centered
    so that zero-lag is in the middle and normalized so the central value is 1.
    """
    image = image.astype(np.float64)

    if mask is None:
        valid = np.ones_like(image, dtype=np.float64)
    else:
        if image.shape != mask.shape:
            raise ValueError(f"Image and mask must have the same shape, got {image.shape} and {mask.shape}")
        valid = (mask > 0).astype(np.float64)

    valid_count = np.sum(valid)

    if valid_count == 0:
        raise ValueError("Mask contains no valid pixels.")

    mean_inside_mask = np.sum(image * valid) / valid_count
    x = (image - mean_inside_mask) * valid

    fft_shape = (2 * image.shape[0], 2 * image.shape[1])

    fx = np.fft.fft2(x, s=fft_shape)
    fv = np.fft.fft2(valid, s=fft_shape)

    numerator = np.fft.ifft2(fx * np.conj(fx)).real
    overlap_count = np.fft.ifft2(fv * np.conj(fv)).real

    eps = 1e-12
    ac = numerator / np.maximum(overlap_count, eps)

    ac = np.fft.fftshift(ac)

    center_y = ac.shape[0] // 2
    center_x = ac.shape[1] // 2
    center_value = ac[center_y, center_x]

    if abs(center_value) > eps:
        ac = ac / center_value

    return ac


def radial_profile(ac: np.ndarray, max_radius: int = 100) -> np.ndarray:
    """
        Convert a 2D autocorrelation map into a 1D radial profile.

        The autocorrelation map is centered, so the middle pixel represents
        zero spatial shift. For each radius r, this function averages all
        autocorrelation values located approximately r pixels away from the
        center. The resulting profile describes how quickly texture similarity
        decreases as spatial distance increases.

        Example:
            profile[0]  = autocorrelation at zero shift
            profile[10] = mean autocorrelation about 10 pixels from the center

        This is useful for extracting descriptors such as correlation length
        or mean autocorrelation in distance bands.
    """
    center_y = ac.shape[0] // 2
    center_x = ac.shape[1] // 2

    yy, xx = np.indices(ac.shape)
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2).astype(int)

    profile = np.full(max_radius + 1, np.nan, dtype=np.float64)

    for r in range(max_radius + 1):
        values = ac[radius == r]
        if values.size > 0:
            profile[r] = np.mean(values)

    return profile


def correlation_length(profile: np.ndarray, threshold: float = 1 / np.e) -> float:
    below = np.where(profile < threshold)[0]

    if below.size == 0:
        return float(len(profile) - 1)

    return float(below[0])


def peak_width(ac: np.ndarray, threshold: float = 0.5) -> float:
    center_y = ac.shape[0] // 2
    center_x = ac.shape[1] // 2

    yy, xx = np.indices(ac.shape)
    radius = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    above = ac >= threshold

    if not np.any(above):
        return 0.0

    return float(np.max(radius[above]))


def anisotropy(ac: np.ndarray, max_offset: int = 80) -> float:
    center_y = ac.shape[0] // 2
    center_x = ac.shape[1] // 2

    horizontal = ac[center_y, center_x:center_x + max_offset]
    vertical = ac[center_y:center_y + max_offset, center_x]

    horizontal_energy = np.sum(np.abs(horizontal))
    vertical_energy = np.sum(np.abs(vertical))

    return float(
        abs(horizontal_energy - vertical_energy)
        / (horizontal_energy + vertical_energy + 1e-12)
    )


def extract_autocorrelation_features(ac: np.ndarray, max_radius: int = 100) -> dict:
    profile = radial_profile(ac, max_radius=max_radius)

    return {
        "corr_length_1e": correlation_length(profile, threshold=1 / np.e),
        "peak_width_0_5": peak_width(ac, threshold=0.5),
        "anisotropy": anisotropy(ac, max_offset=min(80, max_radius)),
        "radial_mean_0_10": float(np.nanmean(profile[0:11])),
        "radial_mean_10_30": float(np.nanmean(profile[10:31])),
        "radial_mean_30_60": float(np.nanmean(profile[30:61])),
        "radial_mean_60_100": float(np.nanmean(profile[60:101])),
    }


def process_autocorrelation_folder(
    output_dir: Path,
    image_suffix: str = "_octa.png",
    mask_name: str = "shared_mask.png",
    save_maps: bool = True,
    grayscale: bool = True,
) -> None:
    shared_mask_path = output_dir / mask_name
    shared_mask = cv2.imread(str(shared_mask_path), cv2.IMREAD_GRAYSCALE)

    if shared_mask is None:
        raise ValueError(f"Could not read shared mask: {shared_mask_path}")

    rows = []

    for image_path in sorted(output_dir.glob(f"*/*{image_suffix}")):
        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

            if image is None:
                print(f"[SKIP] Could not read image: {image_path}")
                continue

            stacked_mask = np.stack([shared_mask, shared_mask, shared_mask], axis=-1)
            gray = image
            if grayscale:
                stacked_mask = shared_mask
                gray = prepare_gray_for_autocorrelation(image, stacked_mask)
            # gray = prepare_gray_for_autocorrelation(image, np.stack([shared_mask, shared_mask, shared_mask], axis=-1)) if grayscale else image
            print(f"stacked_mask shape = {stacked_mask.shape} gray shape = {gray.shape}")
            ac = autocorrelation_2d(gray, stacked_mask)
            features = extract_autocorrelation_features(ac)

            if save_maps:
                ac_path = image_path.with_name(image_path.stem + "_autocorrelation.npy")
                np.save(ac_path, ac)

            row = {
                "subject": image_path.parent.name,
                "file": image_path.name,
                **features,
            }
            rows.append(row)

            print(f"[OK] {image_path}")

        except ValueError as e:
            print(f"[FAIL] {image_path}: {e}")

    if not rows:
        print("No autocorrelation features were extracted.")
        return

    csv_path = output_dir / "autocorrelation_features_no_grayscale.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved features to: {csv_path}")

def process_autocorrelation_folder_indiv_masks(
    output_dir: Path,
    image_suffix: str = "_octa.png",
    mask_suffix: str = "_mask.png",
    save_maps: bool = True,
    grayscale: bool = True,
) -> None:
    rows = []

    for image_path in sorted(output_dir.glob(f"*/*{image_suffix}")):
        try:
            # derive mask path from image path
            mask_path = image_path.with_name(
                image_path.name.replace(image_suffix, mask_suffix)
            )

            if not mask_path.exists():
                print(f"[SKIP] No mask found for {image_path.name} (expected {mask_path.name})")
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            mask  = cv2.imread(str(mask_path),  cv2.IMREAD_GRAYSCALE)
            mask = cv2.bitwise_not(fill_arrow_lines_individual_masks(cv2.bitwise_not(mask)))

            if image is None:
                print(f"[SKIP] Could not read image: {image_path}")
                continue

            if mask is None:
                print(f"[SKIP] Could not read mask: {mask_path}")
                continue

            stacked_mask = np.stack([mask, mask, mask], axis=-1)
            gray = image
            if grayscale:
                stacked_mask = mask
                gray = prepare_gray_for_autocorrelation(image, stacked_mask)
            ac = autocorrelation_2d(gray, stacked_mask)
            features = extract_autocorrelation_features(ac)

            if save_maps:
                ac_path = image_path.with_name(image_path.stem + "_autocorrelation.npy")
                np.save(str(ac_path), ac)

            row = {
                "subject": image_path.parent.name,
                "file": image_path.name,
                **features,
            }
            rows.append(row)

            print(f"[OK] {image_path.name}")

        except ValueError as e:
            print(f"[FAIL] {image_path}: {e}")

    if not rows:
        print("No autocorrelation features were extracted.")
        return

    csv_path = output_dir / "autocorrelation_features_indiv_masks_no_grayscale.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved features to: {csv_path}")


if __name__ == "__main__":
    OUTPUT_DIR = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output_center_arrow_cropped")
    process_autocorrelation_folder(OUTPUT_DIR)
    process_autocorrelation_folder_indiv_masks(OUTPUT_DIR)


