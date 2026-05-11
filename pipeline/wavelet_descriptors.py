import csv
from pathlib import Path
from mask_arrows import find_mode_of_arrows

import cv2
import numpy as np


OUTPUT_ROOT = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output_center_arrow_cropped")
WAVELET_LEVELS = 3


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
    # im = pad_mirror(im, l, axis=0)
    # return 0.5 * (im[:-l, :] + im[l:, :])
    N = im.shape[0]
    bottom = im[N - 2:N - 2 - l:-1, :]  # l mirror rows: N-2, N-3, ..., N-1-l
    padded = np.vstack([im, bottom])  # (N+l, M)
    return 0.5 * (padded[:N, :] + padded[l:N + l, :])  # (N, M)


def filterg(im: np.ndarray, l: int) -> np.ndarray:
    """High-pass Haar filtering along rows/columns, matching the MATLAB logic."""
    # im = pad_mirror(im, l, axis=0)
    # return 0.5 * (im[l:, :] - im[:-l, :])
    N = im.shape[0]
    bottom = im[N - 2:N - 2 - l:-1, :]
    padded = np.vstack([im, bottom])
    return 0.5 * (padded[l:N + l, :] - padded[:N, :])


#4 compare, padding?,
# when filter, do convolution, minus mean from image and then replace values by zeroes.

def waveletdescr(im: np.ndarray, maxlevel: int = 3) -> np.ndarray:
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
    npix = m*n

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

        print(f"[Level {i}] l={l} | imhy={imhy.shape} imgy={imgy.shape}")

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

def divide_in_quadrants(image: np.ndarray, mask: np.ndarray):
    """Split an image into four quadrants."""
    l, r, t, b = find_mode_of_arrows(cv2.bitwise_not(mask))

    return image[:t, :l], image[:t, r:], image[b:, :l], image[b:, r:]

def process_wavelet_folder_quadrants(
    output_dir: Path,
    image_suffix: str = "_octa.png",
    mask_name: str = "shared_mask.png",
    maxlevel: int = WAVELET_LEVELS,
) -> None:
    shared_mask_path = output_dir / mask_name
    shared_mask = cv2.imread(str(shared_mask_path), cv2.IMREAD_GRAYSCALE)

    if shared_mask is None:
        raise ValueError(f"Could not read shared mask: {shared_mask_path}")

    rows = []

    for image_path in sorted(output_dir.glob(f"*/*{image_suffix}")):
        try:

            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"[SKIP] Could not read image: {image_path.name}")
                continue

            gray = image.astype(np.float64) / 255.0
            # print(f"gray shape = {gray.shape}")

            tl, tr, bl, br = divide_in_quadrants(gray, shared_mask)
            print(f"tl shape = {tl.shape} tr shape = {tr.shape} bl shape = {bl.shape} br shape = {br.shape}")

            quadrants = {
                "tl": tl,
                "tr": tr,
                "bl": bl,
                "br": br,
            }

            row = {
                "subject": image_path.parent.name,
                "file":    image_path.name,
            }

            skip = False
            for qname, quad in quadrants.items():
                if quad.size == 0:
                    print(f"[SKIP] Empty quadrant {qname} for {image_path.name}")
                    skip = True
                    break

                try:
                    features = waveletdescr(quad, maxlevel=maxlevel)
                    for idx, value in enumerate(features, start=1):
                        row[f"{qname}_w{idx:02d}"] = f"{value:.8f}"
                except ValueError as e:
                    print(f"[SKIP] Quadrant {qname} failed for {image_path.name}: {e}")
                    skip = True
                    break

            if skip:
                continue

            rows.append(row)
            print(f"[OK] {image_path.name}")

        except ValueError as e:
            print(f"[FAIL] {image_path}: {e}")

    if not rows:
        print("No wavelet features were extracted.")
        return

    csv_path = output_dir / "wavelet_features_quadrants.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved features to: {csv_path}")


def _visualize_quadrant(
        quad: np.ndarray,
        qname: str,
        maxlevel: int,
        out_dir: Path,
        stem: str,
) -> None:
    """Save a filter decomposition figure for one quadrant."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    m, n = quad.shape
    col_titles = ["Low-pass (imhy)", "High-pass (imgy)", "HH", "LH", "HL", "Approx → next level"]

    fig = plt.figure(figsize=(18, 5 * maxlevel))
    gs = gridspec.GridSpec(maxlevel, 6, figure=fig, hspace=0.4, wspace=0.3)

    im = quad.copy()
    for i in range(1, maxlevel + 1):
        l = 2 ** i

        imhy = filterh(im, l)
        imgy = filterg(im, l)
        hh = filterg(imgy.T, l).T
        lh = filterh(imgy.T, l).T
        hl = filterg(imhy.T, l).T
        approx = filterh(imhy.T, l).T

        panels = [imhy, imgy, hh, lh, hl, approx]

        for j, (panel, title) in enumerate(zip(panels, col_titles)):
            ax = fig.add_subplot(gs[i - 1, j])
            cmap = "RdBu_r" if j in (1, 2, 3, 4) else "gray"
            vmax = np.abs(panel).max() or 1.0
            vmin = -vmax if j in (1, 2, 3, 4) else panel.min()
            ax.imshow(panel, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"L{i} — {title}\n{panel.shape}", fontsize=8)
            ax.axis("off")
            if j in (2, 3, 4):
                energy = np.sum(panel ** 2) / (m * n)
                ax.set_xlabel(f"energy={energy:.2e}", fontsize=7)

        im = approx

    quadrant_labels = {"tl": "Top-Left", "tr": "Top-Right", "bl": "Bottom-Left", "br": "Bottom-Right"}
    fig.suptitle(f"Haar wavelet — {quadrant_labels[qname]} quadrant — {stem}", fontsize=12)

    out_path = out_dir / f"{stem}_{qname}_wavelet_filters.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {quadrant_labels[qname]} → {out_path.name}")


def visualize_filters(
        image_path: Path,
        mask_path: Path,
        out_dir: Path,
        maxlevel: int = 3,
) -> None:
    """
    Visualize Haar wavelet filter outputs per quadrant for a single image,
    matching exactly what process_wavelet_folder_quadrants does.

    For each of the four quadrants (TL, TR, BL, BR), saves a figure with
    maxlevel rows and 6 columns:
      Low-pass | High-pass | HH | LH | HL | Approx → next level

    Args:
        image_path: path to a *_octa.png image
        mask_path:  path to the shared_mask.png used for quadrant splitting
        out_dir:    directory where figures are saved
        maxlevel:   number of wavelet decomposition levels (default 3)
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mask: {mask_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    gray = image.astype(np.float64) / 255.0
    tl, tr, bl, br = divide_in_quadrants(gray, mask)

    quadrants = {"tl": tl, "tr": tr, "bl": bl, "br": br}
    stem = image_path.stem

    print(f"Quadrant sizes — TL:{tl.shape} TR:{tr.shape} BL:{bl.shape} BR:{br.shape}")

    for qname, quad in quadrants.items():
        if quad.size == 0:
            print(f"[SKIP] Empty quadrant {qname}")
            continue
        _visualize_quadrant(quad, qname, maxlevel, out_dir, stem)


if __name__ == "__main__":
    OUTPUT_DIR = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output_center_arrow_cropped")
    process_wavelet_folder_quadrants(OUTPUT_DIR)
    # Visualize filters for the first image found
    sample_images = sorted(OUTPUT_DIR.glob("14/*_octa.png"))
    if sample_images:
        visualize_filters(
            image_path=sample_images[0],
            mask_path=OUTPUT_DIR / "shared_mask.png",
            out_dir=OUTPUT_DIR / "filter_visualizations_control",
            maxlevel=3,
        )
    else:
        print("No *_octa.png images found under", OUTPUT_DIR)

    sample_images = sorted(OUTPUT_DIR.glob("1/*_octa.png"))
    if sample_images:
        visualize_filters(
            image_path=sample_images[0],
            mask_path=OUTPUT_DIR / "shared_mask.png",
            out_dir=OUTPUT_DIR / "filter_visualizations",
            maxlevel=3,
        )
    else:
        print("No *_octa.png images found under", OUTPUT_DIR)

#histograms plots samples, train classifier to see if they can be separated