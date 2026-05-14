from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


OUTPUT_DIR = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output_center_arrow_cropped")
FEATURES_CSV = OUTPUT_DIR / "autocorrelation_features.csv"
PLOTS_DIR = OUTPUT_DIR / "group_comparison_plots"

AC_SUFFIX  = "_octa_autocorrelation.npy"
MAX_RADIUS = 30

CONTROL_SUBJECTS = {"14", "15"}

FEATURE_COLUMNS = [
    "corr_length_1e",
    "peak_width_0_5",
    "anisotropy",
    "radial_mean_0_10",
    "radial_mean_10_30",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        print(f"[SKIP] No rows to save: {path}")
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def assign_group(subject: str) -> str:
    if subject in CONTROL_SUBJECTS:
        return "control"

    return "patient"


def add_group_labels(rows: list[dict]) -> list[dict]:
    labeled_rows = []

    for row in rows:
        row = row.copy()
        row["group"] = assign_group(row["subject"])
        labeled_rows.append(row)

    return labeled_rows


def get_feature_values(rows: list[dict], feature: str, group: str) -> np.ndarray:
    values = []

    for row in rows:
        if row["group"] == group:
            values.append(float(row[feature]))

    return np.array(values, dtype=np.float64)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cliff's delta effect size.

    Positive value means values in x tend to be larger than values in y.
    Negative value means values in x tend to be smaller than values in y.
    """
    greater = 0
    less = 0

    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)

    return float((greater - less) / (len(x) * len(y)))


def compare_feature(rows: list[dict], feature: str) -> dict:
    patient_values = get_feature_values(rows, feature, "patient")
    control_values = get_feature_values(rows, feature, "control")

    if len(patient_values) == 0 or len(control_values) == 0:
        raise ValueError(f"Missing patient or control values for {feature}")

    _, p_value = mannwhitneyu(
        patient_values,
        control_values,
        alternative="two-sided",
    )

    return {
        "feature": feature,
        "patient_mean": float(np.mean(patient_values)),
        "patient_median": float(np.median(patient_values)),
        "patient_std": float(np.std(patient_values, ddof=1)) if len(patient_values) > 1 else 0.0,
        "control_mean": float(np.mean(control_values)),
        "control_median": float(np.median(control_values)),
        "control_std": float(np.std(control_values, ddof=1)) if len(control_values) > 1 else 0.0,
        "p_value": float(p_value),
        "cliffs_delta_patient_vs_control": cliffs_delta(patient_values, control_values),
        "n_patient": int(len(patient_values)),
        "n_control": int(len(control_values)),
    }


def plot_feature_boxplot(rows: list[dict], feature: str, out_dir: Path) -> None:
    patient_values = get_feature_values(rows, feature, "patient")
    control_values = get_feature_values(rows, feature, "control")

    groups = [control_values, patient_values]
    labels = ["control", "patient"]

    fig, ax = plt.subplots(figsize=(6, 5))

    box = ax.boxplot(
        groups,
        labels=labels,
        showmeans=True,
        patch_artist=True,
        widths=0.45,
        meanprops={
            "marker": "^",
            "markerfacecolor": "green",
            "markeredgecolor": "green",
            "markersize": 8,
        },
        medianprops={
            "color": "darkorange",
            "linewidth": 2,
        },
        boxprops={
            "facecolor": "white",
            "edgecolor": "black",
            "linewidth": 1.5,
        },
        whiskerprops={
            "color": "black",
            "linewidth": 1.5,
        },
        capprops={
            "color": "black",
            "linewidth": 1.5,
        },
    )

    rng = np.random.default_rng(42)

    for x_position, values, color in zip([1, 2], groups, ["royalblue", "crimson"]):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_position) + jitter,
            values,
            s=45,
            alpha=0.8,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )

    control_median = np.median(control_values)
    patient_median = np.median(patient_values)
    control_mean = np.mean(control_values)
    patient_mean = np.mean(patient_values)

    _, p_value = mannwhitneyu(
        patient_values,
        control_values,
        alternative="two-sided",
    )

    delta = cliffs_delta(patient_values, control_values)

    ax.set_title(
        f"{feature}\n"
        f"Wilcoxon p = {p_value:.4f}, Cliff's δ = {delta:.3f}",
        fontsize=11,
    )

    ax.set_ylabel(feature)
    ax.grid(True, axis="y", alpha=0.25)

    ax.text(
        1,
        ax.get_ylim()[0],
        f"n={len(control_values)}\nmedian={control_median:.3g}\nmean={control_mean:.3g}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    ax.text(
        2,
        ax.get_ylim()[0],
        f"n={len(patient_values)}\nmedian={patient_median:.3g}\nmean={patient_mean:.3g}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.tight_layout()

    out_path = out_dir / f"{feature}_boxplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved plot: {out_path}")


def _radial_profile(ac: np.ndarray, max_radius: int = MAX_RADIUS) -> np.ndarray:
    """Average autocorrelation values at each integer distance from the center."""
    cy = ac.shape[0] // 2
    cx = ac.shape[1] // 2
    yy, xx = np.indices(ac.shape)
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    profile = np.full(max_radius + 1, np.nan, dtype=np.float64)
    for r in range(max_radius + 1):
        vals = ac[radius == r]
        if vals.size > 0:
            profile[r] = np.mean(vals)
    return profile


def _load_profiles(rows: list[dict], output_dir: Path) -> dict[str, list[np.ndarray]]:
    """
    Load saved autocorrelation .npy maps and compute radial profiles per group.
    Returns {"patient": [...], "control": [...]}.
    """
    profiles: dict[str, list[np.ndarray]] = {"patient": [], "control": []}

    for row in rows:
        subject = row["subject"]
        filename = row["file"]                          # e.g. ODHR01_octa.png
        stem = filename.replace("_octa.png", "")
        ac_path = output_dir / subject / f"{stem}{AC_SUFFIX}"

        if not ac_path.exists():
            print(f"[SKIP] AC map not found: {ac_path}")
            continue

        try:
            ac = np.load(str(ac_path))
            profile = _radial_profile(ac, max_radius=MAX_RADIUS)
            profiles[row["group"]].append(profile)
        except Exception as e:
            print(f"[SKIP] Could not load {ac_path}: {e}")

    return profiles


def plot_mean_radial_profiles(rows: list[dict], output_dir: Path, out_dir: Path) -> None:
    """
    Plot mean radial autocorrelation profile ± 1 std for patients vs controls,
    with 1/e and 0.5 thresholds marked.

    Matches the visualization style used in Turco et al. (2022) for OCTA
    choriocapillaris texture comparison.
    """
    profiles = _load_profiles(rows, output_dir)

    pat_profiles  = profiles["patient"]
    ctrl_profiles = profiles["control"]

    if not pat_profiles and not ctrl_profiles:
        print("[SKIP] No autocorrelation maps found — skipping radial profile plot.")
        return

    x = np.arange(MAX_RADIUS + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for group_profiles, color, label in [
        (ctrl_profiles, "royalblue", "Control"),
        (pat_profiles,  "crimson",   "Patient"),
    ]:
        if not group_profiles:
            continue
        arr  = np.array(group_profiles)          # (n_images, MAX_RADIUS+1)
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr,  axis=0)
        n    = len(group_profiles)
        ax.plot(x, mean, color=color, linewidth=2, label=f"{label} (n={n})")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    # threshold lines matching corr_length_1e and peak_width_0_5 features
    ax.axhline(1 / np.e, color="gray",  linestyle="--", linewidth=1.2,
               label=f"1/e ≈ {1/np.e:.3f}  (corr_length_1e threshold)")
    ax.axhline(0.5,      color="dimgray", linestyle=":",  linewidth=1.2,
               label="0.5  (peak_width_0_5 threshold)")

    ax.set_xlabel("Radial distance (pixels)", fontsize=11)
    ax.set_ylabel("Mean autocorrelation", fontsize=11)
    ax.set_title("Mean radial autocorrelation profile: patients vs controls\n"
                 "(shaded = ±1 std)", fontsize=12)
    ax.set_xlim(0, MAX_RADIUS)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "mean_radial_autocorrelation_profiles.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved radial profile plot: {out_path.name}")


def plot_individual_radial_profiles(rows: list[dict], output_dir: Path, out_dir: Path) -> None:
    """
    Plot each image's radial profile as a thin line, colored by group,
    with the group mean overlaid as a thick line.

    Useful for spotting outliers and seeing individual variation.
    """
    profiles = _load_profiles(rows, output_dir)
    x = np.arange(MAX_RADIUS + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for group, color, label in [
        ("control", "royalblue", "Control"),
        ("patient", "crimson",   "Patient"),
    ]:
        group_profiles = profiles[group]
        if not group_profiles:
            continue
        arr = np.array(group_profiles)
        for profile in arr:
            ax.plot(x, profile, color=color, linewidth=0.6, alpha=0.35)
        ax.plot(x, np.nanmean(arr, axis=0), color=color, linewidth=2.5,
                label=f"{label} mean (n={len(group_profiles)})")

    ax.axhline(1 / np.e, color="gray",    linestyle="--", linewidth=1.2,
               label=f"1/e ≈ {1/np.e:.3f}")
    ax.axhline(0.5,      color="dimgray", linestyle=":",  linewidth=1.2,
               label="0.5")

    ax.set_xlabel("Radial distance (pixels)", fontsize=11)
    ax.set_ylabel("Autocorrelation", fontsize=11)
    ax.set_title("Individual radial autocorrelation profiles: patients vs controls", fontsize=12)
    ax.set_xlim(0, MAX_RADIUS)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "individual_radial_autocorrelation_profiles.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved individual profile plot: {out_path.name}")


def plot_feature_violin(rows: list[dict], feature: str, out_dir: Path) -> None:
    """
    Combined violin + strip plot.

    - Patients: violin (distribution shape) + individual points
    - Controls: individual points only (too few for a meaningful violin)
    - Mean marked with a horizontal bar, median with a diamond
    - Wilcoxon p and Cliff's delta in the title
    """
    patient_values = get_feature_values(rows, feature, "patient")
    control_values = get_feature_values(rows, feature, "control")

    _, p_value = mannwhitneyu(patient_values, control_values, alternative="two-sided")
    delta = cliffs_delta(patient_values, control_values)

    fig, ax = plt.subplots(figsize=(6, 5))

    # ── violin for patients (position 2) ─────────────────────────────────────
    if len(patient_values) > 4:
        parts = ax.violinplot(
            [patient_values],
            positions=[2],
            widths=0.5,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("crimson")
            pc.set_alpha(0.25)
            pc.set_edgecolor("crimson")

    # ── violin for controls (position 1) if enough points ────────────────────
    if len(control_values) > 4:
        parts = ax.violinplot(
            [control_values],
            positions=[1],
            widths=0.5,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("royalblue")
            pc.set_alpha(0.25)
            pc.set_edgecolor("royalblue")

    # ── individual points ─────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    for x_pos, values, color in zip([1, 2], [control_values, patient_values], ["royalblue", "crimson"]):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), x_pos) + jitter,
            values,
            s=50, alpha=0.85, color=color,
            edgecolor="black", linewidth=0.5, zorder=3,
        )

    # ── mean and median markers ───────────────────────────────────────────────
    for x_pos, values, color in zip([1, 2], [control_values, patient_values], ["royalblue", "crimson"]):
        mean   = np.mean(values)
        median = np.median(values)
        # mean: horizontal line
        ax.plot([x_pos - 0.18, x_pos + 0.18], [mean, mean],
                color=color, linewidth=2.5, zorder=4, solid_capstyle="round")
        # median: diamond
        ax.scatter([x_pos], [median], marker="D", s=55, color="white",
                   edgecolor=color, linewidth=1.8, zorder=5)

    # ── annotations ──────────────────────────────────────────────────────────
    ymin = ax.get_ylim()[0]
    for x_pos, values, color in zip([1, 2], [control_values, patient_values], ["royalblue", "crimson"]):
        ax.text(
            x_pos, ymin,
            f"n={len(values)}\nmedian={np.median(values):.3g}\nmean={np.mean(values):.3g}",
            ha="center", va="bottom", fontsize=8, color=color,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Control", "Patient"], fontsize=11)
    ax.set_ylabel(feature, fontsize=10)
    ax.set_title(
        f"{feature}\nWilcoxon p = {p_value:.4f},  Cliff's δ = {delta:.3f}",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.25)

    # legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", linewidth=2.5, label="Mean"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="gray",
               markersize=7, label="Median", linewidth=0),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    fig.tight_layout()
    out_path = out_dir / f"{feature}_violin.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(FEATURES_CSV)
    labeled_rows = add_group_labels(rows)

    labeled_csv = OUTPUT_DIR / "autocorrelation_features_labeled.csv"
    save_csv(labeled_rows, labeled_csv)
    print(f"[OK] Saved labeled features: {labeled_csv}")

    comparison_rows = []

    for feature in FEATURE_COLUMNS:
        result = compare_feature(labeled_rows, feature)
        comparison_rows.append(result)
        plot_feature_boxplot(labeled_rows, feature, PLOTS_DIR)
        plot_feature_violin(labeled_rows, feature, PLOTS_DIR)

        print(
            f"{feature}: "
            f"patient median={result['patient_median']:.4f}, "
            f"control median={result['control_median']:.4f}, "
            f"p={result['p_value']:.4f}, "
            f"delta={result['cliffs_delta_patient_vs_control']:.4f}"
        )

    comparison_csv = OUTPUT_DIR / "autocorrelation_group_comparison.csv"
    save_csv(comparison_rows, comparison_csv)
    print(f"[OK] Saved comparison results: {comparison_csv}")

    # radial profile plots
    plot_mean_radial_profiles(labeled_rows, OUTPUT_DIR, PLOTS_DIR)
    plot_individual_radial_profiles(labeled_rows, OUTPUT_DIR, PLOTS_DIR)


if __name__ == "__main__":
    main()