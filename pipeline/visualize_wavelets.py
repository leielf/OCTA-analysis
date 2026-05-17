from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


OUTPUT_DIR   = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output_center_arrow_cropped")
FEATURES_CSV = OUTPUT_DIR / "wavelet_features_quadrants.csv"
PLOTS_DIR    = OUTPUT_DIR / "group_comparison_plots_wavelets"

CONTROL_SUBJECTS = {"14", "15"}  # <-- fill in your actual control subject IDs

QUADRANTS = ["tl", "tr", "bl", "br"]
QUADRANT_NAMES = {"tl": "Top Left", "tr": "Top Right", "bl": "Bottom Left", "br": "Bottom Right"}
N_WAVELET_FEATURES = 7

FEATURE_LABELS = [
    "HH lv2", "LH lv2", "HL lv2",
    "HH lv3", "LH lv3", "HL lv3",
    "Residual",
]

FEATURE_COLUMNS = [
    f"{q}_w{i:02d}"
    for q in QUADRANTS
    for i in range(4, N_WAVELET_FEATURES + 4)
]


# ── I/O ───────────────────────────────────────────────────────────────────────

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


# ── Group labelling ───────────────────────────────────────────────────────────

def assign_group(subject: str) -> str:
    return "control" if subject in CONTROL_SUBJECTS else "patient"


def add_group_labels(rows: list[dict]) -> list[dict]:
    labeled = []
    for row in rows:
        row = row.copy()
        row["group"] = assign_group(row["subject"])
        labeled.append(row)
    return labeled


# ── Statistics ────────────────────────────────────────────────────────────────

def get_values(rows: list[dict], feature: str, group: str) -> np.ndarray:
    return np.array(
        [float(row[feature]) for row in rows if row["group"] == group],
        dtype=np.float64,
    )


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta effect size.
    Positive: values in x tend to be larger than values in y.
    """
    greater = sum(np.sum(xi > y) for xi in x)
    less    = sum(np.sum(xi < y) for xi in x)
    return float((greater - less) / (len(x) * len(y)))


def compare_feature(rows: list[dict], feature: str) -> dict:
    patient_values = get_values(rows, feature, "patient")
    control_values = get_values(rows, feature, "control")

    if len(patient_values) == 0 or len(control_values) == 0:
        raise ValueError(f"Missing patient or control values for {feature}")

    _, p_value = mannwhitneyu(patient_values, control_values, alternative="two-sided")

    return {
        "feature":                         feature,
        "patient_mean":                    float(np.mean(patient_values)),
        "patient_median":                  float(np.median(patient_values)),
        "patient_std":                     float(np.std(patient_values, ddof=1)) if len(patient_values) > 1 else 0.0,
        "control_mean":                    float(np.mean(control_values)),
        "control_median":                  float(np.median(control_values)),
        "control_std":                     float(np.std(control_values, ddof=1)) if len(control_values) > 1 else 0.0,
        "p_value":                         float(p_value),
        "cliffs_delta_patient_vs_control": cliffs_delta(patient_values, control_values),
        "n_patient":                       int(len(patient_values)),
        "n_control":                       int(len(control_values)),
    }


# ── 1. Individual boxplots ────────────────────────────────────────────────────

def plot_feature_boxplot(rows: list[dict], feature: str, out_dir: Path) -> None:
    patient_values = get_values(rows, feature, "patient")
    control_values = get_values(rows, feature, "control")

    groups = [control_values, patient_values]

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.boxplot(
        groups,
        tick_labels=["control", "patient"],
        showmeans=True,
        patch_artist=True,
        widths=0.45,
        meanprops={
            "marker": "^",
            "markerfacecolor": "green",
            "markeredgecolor": "green",
            "markersize": 8,
        },
        medianprops={"color": "darkorange", "linewidth": 2},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.5},
        whiskerprops={"color": "black", "linewidth": 1.5},
        capprops={"color": "black", "linewidth": 1.5},
    )

    rng = np.random.default_rng(42)
    for x_pos, values, color in zip([1, 2], groups, ["royalblue", "crimson"]):
        jitter = rng.normal(loc=0.0, scale=0.045, size=len(values))
        ax.scatter(
            np.full(len(values), x_pos) + jitter,
            values,
            s=45, alpha=0.8, color=color,
            edgecolor="black", linewidth=0.5, zorder=3,
        )

    _, p_value = mannwhitneyu(patient_values, control_values, alternative="two-sided")
    delta = cliffs_delta(patient_values, control_values)

    ax.set_title(
        f"{feature}\nMann–Whitney p = {p_value:.4f},  Cliff's δ = {delta:.3f}",
        fontsize=11,
    )
    ax.set_ylabel(feature)
    ax.grid(True, axis="y", alpha=0.25)

    for x_pos, values in zip([1, 2], groups):
        ax.text(
            x_pos, ax.get_ylim()[0],
            f"n={len(values)}\nmedian={np.median(values):.3g}\nmean={np.mean(values):.3g}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.tight_layout()
    out_path = out_dir / f"{feature}_boxplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path.name}")


# ── 2. Summary heatmap ────────────────────────────────────────────────────────

def plot_summary_heatmap(rows: list[dict], out_dir: Path) -> None:
    """
    Two side-by-side heatmaps:
      Left:  -log10(p-value) — higher = more significant
      Right: Cliff's delta   — direction and magnitude of effect

    Rows = wavelet features (labelled by subband and level)
    Columns = quadrants (TL, TR, BL, BR)
    """
    pval_matrix  = np.zeros((N_WAVELET_FEATURES, len(QUADRANTS)))
    delta_matrix = np.zeros((N_WAVELET_FEATURES, len(QUADRANTS)))

    for j, q in enumerate(QUADRANTS):
        for i in range(1, N_WAVELET_FEATURES + 1):
            feature = f"{q}_w{i:02d}"
            pat  = get_values(rows, feature, "patient")
            ctrl = get_values(rows, feature, "control")
            if len(pat) == 0 or len(ctrl) == 0:
                pval_matrix[i-1, j]  = np.nan
                delta_matrix[i-1, j] = np.nan
                continue
            _, p = mannwhitneyu(pat, ctrl, alternative="two-sided")
            pval_matrix[i-1, j]  = -np.log10(p + 1e-10)
            delta_matrix[i-1, j] = cliffs_delta(pat, ctrl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # left: significance
    ax = axes[0]
    im = ax.imshow(pval_matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(QUADRANTS)))
    ax.set_xticklabels([QUADRANT_NAMES[q] for q in QUADRANTS], fontsize=10)
    ax.set_yticks(range(N_WAVELET_FEATURES))
    ax.set_yticklabels(FEATURE_LABELS, fontsize=9)
    ax.set_title("Significance\n−log₁₀(p-value)\n(higher = more significant)", fontsize=11)
    threshold = -np.log10(0.05)
    for i in range(N_WAVELET_FEATURES):
        for j in range(len(QUADRANTS)):
            val = pval_matrix[i, j]
            if not np.isnan(val) and val >= threshold:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                    fill=False, edgecolor="black", linewidth=2))
            ax.text(j, i, f"{val:.1f}" if not np.isnan(val) else "",
                    ha="center", va="center", fontsize=7,
                    color="black" if val < 2 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # right: effect size
    ax = axes[1]
    im2 = ax.imshow(delta_matrix, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(QUADRANTS)))
    ax.set_xticklabels([QUADRANT_NAMES[q] for q in QUADRANTS], fontsize=10)
    ax.set_yticks(range(N_WAVELET_FEATURES))
    ax.set_yticklabels(FEATURE_LABELS, fontsize=9)
    ax.set_title("Effect size\nCliff's delta\n(+1 = patients always higher)", fontsize=11)
    for i in range(N_WAVELET_FEATURES):
        for j in range(len(QUADRANTS)):
            val = delta_matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Wavelet feature comparison: patients vs controls", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = out_dir / "wavelet_summary_heatmap.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Summary heatmap saved: {out_path.name}")


# ── 3. Mean profile plot ──────────────────────────────────────────────────────

def plot_mean_profiles(rows: list[dict], out_dir: Path) -> None:
    """
    For each quadrant, plot the mean feature value per wavelet level/subband
    for patients vs controls as a line plot with shaded std.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for ax, q in zip(axes, QUADRANTS):
        pat_means, pat_stds   = [], []
        ctrl_means, ctrl_stds = [], []

        for i in range(1, N_WAVELET_FEATURES + 1):
            feature = f"{q}_w{i:02d}"
            pat  = get_values(rows, feature, "patient")
            ctrl = get_values(rows, feature, "control")
            pat_means.append(np.mean(pat))
            pat_stds.append(np.std(pat, ddof=1) if len(pat) > 1 else 0)
            ctrl_means.append(np.mean(ctrl))
            ctrl_stds.append(np.std(ctrl, ddof=1) if len(ctrl) > 1 else 0)

        x = np.arange(N_WAVELET_FEATURES)
        pat_means  = np.array(pat_means)
        pat_stds   = np.array(pat_stds)
        ctrl_means = np.array(ctrl_means)
        ctrl_stds  = np.array(ctrl_stds)

        ax.plot(x, ctrl_means, color="royalblue", marker="o", label="Control")
        ax.fill_between(x, ctrl_means - ctrl_stds, ctrl_means + ctrl_stds,
                        color="royalblue", alpha=0.15)
        ax.plot(x, pat_means, color="crimson", marker="o", label="Patient")
        ax.fill_between(x, pat_means - pat_stds, pat_means + pat_stds,
                        color="crimson", alpha=0.15)

        ax.set_xticks(x)
        ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_title(QUADRANT_NAMES[q], fontsize=11)
        ax.set_ylabel("Mean energy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Mean wavelet energy profile: patients vs controls\n(shaded = ±1 std)", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "wavelet_mean_profiles.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Mean profiles saved: {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv(FEATURES_CSV)
    labeled_rows = add_group_labels(rows)

    labeled_csv = OUTPUT_DIR / "wavelet_features_labeled.csv"
    save_csv(labeled_rows, labeled_csv)
    print(f"[OK] Saved labeled features: {labeled_csv}")

    # individual boxplots + comparison CSV
    comparison_rows = []
    for feature in FEATURE_COLUMNS:
        try:
            result = compare_feature(labeled_rows, feature)
            comparison_rows.append(result)
            plot_feature_boxplot(labeled_rows, feature, PLOTS_DIR)
            print(
                f"{feature}: "
                f"patient median={result['patient_median']:.4f}, "
                f"control median={result['control_median']:.4f}, "
                f"p={result['p_value']:.4f}, "
                f"delta={result['cliffs_delta_patient_vs_control']:.4f}"
            )
        except ValueError as e:
            print(f"[SKIP] {feature}: {e}")

    comparison_csv = OUTPUT_DIR / "wavelet_group_comparison.csv"
    save_csv(comparison_rows, comparison_csv)
    print(f"[OK] Saved comparison results: {comparison_csv.name}")

    # summary plots
    plot_summary_heatmap(labeled_rows, PLOTS_DIR)
    plot_mean_profiles(labeled_rows, PLOTS_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
