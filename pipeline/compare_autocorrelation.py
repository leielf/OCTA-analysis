from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


OUTPUT_DIR = Path("/Users/leielf/Desktop/uni/cvut/semestral project/medical_images/output_center_arrow_cropped")
FEATURES_CSV = OUTPUT_DIR / "autocorrelation_features_indiv_masks_no_grayscale.csv"
PLOTS_DIR = OUTPUT_DIR / "group_comparison_plots_indiv_masks_no_grayscale"

CONTROL_SUBJECTS = {"14", "15"}

FEATURE_COLUMNS = [
    "corr_length_1e",
    "peak_width_0_5",
    "anisotropy",
    "radial_mean_0_10",
    "radial_mean_10_30",
    "radial_mean_30_60",
    "radial_mean_60_100",
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
        f"Mann–Whitney p = {p_value:.4f}, Cliff's δ = {delta:.3f}",
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


if __name__ == "__main__":
    main()