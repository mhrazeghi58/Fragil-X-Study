import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- SETTINGS ---------------- #
ROOT = r"C:\Users\hrazeghikondela\Desktop\Paper\FLIM\HP_NEW_APR24\Results"
SAVE_DIR = ROOT

SAVE_FORMAT = "png"   # choose: "png" or "tif"
DPI = 400

WT_SAMPLE_NAMES = ["Sample_1", "Sample_2", "Sample_3", "Sample_6"]
KO_SAMPLE_NAMES = ["Sample_2", "Sample_3", "Sample_4", "Sample_6"]

WT_COLOR = "forestgreen"
KO_COLOR = "firebrick"

WT_LIGHT = "#8fd19e"
KO_LIGHT = "#e58b8b"

WT_SHADE_COLORS = [
    "#00441b",
    "#006d2c",
    "#238b45",
    "#41ab5d",
    "#74c476",
    "#a1d99b",
]

KO_SHADE_COLORS = [
    "#67000d",
    "#a50f15",
    "#cb181d",
    "#ef3b2c",
    "#fb6a4a",
    "#fc9272",
]

RANDOM_SEED = 42

# ---------------- SEPARATE Y-AXIS RANGES ---------------- #
# Boxplot figures
BOX_RATIO_YLIM = (0.8, 2.)
BOX_TM_YLIM = (1000, 1800)

# Scatter figures
SCATTER_RATIO_YLIM = (0.6, 2.2)
SCATTER_TM_YLIM = (700, 2100)

# Gap between WT and KO in boxplot figures only
BOX_GROUP_GAP = 1.5

# Scatter settings
N_POINTS_SELECT_PER_SAMPLE = 3000
FRACTION_TO_PLOT = 0.2          # 0.2 of 4000 = 800 plotted points/sample
SCATTER_ALPHA = 0.25
SCATTER_SIZE = 60
SCATTER_JITTER_WIDTH = 0.25


# ---------------- FILE READING ---------------- #
def read_pixel_data(file_path):
    """Read 2D pixel data from ASC and flatten."""
    try:
        data = np.loadtxt(file_path, skiprows=6)
        return data.flatten()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def find_files_with_pattern(sample_path, pattern):
    """Return list of ASC files in S_* folders, sorted numerically."""
    matching_files = []

    if not os.path.isdir(sample_path):
        print(f"Folder not found: {sample_path}")
        return matching_files

    subfolders = [
        os.path.join(sample_path, d)
        for d in os.listdir(sample_path)
        if os.path.isdir(os.path.join(sample_path, d)) and d.startswith("S_")
    ]

    def extract_index(name):
        try:
            return int(name.split("_")[1])
        except Exception:
            return 9999

    subfolders.sort(key=lambda x: extract_index(os.path.basename(x)))

    for folder in subfolders:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and f.endswith(pattern):
                matching_files.append(full)

    return matching_files


# ---------------- OUTLIER REMOVAL ---------------- #
def exclude_outliers_joint(a1, a2, tm=None, lower=2.5, upper=97.5):
    """Remove pixel-level outliers jointly across a1, a2, tm."""
    if a1.size == 0 or a2.size == 0:
        return a1, a2, tm

    lo_a1, hi_a1 = np.percentile(a1, [lower, upper])
    lo_a2, hi_a2 = np.percentile(a2, [lower, upper])

    mask = (a1 >= lo_a1) & (a1 <= hi_a1) & (a2 >= lo_a2) & (a2 <= hi_a2)

    if tm is not None and tm.size > 0:
        lo_tm, hi_tm = np.percentile(tm, [lower, upper])
        mask = mask & (tm >= lo_tm) & (tm <= hi_tm)
        return a1[mask], a2[mask], tm[mask]

    return a1[mask], a2[mask], tm


# ---------------- LOAD ONE SAMPLE ---------------- #
def load_sample_data(sample_path):
    """
    Load all a1, a2, tm files from one sample folder containing S_* folders.
    Concatenate them, remove NaNs, remove joint outliers, and return arrays.
    """
    a1_files = find_files_with_pattern(sample_path, "_740_a1.asc")
    a2_files = find_files_with_pattern(sample_path, "_740_a2.asc")
    tm_files = find_files_with_pattern(sample_path, "_740_color coded value.asc")

    print(f"\nProcessing: {sample_path}")
    print(f"Found a1 files: {len(a1_files)}")
    print(f"Found a2 files: {len(a2_files)}")
    print(f"Found tm files: {len(tm_files)}")

    n_pairs = min(len(a1_files), len(a2_files), len(tm_files))
    print(f"Using matched file sets: {n_pairs}")

    all_a1, all_a2, all_tm = [], [], []

    for i, (f1, f2, ftm) in enumerate(
        zip(a1_files[:n_pairs], a2_files[:n_pairs], tm_files[:n_pairs]),
        start=1
    ):
        a1 = read_pixel_data(f1)
        a2 = read_pixel_data(f2)
        tm = read_pixel_data(ftm)

        if a1 is None or a2 is None or tm is None:
            print(f"Skipping set {i}: file read failed")
            continue

        if not (len(a1) == len(a2) == len(tm)):
            print(
                f"Skipping set {i}: size mismatch -> "
                f"a1={len(a1)}, a2={len(a2)}, tm={len(tm)}"
            )
            continue

        valid = np.isfinite(a1) & np.isfinite(a2) & np.isfinite(tm)

        a1 = a1[valid]
        a2 = a2[valid]
        tm = tm[valid]

        if len(a1) == 0:
            print(f"Skipping set {i}: no valid pixels after NaN removal")
            continue

        all_a1.append(a1)
        all_a2.append(a2)
        all_tm.append(tm)

    if not all_a1:
        return np.array([]), np.array([]), np.array([])

    a1_all = np.concatenate(all_a1)
    a2_all = np.concatenate(all_a2)
    tm_all = np.concatenate(all_tm)

    print(f"Pixels before outlier removal: {len(a1_all)}")

    a1_all, a2_all, tm_all = exclude_outliers_joint(a1_all, a2_all, tm_all)

    print(f"Pixels after outlier removal: {len(a1_all)}")

    return a1_all, a2_all, tm_all


# ---------------- METRIC HELPERS ---------------- #
def clean_array(x):
    """Remove NaN and infinite values."""
    x = np.asarray(x)
    return x[np.isfinite(x)]


def compute_ratio_and_tm(a1, a2, tm):
    """
    Compute a1/a2 safely and return matched ratio and tm arrays.
    """
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    tm = np.asarray(tm)

    valid = (
        np.isfinite(a1)
        & np.isfinite(a2)
        & np.isfinite(tm)
        & (a2 != 0)
    )

    if np.sum(valid) == 0:
        return np.array([]), np.array([])

    ratio = a1[valid] / a2[valid]
    tm_valid = tm[valid]

    valid2 = np.isfinite(ratio) & np.isfinite(tm_valid)

    return ratio[valid2], tm_valid[valid2]


def summarize_one_sample(a1, a2, tm):
    """
    Return one summary value per sample.
    Median is better than mean because pixel distributions can be skewed.
    """
    ratio, tm_valid = compute_ratio_and_tm(a1, a2, tm)

    if len(ratio) == 0 or len(tm_valid) == 0:
        return {
            "ratio_median": np.nan,
            "ratio_mean": np.nan,
            "tm_median": np.nan,
            "tm_mean": np.nan,
            "n_pixels": 0,
        }

    return {
        "ratio_median": np.median(ratio),
        "ratio_mean": np.mean(ratio),
        "tm_median": np.median(tm_valid),
        "tm_mean": np.mean(tm_valid),
        "n_pixels": len(ratio),
    }


def get_sample_summaries(samples, group_name):
    """
    Convert each sample's pixel data into one sample-level summary.
    """
    summaries = []

    for i, (a1, a2, tm) in enumerate(samples, start=1):
        summary = summarize_one_sample(a1, a2, tm)
        summary["group"] = group_name
        summary["sample"] = f"{group_name} Sample {i}"
        summary["sample_index"] = i

        summaries.append(summary)

        print(
            f"{group_name} Sample {i}: "
            f"median a1/a2 = {summary['ratio_median']:.4f}, "
            f"median tm = {summary['tm_median']:.2f}, "
            f"pixels = {summary['n_pixels']}"
        )

    return summaries


def build_samplewise_pixel_arrays(wt_samples, ko_samples):
    """
    Build pixel-level ratio and tm arrays for each sample.
    Used for sample-wise boxplot visualization.
    """
    ratio_data = []
    tm_data = []
    labels = []
    face_colors = []
    edge_colors = []

    for i, (a1, a2, tm) in enumerate(wt_samples, start=1):
        ratio, tm_valid = compute_ratio_and_tm(a1, a2, tm)

        ratio_data.append(clean_array(ratio))
        tm_data.append(clean_array(tm_valid))

        labels.append(f"WT\nS{i}")
        face_colors.append(WT_LIGHT)
        edge_colors.append(WT_SHADE_COLORS[(i - 1) % len(WT_SHADE_COLORS)])

    for i, (a1, a2, tm) in enumerate(ko_samples, start=1):
        ratio, tm_valid = compute_ratio_and_tm(a1, a2, tm)

        ratio_data.append(clean_array(ratio))
        tm_data.append(clean_array(tm_valid))

        labels.append(f"KO\nS{i}")
        face_colors.append(KO_LIGHT)
        edge_colors.append(KO_SHADE_COLORS[(i - 1) % len(KO_SHADE_COLORS)])

    return ratio_data, tm_data, labels, face_colors, edge_colors


def randomly_select_fraction(values, rng):
    """
    First select up to N_POINTS_SELECT_PER_SAMPLE random points from one sample.
    Then plot only FRACTION_TO_PLOT of that selected set.
    """
    values = clean_array(values)

    if len(values) == 0:
        return np.array([])

    n_select = min(N_POINTS_SELECT_PER_SAMPLE, len(values))
    idx_select = rng.choice(len(values), size=n_select, replace=False)
    selected = values[idx_select]

    n_plot = max(1, int(np.floor(len(selected) * FRACTION_TO_PLOT)))
    idx_plot = rng.choice(len(selected), size=n_plot, replace=False)

    return selected[idx_plot]


# ---------------- COMMON STYLE ---------------- #
def style_axis(ax):
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.30)
    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    ax.tick_params(axis="both", labelsize=11, width=1.1)


# ---------------- FIGURE 1: SAMPLE-LEVEL BOXPLOT COMPARISON ---------------- #
def plot_group_summary(ax, wt_values, ko_values, ylabel, title, ylim=None):
    """
    Sample-level boxplot with gap between WT and KO.

    - colored box = IQR, 25th to 75th percentile
    - white line = group median
    - whisker bars = standard boxplot range within 1.5 × IQR
    - circles = individual sample medians
    - white diamond = group mean
    """
    rng = np.random.default_rng(RANDOM_SEED)

    wt_values = clean_array(wt_values)
    ko_values = clean_array(ko_values)

    wt_pos = 1
    ko_pos = 1 + BOX_GROUP_GAP + 1

    data = [wt_values, ko_values]
    positions = [wt_pos, ko_pos]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", linewidth=2.4),
        boxprops=dict(linewidth=1.8),
        whiskerprops=dict(linewidth=1.8),
        capprops=dict(linewidth=1.8),
    )

    box_colors = [WT_COLOR, KO_COLOR]

    for i, patch in enumerate(bp["boxes"]):
        color = box_colors[i]
        patch.set_facecolor(color)
        patch.set_alpha(0.50)
        patch.set_edgecolor(color)
        patch.set_linewidth(2.0)

    for i, whisker in enumerate(bp["whiskers"]):
        group_index = i // 2
        whisker.set_color(box_colors[group_index])
        whisker.set_linewidth(1.8)

    for i, cap in enumerate(bp["caps"]):
        group_index = i // 2
        cap.set_color(box_colors[group_index])
        cap.set_linewidth(1.8)

    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2.4)

    wt_x = wt_pos + rng.uniform(-0.08, 0.08, size=len(wt_values))
    ko_x = ko_pos + rng.uniform(-0.08, 0.08, size=len(ko_values))

    ax.scatter(
        wt_x,
        wt_values,
        s=85,
        color=WT_COLOR,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
        zorder=5
    )

    ax.scatter(
        ko_x,
        ko_values,
        s=85,
        color=KO_COLOR,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95,
        zorder=5
    )

    if len(wt_values) > 0:
        ax.scatter(
            wt_pos,
            np.mean(wt_values),
            marker="D",
            s=95,
            facecolor="white",
            edgecolor=WT_COLOR,
            linewidth=1.5,
            zorder=6
        )

    if len(ko_values) > 0:
        ax.scatter(
            ko_pos,
            np.mean(ko_values),
            marker="D",
            s=95,
            facecolor="white",
            edgecolor=KO_COLOR,
            linewidth=1.5,
            zorder=6
        )

    ax.set_xlim(wt_pos - 0.8, ko_pos + 0.8)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks([wt_pos, ko_pos])
    ax.set_xticklabels(["WT", "KO"], fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    style_axis(ax)


def plot_sample_trend(ax, wt_values, ko_values, ylabel, title, ylim=None):
    """
    Plot sample medians by sample number.
    """
    wt_values = clean_array(wt_values)
    ko_values = clean_array(ko_values)

    wt_x = np.arange(1, len(wt_values) + 1)
    ko_x = np.arange(1, len(ko_values) + 1)

    ax.plot(
        wt_x,
        wt_values,
        marker="o",
        markersize=7,
        linewidth=2,
        color=WT_COLOR,
        label="WT"
    )

    ax.plot(
        ko_x,
        ko_values,
        marker="o",
        markersize=7,
        linewidth=2,
        color=KO_COLOR,
        label="KO"
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xlabel("Sample number", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    max_n = max(len(wt_values), len(ko_values))
    ax.set_xticks(np.arange(1, max_n + 1))

    ax.legend(frameon=False, fontsize=11)

    style_axis(ax)


def save_sample_level_figure(wt_samples, ko_samples, save_dir, save_format="png", dpi=300):
    wt_summaries = get_sample_summaries(wt_samples, "WT")
    ko_summaries = get_sample_summaries(ko_samples, "KO")

    wt_ratio_medians = np.array([s["ratio_median"] for s in wt_summaries])
    ko_ratio_medians = np.array([s["ratio_median"] for s in ko_summaries])

    wt_tm_medians = np.array([s["tm_median"] for s in wt_summaries])
    ko_tm_medians = np.array([s["tm_median"] for s in ko_summaries])

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor("white")

    plot_group_summary(
        axs[0, 0],
        wt_ratio_medians,
        ko_ratio_medians,
        ylabel="Sample median a1/a2 ratio",
        title="Group Boxplot: a1/a2",
        ylim=BOX_RATIO_YLIM
    )

    plot_group_summary(
        axs[0, 1],
        wt_tm_medians,
        ko_tm_medians,
        ylabel="Sample median tm",
        title="Group Boxplot: tm",
        ylim=BOX_TM_YLIM
    )

    plot_sample_trend(
        axs[1, 0],
        wt_ratio_medians,
        ko_ratio_medians,
        ylabel="Sample median a1/a2 ratio",
        title="Sample Medians: a1/a2",
        ylim=BOX_RATIO_YLIM
    )

    plot_sample_trend(
        axs[1, 1],
        wt_tm_medians,
        ko_tm_medians,
        ylabel="Sample median tm",
        title="Sample Medians: tm",
        ylim=BOX_TM_YLIM
    )

    fig.suptitle(
        "WT vs KO FLIM Comparison Using Sample-Level Medians",
        fontsize=18,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"WT_vs_KO_sample_level_boxplot_with_gap.{save_format}"
    )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"\nSample-level boxplot figure saved to:\n{save_path}")

    plt.show()


# ---------------- FIGURE 2: SAMPLE-WISE PIXEL BOXPLOTS ---------------- #
def make_samplewise_iqr_boxplot(
    ax,
    data_list,
    labels,
    face_colors,
    edge_colors,
    title,
    ylabel,
    ylim=None,
    n_wt=0
):
    """
    Sample-wise pixel-level boxplots with gap between WT and KO.

    - colored box = IQR, 25th to 75th percentile
    - white line = median
    - whisker bars = standard boxplot range within 1.5 × IQR
    - white diamond = mean
    """
    cleaned_data = [clean_array(arr) for arr in data_list]

    n_total = len(cleaned_data)
    n_ko = n_total - n_wt

    wt_positions = list(np.arange(1, n_wt + 1))
    ko_start = n_wt + BOX_GROUP_GAP + 1
    ko_positions = list(np.arange(ko_start, ko_start + n_ko))

    positions = wt_positions + ko_positions

    bp = ax.boxplot(
        cleaned_data,
        positions=positions,
        patch_artist=True,
        labels=labels,
        showfliers=False,
        widths=0.58,
        medianprops=dict(color="white", linewidth=2.2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.6),
        capprops=dict(linewidth=1.6),
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(face_colors[i])
        patch.set_alpha(0.78)
        patch.set_edgecolor(edge_colors[i])
        patch.set_linewidth(1.8)

    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2.4)

    for i, whisker in enumerate(bp["whiskers"]):
        sample_index = i // 2
        whisker.set_color(edge_colors[sample_index])
        whisker.set_linewidth(1.6)

    for i, cap in enumerate(bp["caps"]):
        sample_index = i // 2
        cap.set_color(edge_colors[sample_index])
        cap.set_linewidth(1.6)

    # Add mean as white diamond with same sample-colored edge
    for pos, arr, color in zip(positions, cleaned_data, edge_colors):
        if len(arr) == 0:
            continue

        mean_val = np.mean(arr)

        ax.scatter(
            pos,
            mean_val,
            marker="D",
            s=52,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=6
        )

    # Add subtle vertical separator between WT and KO
    if n_wt > 0 and n_ko > 0:
        separator_x = (wt_positions[-1] + ko_positions[0]) / 2
        ax.axvline(
            separator_x,
            color="gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.35,
            zorder=1
        )

    ax.set_xlim(min(positions) - 0.8, max(positions) + 0.8)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold", rotation=25)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=12)

    style_axis(ax)


def save_samplewise_boxplot_figure(wt_samples, ko_samples, save_dir, save_format="png", dpi=300):
    ratio_data, tm_data, labels, face_colors, edge_colors = build_samplewise_pixel_arrays(
        wt_samples,
        ko_samples
    )

    n_wt = len(wt_samples)

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("white")

    make_samplewise_iqr_boxplot(
        axs[0],
        ratio_data,
        labels,
        face_colors,
        edge_colors,
        title="Sample-wise Boxplot: a1/a2 Ratio",
        ylabel="a1/a2 Ratio",
        ylim=BOX_RATIO_YLIM,
        n_wt=n_wt
    )

    make_samplewise_iqr_boxplot(
        axs[1],
        tm_data,
        labels,
        face_colors,
        edge_colors,
        title="Sample-wise Boxplot: tm",
        ylabel="tm",
        ylim=BOX_TM_YLIM,
        n_wt=n_wt
    )

    fig.suptitle(
        "WT vs KO Sample-wise FLIM Distributions",
        fontsize=18,
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"WT_vs_KO_samplewise_boxplots_with_gap.{save_format}"
    )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"\nSample-wise boxplot figure saved to:\n{save_path}")

    plt.show()


# ---------------- FIGURE 3: GROUP SCATTER ONLY — NO IQR ---------------- #
def build_group_scatter_arrays(wt_samples, ko_samples):
    """
    Build pooled WT and KO scatter arrays.

    Scatter only:
    - no IQR
    - no boxplot
    - black line = median only
    """
    rng = np.random.default_rng(RANDOM_SEED)

    wt_ratio_parts = []
    wt_tm_parts = []
    wt_ratio_color_parts = []
    wt_tm_color_parts = []

    ko_ratio_parts = []
    ko_tm_parts = []
    ko_ratio_color_parts = []
    ko_tm_color_parts = []

    # WT samples
    for i, (a1, a2, tm) in enumerate(wt_samples, start=1):
        ratio, tm_valid = compute_ratio_and_tm(a1, a2, tm)

        ratio_plot = randomly_select_fraction(ratio, rng)
        tm_plot = randomly_select_fraction(tm_valid, rng)

        wt_color = WT_SHADE_COLORS[(i - 1) % len(WT_SHADE_COLORS)]

        wt_ratio_parts.append(ratio_plot)
        wt_tm_parts.append(tm_plot)

        wt_ratio_color_parts.append(np.array([wt_color] * len(ratio_plot), dtype=object))
        wt_tm_color_parts.append(np.array([wt_color] * len(tm_plot), dtype=object))

    # KO samples
    for i, (a1, a2, tm) in enumerate(ko_samples, start=1):
        ratio, tm_valid = compute_ratio_and_tm(a1, a2, tm)

        ratio_plot = randomly_select_fraction(ratio, rng)
        tm_plot = randomly_select_fraction(tm_valid, rng)

        ko_color = KO_SHADE_COLORS[(i - 1) % len(KO_SHADE_COLORS)]

        ko_ratio_parts.append(ratio_plot)
        ko_tm_parts.append(tm_plot)

        ko_ratio_color_parts.append(np.array([ko_color] * len(ratio_plot), dtype=object))
        ko_tm_color_parts.append(np.array([ko_color] * len(tm_plot), dtype=object))

    wt_ratio = np.concatenate(wt_ratio_parts) if wt_ratio_parts else np.array([])
    wt_tm = np.concatenate(wt_tm_parts) if wt_tm_parts else np.array([])
    wt_ratio_colors = np.concatenate(wt_ratio_color_parts) if wt_ratio_color_parts else np.array([], dtype=object)
    wt_tm_colors = np.concatenate(wt_tm_color_parts) if wt_tm_color_parts else np.array([], dtype=object)

    ko_ratio = np.concatenate(ko_ratio_parts) if ko_ratio_parts else np.array([])
    ko_tm = np.concatenate(ko_tm_parts) if ko_tm_parts else np.array([])
    ko_ratio_colors = np.concatenate(ko_ratio_color_parts) if ko_ratio_color_parts else np.array([], dtype=object)
    ko_tm_colors = np.concatenate(ko_tm_color_parts) if ko_tm_color_parts else np.array([], dtype=object)

    print("\nGroup scatter-only figure point counts:")
    print(f"WT a1/a2 plotted points: {len(wt_ratio)}")
    print(f"KO a1/a2 plotted points: {len(ko_ratio)}")
    print(f"WT tm plotted points: {len(wt_tm)}")
    print(f"KO tm plotted points: {len(ko_tm)}")

    return (
        wt_ratio,
        wt_tm,
        wt_ratio_colors,
        wt_tm_colors,
        ko_ratio,
        ko_tm,
        ko_ratio_colors,
        ko_tm_colors,
    )


def make_group_scatter_x(values, center, rng, max_width=0.30):
    """Make cloudy x positions around WT or KO center."""
    values = np.asarray(values)

    if len(values) == 0:
        return np.array([])

    x = center + rng.normal(loc=0, scale=max_width / 2.5, size=len(values))
    x = np.clip(x, center - max_width, center + max_width)

    return x


def plot_group_scatter_only(ax, wt_values, ko_values, wt_colors, ko_colors, ylabel, title, ylim=None):
    """
    Plot WT vs KO scatter only.

    No IQR here.
    Black horizontal line = median only.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    wt_values = clean_array(wt_values)
    ko_values = clean_array(ko_values)

    wt_x = make_group_scatter_x(wt_values, center=1, rng=rng, max_width=SCATTER_JITTER_WIDTH)
    ko_x = make_group_scatter_x(ko_values, center=2, rng=rng, max_width=SCATTER_JITTER_WIDTH)

    ax.scatter(
        wt_x,
        wt_values,
        s=SCATTER_SIZE,
        c=wt_colors,
        alpha=SCATTER_ALPHA,
        edgecolors="none",
        zorder=3
    )

    ax.scatter(
        ko_x,
        ko_values,
        s=SCATTER_SIZE,
        c=ko_colors,
        alpha=SCATTER_ALPHA,
        edgecolors="none",
        zorder=3
    )

    # Median only, no IQR
    if len(wt_values) > 0:
        wt_median = np.median(wt_values)
        ax.plot(
            [0.78, 1.22],
            [wt_median, wt_median],
            color="black",
            linewidth=2.0,
            zorder=5
        )

    if len(ko_values) > 0:
        ko_median = np.median(ko_values)
        ax.plot(
            [1.78, 2.22],
            [ko_median, ko_median],
            color="black",
            linewidth=2.0,
            zorder=5
        )

    ax.set_xlim(0.45, 2.55)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["WT", "KO"], fontsize=13, fontweight="bold")

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    style_axis(ax)


def save_group_scatter_only_figure(wt_samples, ko_samples, save_dir, save_format="png", dpi=300):
    """
    Save separate group scatter-only figure:
    WT vs KO scatter only for a1/a2 and tm.
    """
    (
        wt_ratio,
        wt_tm,
        wt_ratio_colors,
        wt_tm_colors,
        ko_ratio,
        ko_tm,
        ko_ratio_colors,
        ko_tm_colors,
    ) = build_group_scatter_arrays(wt_samples, ko_samples)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor("white")

    plot_group_scatter_only(
        axs[0],
        wt_ratio,
        ko_ratio,
        wt_ratio_colors,
        ko_ratio_colors,
        ylabel="a1/a2 Ratio",
        title="Scatter Only: a1/a2 Ratio",
        ylim=SCATTER_RATIO_YLIM
    )

    plot_group_scatter_only(
        axs[1],
        wt_tm,
        ko_tm,
        wt_tm_colors,
        ko_tm_colors,
        ylabel="tm",
        title="Scatter Only: tm",
        ylim=SCATTER_TM_YLIM
    )

    fig.suptitle(
        "WT vs KO Scatter Plot",
        fontsize=17,
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"WT_vs_KO_scatter_only_scatterrange.{save_format}"
    )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"\nGroup scatter-only figure saved to:\n{save_path}")

    plt.show()


# ---------------- FIGURE 4: SAMPLE-WISE SCATTER ONLY — NO IQR ---------------- #
def build_samplewise_scatter_arrays(wt_samples, ko_samples):
    """
    Build sample-wise scatter arrays.

    Each sample plotted separately.
    Scatter only:
    - no IQR
    - no boxplot
    - black line = sample median only
    """
    ratio_data = []
    tm_data = []
    labels = []
    colors = []

    rng = np.random.default_rng(RANDOM_SEED)

    # WT samples
    for i, (a1, a2, tm) in enumerate(wt_samples, start=1):
        ratio, tm_valid = compute_ratio_and_tm(a1, a2, tm)

        ratio_plot = randomly_select_fraction(ratio, rng)
        tm_plot = randomly_select_fraction(tm_valid, rng)

        ratio_data.append(ratio_plot)
        tm_data.append(tm_plot)

        labels.append(f"WT\nS{i}")
        colors.append(WT_SHADE_COLORS[(i - 1) % len(WT_SHADE_COLORS)])

    # KO samples
    for i, (a1, a2, tm) in enumerate(ko_samples, start=1):
        ratio, tm_valid = compute_ratio_and_tm(a1, a2, tm)

        ratio_plot = randomly_select_fraction(ratio, rng)
        tm_plot = randomly_select_fraction(tm_valid, rng)

        ratio_data.append(ratio_plot)
        tm_data.append(tm_plot)

        labels.append(f"KO\nS{i}")
        colors.append(KO_SHADE_COLORS[(i - 1) % len(KO_SHADE_COLORS)])

    print("\nSample-wise scatter-only figure point counts:")
    for label, ratio_values, tm_values in zip(labels, ratio_data, tm_data):
        print(f"{label.replace(chr(10), ' ')}: ratio points = {len(ratio_values)}, tm points = {len(tm_values)}")

    return ratio_data, tm_data, labels, colors


def plot_samplewise_scatter_only(ax, data_list, labels, colors, ylabel, title, ylim=None):
    """
    Scatter-only plot for each individual sample.

    No IQR here.
    Black horizontal line = sample median only.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    for i, values in enumerate(data_list, start=1):
        values = clean_array(values)

        if len(values) == 0:
            continue

        x = i + rng.normal(
            loc=0,
            scale=SCATTER_JITTER_WIDTH / 2.5,
            size=len(values)
        )

        x = np.clip(
            x,
            i - SCATTER_JITTER_WIDTH,
            i + SCATTER_JITTER_WIDTH
        )

        ax.scatter(
            x,
            values,
            s=SCATTER_SIZE,
            color=colors[i - 1],
            alpha=SCATTER_ALPHA,
            edgecolors="none",
            zorder=3
        )

        # Median only, no IQR
        sample_median = np.median(values)

        ax.plot(
            [i - 0.22, i + 0.22],
            [sample_median, sample_median],
            color="black",
            linewidth=2.0,
            zorder=5
        )

    ax.set_xlim(0.4, len(labels) + 0.6)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold", rotation=25)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    style_axis(ax)


def save_samplewise_scatter_only_figure(wt_samples, ko_samples, save_dir, save_format="png", dpi=300):
    """
    Save separate sample-wise scatter-only figure.

    No IQR in this figure.
    """
    ratio_data, tm_data, labels, colors = build_samplewise_scatter_arrays(
        wt_samples,
        ko_samples
    )

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor("white")

    plot_samplewise_scatter_only(
        axs[0],
        ratio_data,
        labels,
        colors,
        ylabel="a1/a2 Ratio",
        title="Sample-wise Scatter: a1/a2 Ratio",
        ylim=SCATTER_RATIO_YLIM
    )

    plot_samplewise_scatter_only(
        axs[1],
        tm_data,
        labels,
        colors,
        ylabel="tm",
        title="Sample-wise Scatter: tm",
        ylim=SCATTER_TM_YLIM
    )

    fig.suptitle(
        "WT vs KO Sample-wise Scatter Plots",
        fontsize=18,
        fontweight="bold",
        y=1.03
    )

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"WT_vs_KO_samplewise_scatter_only_scatterrange.{save_format}"
    )

    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"\nSample-wise scatter-only figure saved to:\n{save_path}")

    plt.show()


# ---------------- MAIN ---------------- #
def main():
    wt_root = os.path.join(ROOT, "WT")
    ko_root = os.path.join(ROOT, "KO")

    wt_samples = []
    ko_samples = []

    # Load WT
    for sample_name in WT_SAMPLE_NAMES:
        sample_path = os.path.join(wt_root, sample_name)
        a1, a2, tm = load_sample_data(sample_path)

        if len(a1) == 0:
            print(f"Warning: WT {sample_name} has no valid data")

        wt_samples.append((a1, a2, tm))

    # Load KO
    for sample_name in KO_SAMPLE_NAMES:
        sample_path = os.path.join(ko_root, sample_name)
        a1, a2, tm = load_sample_data(sample_path)

        if len(a1) == 0:
            print(f"Warning: KO {sample_name} has no valid data")

        ko_samples.append((a1, a2, tm))

    print("\nLoaded sample sizes:")

    for i, (a1, a2, tm) in enumerate(wt_samples, start=1):
        print(f"WT Sample {i}: {len(a1)} pixels")

    for i, (a1, a2, tm) in enumerate(ko_samples, start=1):
        print(f"KO Sample {i}: {len(a1)} pixels")

    save_format = SAVE_FORMAT.lower().replace(".", "")

    # Figure 1: sample-level boxplot comparison with WT-KO gap
    save_sample_level_figure(
        wt_samples,
        ko_samples,
        save_dir=SAVE_DIR,
        save_format=save_format,
        dpi=DPI
    )

    # Figure 2: sample-wise boxplots with WT-KO gap
    save_samplewise_boxplot_figure(
        wt_samples,
        ko_samples,
        save_dir=SAVE_DIR,
        save_format=save_format,
        dpi=DPI
    )

    # Figure 3: pooled WT vs KO scatter-only figure, no IQR
    save_group_scatter_only_figure(
        wt_samples,
        ko_samples,
        save_dir=SAVE_DIR,
        save_format=save_format,
        dpi=DPI
    )

    # Figure 4: sample-wise scatter-only figure, no IQR
    save_samplewise_scatter_only_figure(
        wt_samples,
        ko_samples,
        save_dir=SAVE_DIR,
        save_format=save_format,
        dpi=DPI
    )


if __name__ == "__main__":
    main()