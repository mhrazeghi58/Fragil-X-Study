# -*- coding: utf-8 -*-
"""
FTIR WT vs KO – similarity + centroid leaning + max-T + ratio strip plots
CLEANED / DEDUPLICATED / PUB-STYLE SAVING

UPDATE (your request):
1) max-T now runs for BOTH:
   - RAW (masked-cube mean spectra)
   - 2nd-deriv (masked-cube mean spectra)
   and plots KO vs WT (KO − WT).

2) On the max-T plots:
   - highlight canonical FTIR bands (Amide, Lipid, etc.) as spans (band ± 15 cm-1)
     with colors + band names written on the plot (NO legend).
   - also highlight Top-15 significant *bands* (single wavenumber bins, max-T p*)
     with spans (±15 cm-1), colored by category if matched to a known band,
     otherwise a default color, and write the band name.

3) Still NO “region evaluation / peak expansion”; it’s band-only.

NOTE:
- Seaborn is kept (clustermap).
"""

from __future__ import annotations

import os
import logging, sys
from time import perf_counter
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter


# =============================================================================
# Logging & timing
# =============================================================================
logger = logging.getLogger("ftir")
logger.setLevel(logging.INFO)
if not logger.handlers:
    sh = logging.StreamHandler(sys.stdout)
    fh = logging.FileHandler("analysis.log", mode="a", encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh.setFormatter(fmt); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)


class LogTimer:
    def __init__(self, label, lvl=logging.INFO):
        self.label = label; self.lvl = lvl
    def __enter__(self):
        logger.log(self.lvl, f"▶ {self.label} ...")
        self.t0 = perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = timedelta(seconds=perf_counter() - self.t0)
        logger.log(self.lvl, f"✔ {self.label} done | elapsed: {dt}\n")


# =============================================================================
# Global constants
# =============================================================================
SEED   = 50
BPERM  = 5000

# Savitzky–Golay (2nd deriv in similarity and sample-mean)
SAVGOL_WINDOW = 11
POLYORDER     = 3

# Similarity subsampling of pixels per sample
SUBSAMPLE_SIZE = 2000


# =============================================================================
# PUB STYLE + SAVING
# =============================================================================
def apply_pub_style(font_family="Arial", base_fontsize=11, dpi_fig=300, dpi_save=600):
    plt.rcParams.update({
        "figure.dpi": dpi_fig,
        "savefig.dpi": dpi_save,
        "font.family": font_family,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.labelsize": base_fontsize + 1,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "mathtext.default": "regular",
    })


def save_figure(fig, out_base: str, dpi=600, save_pdf=True, tight=True):
    """
    out_base: path WITHOUT extension, e.g. r".../fig_name"
    Saves PNG + (optional) PDF.
    """
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    bbox = "tight" if tight else None
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches=bbox)
    if save_pdf:
        fig.savefig(out_base + ".pdf", bbox_inches=bbox)
    logger.info("Saved: %s(.png/.pdf)", out_base)


# =============================================================================
# Known FTIR bands (edit freely)
# =============================================================================
# "category" is used to color the spans consistently.
KNOWN_BANDS = [
    # Proteins
    {"name": "Amide I",        "center": 1655, "category": "Protein"},
    {"name": "Amide II",       "center": 1545, "category": "Protein"},
    {"name": "Amide III",      "center": 1240, "category": "Protein"},
    # Lipids
    {"name": "Lipid CH2 bend", "center": 1464, "category": "Lipid"},
    {"name": "Lipid CH3",      "center": 1375, "category": "Lipid"},
    {"name": "Ester C=O",      "center": 1740, "category": "Lipid"},
    # Nucleic acids / phosphate
    {"name": "PO2− asym",      "center": 1235, "category": "Phosphate"},
    {"name": "PO2− sym",       "center": 1080, "category": "Phosphate"},
    # Carbohydrates / glycogen-ish region (common assignments)
    {"name": "C–O stretch",    "center": 1045, "category": "Carbohydrate"},
    {"name": "C–O–C",          "center": 1155, "category": "Carbohydrate"},
]

CATEGORY_COLORS = {
    "Protein":       "#1f77b4",  # blue
    "Lipid":         "#ff7f0e",  # orange
    "Phosphate":     "#2ca02c",  # green
    "Carbohydrate":  "#9467bd",  # purple
    "Other":         "#7f7f7f",  # gray
}


# =============================================================================
# Utilities
# =============================================================================
def blend_colors(color1, color2):
    rgb1 = np.array(mcolors.to_rgb(color1))
    rgb2 = np.array(mcolors.to_rgb(color2))
    return tuple((rgb1 + rgb2) / 2)


def _orient_sign(spectra: np.ndarray, ref_vec: np.ndarray) -> np.ndarray:
    """
    Flip any spectrum whose cosine to ref is negative (sign-consistent derivatives).
    """
    if spectra.size == 0:
        return spectra
    ref_norm = np.linalg.norm(ref_vec)
    if not np.isfinite(ref_norm) or ref_norm == 0:
        return spectra
    ref = ref_vec / ref_norm
    dots = spectra @ ref
    flip = dots < 0
    if np.any(flip):
        spectra = spectra.copy()
        spectra[flip] *= -1.0
    return spectra


def _clean_selected_clusters(raw: dict, n_clusters=8) -> dict[str, list[int]]:
    """
    Clean cluster lists (e.g., '3.6' -> 3) and clamp to [0..n_clusters-1].
    """
    cleaned = {}
    for name, lst in raw.items():
        ints = []
        for x in lst:
            try:
                xi = int(x)
            except Exception:
                continue
            if 0 <= xi < n_clusters:
                ints.append(xi)
        cleaned[name] = sorted(set(ints))
    return cleaned


def match_known_band(wn_value: float, known_bands=KNOWN_BANDS, tol_cm1=15.0):
    """
    Returns a dict with {name, center, category} of the closest known band if within tol, else None.
    """
    best = None
    best_d = np.inf
    for b in known_bands:
        d = abs(float(wn_value) - float(b["center"]))
        if d < best_d:
            best_d = d
            best = b
    if best is not None and best_d <= tol_cm1:
        return best
    return None


# =============================================================================
# Similarity matrices (raw / 2nd-deriv, optional sign-fix + normalization)
# =============================================================================
def build_similarity_matrices(
    sample_paths: list[str],
    sample_names: list[str],
    groups: dict,
    selected_clusters: dict,
    second_derivative: bool = False,
    subsample_size: int = 2000,
    random_state: int = SEED,
    normalize_spectra: bool = True,
    fix_derivative_sign: bool = True,
    abs_cosine: bool = False,
    savgol_window: int = SAVGOL_WINDOW,
    polyorder: int = POLYORDER,
):
    """
    Returns:
      matrix: (n_samples x n_samples) mean cosine similarity matrix
      sims: dict of flattened similarities by pair-group
      sim_pairs: dict mapping group -> list of (name1, name2, [pairwise sims])
    """
    rng = np.random.default_rng(random_state)
    staged = {}

    # 1) Load -> optional 2nd deriv -> cluster select -> finite mask -> subsample
    for path, name in zip(sample_paths, sample_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        data = np.load(path)
        spec = data["spectra"]  # (n_pixels, n_bands)

        if second_derivative:
            spec = savgol_filter(spec, window_length=savgol_window, polyorder=polyorder, deriv=2, axis=1)

        clusters = selected_clusters.get(name, [])
        mask = np.isin(data["cluster_labels"], clusters)
        spec_sel = spec[mask]
        spec_sel = spec_sel[np.all(np.isfinite(spec_sel), axis=1)]

        if len(spec_sel) > subsample_size:
            idx = rng.choice(len(spec_sel), size=subsample_size, replace=False)
            spec_sel = spec_sel[idx]

        staged[name] = spec_sel

    # 2) Sign fix (only meaningful for derivatives)
    if fix_derivative_sign and second_derivative:
        all_for_ref = np.vstack([v for v in staged.values() if len(v) > 0]) if any(len(v) for v in staged.values()) else None
        ref_vec = None
        if all_for_ref is not None and all_for_ref.shape[0] > 0:
            ref_vec = np.median(all_for_ref, axis=0) if all_for_ref.shape[0] >= 5 else np.mean(all_for_ref, axis=0)

        if ref_vec is not None and np.all(np.isfinite(ref_vec)):
            for name, S in staged.items():
                if S.size:
                    staged[name] = _orient_sign(S, ref_vec)

    # 3) Optional L2 normalize per spectrum
    X = {}
    for name, S in staged.items():
        if normalize_spectra and len(S) > 0:
            S = normalize(S, axis=1)
        X[name] = S

    # 4) Similarities
    n = len(sample_names)
    matrix = np.full((n, n), np.nan, dtype=float)
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}
    sim_pairs = {"WT-WT": [], "KO-KO": [], "WT-KO": []}

    def grp(nm): return "WT" if nm in groups["WT"] else "KO"

    for i, name1 in enumerate(sample_names):
        for j, name2 in enumerate(sample_names):
            s1 = X[name1]; s2 = X[name2]
            if len(s1) == 0 or len(s2) == 0:
                continue
            sim_matrix = cosine_similarity(s1, s2)
            if abs_cosine:
                sim_matrix = np.abs(sim_matrix)
            sims_pair = sim_matrix.ravel().astype(np.float32)
            matrix[i, j] = float(np.nanmean(sims_pair))

            g1, g2 = grp(name1), grp(name2)
            key = f"{g1}-{g2}" if g1 == g2 else "WT-KO"
            sims[key].extend(sims_pair.tolist())
            sim_pairs[key].append((name1, name2, sims_pair.tolist()))

    return matrix, sims, sim_pairs


def pair_median_matrix(sim_pairs, sample_names):
    name_to_idx = {n: i for i, n in enumerate(sample_names)}
    n = len(sample_names)
    M = np.full((n, n), np.nan, dtype=float)

    for _, pairs in sim_pairs.items():
        for n1, n2, sims in pairs:
            if n1 not in name_to_idx or n2 not in name_to_idx:
                continue
            v = np.asarray(sims, float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            med = float(np.median(v))
            i, j = name_to_idx[n1], name_to_idx[n2]
            M[i, j] = med
            M[j, i] = med

    np.fill_diagonal(M, 1.0)
    return M


# =============================================================================
# Plot: similarity scatter with meta-hue + optional 2nd-deriv transform
# =============================================================================
def plot_meta_color_scatter(
    raw_sims_pairs,
    sec_sims_pairs,
    groups,
    out_base: str,
    fraction=0.6,
    random_state=SEED,
    figsize=(10, 9),
    font_family="Arial",
    base_fontsize=11,
    MAX_PER_PAIR=5000,
    MAX_GLOBAL=50_000_000,
    y_lim_raw=(-1.0, 1.0),
    y_lim_sec=(0.6, 1.0),
):
    """
    Produces 3 figures:
      - raw cosine scatter
      - 2nd-deriv cosine scatter (linear)
      - 2nd-deriv scatter using log(1-cosine) distance
    """
    rng = np.random.default_rng(random_state)
    mpl.rcParams["agg.path.chunksize"] = 15_000
    apply_pub_style(font_family=font_family, base_fontsize=base_fontsize)

    # per-sample colors
    wt_colors = sns.color_palette("Greens", n_colors=len(groups["WT"]))
    ko_colors = sns.color_palette("Reds",   n_colors=len(groups["KO"]))
    sample_color = {name: col for name, col in zip(groups["WT"], wt_colors)}
    sample_color.update({name: col for name, col in zip(groups["KO"], ko_colors)})

    def scatter_linear(ax, sim_pairs, title, y_lim):
        jitter_scale = 0.09
        marker_s = 40
        alpha = 0.05
        positions = {"WT-WT": 1, "KO-KO": 2, "WT-KO": 3}
        drawn = 0

        for key, pairs in sim_pairs.items():
            pos = positions[key]
            for n1, n2, sims_list in pairs:
                if drawn >= MAX_GLOBAL:
                    break
                sims = np.asarray(sims_list, dtype=np.float32)
                sims = sims[np.isfinite(sims)]
                if sims.size == 0:
                    continue

                subsample_n = max(10, int(sims.size * fraction))
                k = min(subsample_n, sims.size, MAX_PER_PAIR, MAX_GLOBAL - drawn)
                if k <= 0:
                    continue

                subsampled = rng.choice(sims, size=k, replace=False)
                x = rng.normal(pos, jitter_scale, size=k)

                c1 = sample_color.get(n1, (0.5, 0.5, 0.5))
                c2 = sample_color.get(n2, (0.5, 0.5, 0.5))
                color_pair = blend_colors(c1, c2)

                ax.scatter(
                    x, subsampled, color=color_pair, alpha=alpha, s=marker_s,
                    linewidths=0, edgecolors="none", marker=".",
                    rasterized=True,
                )
                drawn += k

        ax.set_xlim(0.4, 3.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(*y_lim)
        ax.set_title(title)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def scatter_logdist(ax, sim_pairs, title, y_lim_cos):
        """
        Plot d = (1 - cosine) on log scale.
        """
        jitter_scale = 0.10
        marker_s = 40
        alpha = 0.06
        positions = {"WT-WT": 1, "KO-KO": 2, "WT-KO": 3}
        drawn = 0
        eps = 2.5e-4

        for key, pairs in sim_pairs.items():
            pos = positions[key]
            for n1, n2, sims_list in pairs:
                if drawn >= MAX_GLOBAL:
                    break
                sims = np.asarray(sims_list, dtype=np.float32)
                sims = sims[np.isfinite(sims)]
                if sims.size == 0:
                    continue

                subsample_n = max(10, int(sims.size * fraction))
                k = min(subsample_n, sims.size, MAX_PER_PAIR, MAX_GLOBAL - drawn)
                if k <= 0:
                    continue

                subsampled = rng.choice(sims, size=k, replace=False)
                d = np.maximum(1.0 - subsampled, eps)
                x = rng.normal(pos, jitter_scale, size=k)

                c1 = sample_color.get(n1, (0.5, 0.5, 0.5))
                c2 = sample_color.get(n2, (0.5, 0.5, 0.5))
                color_pair = blend_colors(c1, c2)

                ax.scatter(
                    x, d, color=color_pair, alpha=alpha, s=marker_s,
                    linewidths=0, edgecolors="none", marker=".",
                    rasterized=True,
                )
                drawn += k

        ax.set_xlim(0.4, 3.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
        ax.set_ylabel("1 − Cosine similarity (log scale)")
        ax.set_yscale("log")

        lo, hi = y_lim_cos
        d_min = max(eps, 1.0 - hi)
        d_max = max(eps, 1.0 - lo)
        ax.set_ylim(d_min, d_max)
        ax.invert_yaxis()

        ax.set_title(title)
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # RAW
    fig, ax = plt.subplots(figsize=figsize)
    scatter_linear(ax, raw_sims_pairs, "Raw Spectra Similarities (Meta-hue)", y_lim_raw)
    fig.tight_layout()
    save_figure(fig, out_base + "_raw_meta", dpi=600, save_pdf=True)
    plt.close(fig)

    # SEC (linear)
    fig, ax = plt.subplots(figsize=figsize)
    scatter_linear(ax, sec_sims_pairs, "2nd Derivative Similarities (Linear)", y_lim_sec)
    fig.tight_layout()
    save_figure(fig, out_base + "_sec_meta_linear", dpi=600, save_pdf=True)
    plt.close(fig)

    # SEC (log distance)
    fig, ax = plt.subplots(figsize=figsize)
    scatter_logdist(ax, sec_sims_pairs, "2nd Derivative Similarities (1 − cosine, log)", y_lim_sec)
    fig.tight_layout()
    save_figure(fig, out_base + "_sec_meta_logdist", dpi=600, save_pdf=True)
    plt.close(fig)


# =============================================================================
# Plot: similarity heatmaps
# =============================================================================
def plot_similarity_heatmap(M, sample_names, groups, title, out_base, cluster=False, cmap="vlag"):
    apply_pub_style()

    M = np.asarray(M, float)
    names = list(sample_names)

    # --- ensure symmetric + diagonal ---
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 1.0)

    # --- identify valid samples (enough finite entries in row) ---
    finite_frac = np.mean(np.isfinite(M), axis=1)  # fraction finite per row
    keep = finite_frac >= 0.80                      # keep rows with >=80% finite
    if cluster and np.sum(keep) < 3:
        logger.warning("Not enough valid samples for clustering (kept=%d). Skipping clustered heatmap.", int(np.sum(keep)))
        return

    if cluster and not np.all(keep):
        dropped = [names[i] for i in np.where(~keep)[0]]
        logger.warning("Dropping samples with too many NaNs for clustering: %s", dropped)

        idx = np.where(keep)[0]
        M = M[np.ix_(idx, idx)]
        names = [names[i] for i in idx]

    # --- fill remaining NaNs with median (must be finite for clustermap) ---
    med = np.nanmedian(M[np.isfinite(M)]) if np.any(np.isfinite(M)) else 0.0
    M = np.where(np.isfinite(M), M, med)

    # keep within sensible bounds
    M = np.clip(M, -1.0, 1.0)

    group_label = ["WT" if n in groups["WT"] else "KO" for n in names]
    lut = {"WT": "#2ca02c", "KO": "#d62728"}

    if cluster:
        cg = sns.clustermap(
            M, method="average", metric="euclidean", cmap=cmap,
            center=np.median(M),
            xticklabels=names, yticklabels=names,
            row_colors=[lut[g] for g in group_label],
            col_colors=[lut[g] for g in group_label],
            linewidths=0.3, figsize=(9, 8)
        )
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90)
        cg.ax_heatmap.set_title(title)
        cg.savefig(out_base + "_clustered_heatmap.png", dpi=600, bbox_inches="tight")
        cg.savefig(out_base + "_clustered_heatmap.pdf", bbox_inches="tight")
        plt.close(cg.fig)
        logger.info("Saved: %s(_clustered_heatmap.png/.pdf)", out_base)
        return

    # (optional: you can implement a non-cluster heatmap here if you want)


# =============================================================================
# Signed 2nd-deriv matrices + centroid leaning
# =============================================================================
def build_signed_secderiv_matrices(
    sample_paths,
    sample_names,
    selected_clusters,
    subsample_size=SUBSAMPLE_SIZE,
    random_state=SEED,
    do_normalize=True,
    savgol_window=SAVGOL_WINDOW,
    polyorder=POLYORDER,
):
    rng = np.random.default_rng(random_state)

    staged = {}
    for path, name in zip(sample_paths, sample_names):
        data = np.load(path)
        spec = data["spectra"]
        spec = savgol_filter(spec, window_length=savgol_window, polyorder=polyorder, deriv=2, axis=1)

        mask = np.isin(data["cluster_labels"], selected_clusters.get(name, []))
        spec = spec[mask]
        spec = spec[np.all(np.isfinite(spec), axis=1)]

        if len(spec) > subsample_size:
            spec = spec[rng.choice(len(spec), size=subsample_size, replace=False)]
        staged[name] = spec

    all_for_ref = np.vstack([v for v in staged.values() if len(v) > 0]) if any(len(v) for v in staged.values()) else None
    if all_for_ref is None or all_for_ref.shape[0] == 0:
        return {name: staged[name] for name in sample_names}

    ref_vec = np.median(all_for_ref, axis=0) if all_for_ref.shape[0] >= 5 else np.mean(all_for_ref, axis=0)
    ref = ref_vec / (np.linalg.norm(ref_vec) + 1e-12)

    X = {}
    for name, S in staged.items():
        if S.size == 0:
            X[name] = S
            continue
        S = S.copy()
        dots = S @ ref
        S[dots < 0] *= -1.0
        if do_normalize:
            S = normalize(S, axis=1)
        X[name] = S
    return X


def centroid_leaning_barplot(X, sample_names, groups, out_base):
    def mean_spec(S):
        return np.nanmean(S, axis=0)

    WT_means = [mean_spec(X[n]) for n in sample_names if n in groups["WT"] and X[n].size]
    KO_means = [mean_spec(X[n]) for n in sample_names if n in groups["KO"] and X[n].size]
    WT_centroid = np.nanmean(WT_means, axis=0)
    KO_centroid = np.nanmean(KO_means, axis=0)

    def cos(a, b):
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return np.nan
        return float((a @ b) / (na * nb))

    scores = {}
    for n in sample_names:
        if X[n].size == 0:
            scores[n] = (np.nan, np.nan)
            continue
        m = mean_spec(X[n])
        scores[n] = (cos(m, WT_centroid), cos(m, KO_centroid))

    # angles for margin
    ang = {}
    for n, (cwt, cko) in scores.items():
        if not np.isfinite(cwt) or not np.isfinite(cko):
            ang[n] = (np.nan, np.nan, np.nan)
            continue
        th_wt = np.arccos(np.clip(cwt, -1, 1))
        th_ko = np.arccos(np.clip(cko, -1, 1))
        ang[n] = (th_wt * 1e3, th_ko * 1e3, (th_ko - th_wt) * 1e3)

    bar = sorted(
        [(n, ("WT" if n in groups["WT"] else "KO"), ang[n][2]) for n in sample_names if np.isfinite(ang[n][2])],
        key=lambda r: r[2],
        reverse=True
    )

    apply_pub_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(bar))
    margins = [mm for _, _, mm in bar]
    cols = ["#2ca02c" if g == "WT" else "#d62728" for _, g, _ in bar]
    ax.axhline(0, ls="--", lw=1)
    ax.bar(x, margins, color=cols, edgecolor="k", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([n for n, _, _ in bar], rotation=90)
    ax.set_ylabel("Angle margin (mrad): θKO − θWT\n(+ ⇒ WT-like, − ⇒ KO-like)")
    ax.set_title("Centroid leaning (2nd-deriv, sign-corrected)")
    fig.tight_layout()
    save_figure(fig, out_base + "_centroid_margins", dpi=600, save_pdf=True)
    plt.close(fig)

    KO_like = [n for n in groups["KO"] if np.isfinite(ang[n][2]) and ang[n][2] < 0]
    WT_like = [n for n in groups["KO"] if np.isfinite(ang[n][2]) and ang[n][2] >= 0]
    logger.info("KO-A (KO-like): %s", KO_like)
    logger.info("KO-B (WT-like KO): %s", WT_like)
    return KO_like, WT_like, ang


# =============================================================================
# Masked-cube sample mean spectra (RAW and 2nd-deriv)
# =============================================================================
def _load_cluster_and_cube(sample, cluster_dir, cube_dir):
    cluster_file = os.path.join(cluster_dir, f"{sample}_umap_kmeans.npz")
    cube_file = os.path.join(cube_dir, f"masked_cube_{sample}.npz")
    if not os.path.exists(cluster_file) or not os.path.exists(cube_file):
        logger.warning("Missing files for %s", sample)
        return None

    clust = np.load(cluster_file)
    labels = clust["cluster_labels"]
    indices = clust["pixel_indices"]

    cube = np.load(cube_file)["data"]  # (Y, X, n_bands)
    return labels, indices, cube


def _flat_indices_from_pixel_indices(indices, Xdim):
    if indices.ndim == 2 and indices.shape[1] == 2:
        rows = indices[:, 0]
        cols = indices[:, 1]
        return rows * Xdim + cols
    return indices


def _ensure_expected_bands(cube, expected_bands, sample):
    Y, X, Z = cube.shape
    if Z < expected_bands:
        missing = expected_bands - Z
        logger.warning("%s: missing %d bands – padding with edge values at the beginning", sample, missing)
        edge = cube[:, :, :1]
        pad = np.repeat(edge, missing, axis=2)
        cube = np.concatenate([pad, cube], axis=2)
        Z = expected_bands
    elif Z > expected_bands:
        logger.warning("%s: has %d extra bands – trimming from start", sample, Z - expected_bands)
        cube = cube[:, :, -expected_bands:]
        Z = expected_bands
    return cube


def get_sample_mean_raw(sample, cluster_dir, cube_dir, selected_clusters,
                        expected_bands=426, wn_min=950, wn_max=1800):
    out = _load_cluster_and_cube(sample, cluster_dir, cube_dir)
    if out is None:
        return None
    labels, indices, cube = out
    cube = _ensure_expected_bands(cube, expected_bands, sample)
    Y, X, Z = cube.shape

    sel_clusters = selected_clusters.get(sample, [])
    if len(sel_clusters) == 0:
        logger.warning("No selected clusters for %s", sample)
        return None

    flat_indices = _flat_indices_from_pixel_indices(indices, X)
    use = np.isin(labels, sel_clusters)
    selected_indices = flat_indices[use]
    if selected_indices.size == 0:
        logger.warning("No pixels in selected clusters for %s", sample)
        return None

    flat_cube = cube.reshape(-1, Z)
    pix = flat_cube[selected_indices, :]
    pix = pix[np.all(np.isfinite(pix), axis=1)]
    if pix.size == 0:
        logger.warning("No valid pixels for %s", sample)
        return None

    mean_spec = np.nanmean(pix, axis=0)
    wn = np.linspace(wn_min, wn_max, expected_bands)
    return mean_spec, wn


def get_sample_mean_2nd_deriv(sample, cluster_dir, cube_dir, selected_clusters,
                             expected_bands=426, wn_min=950, wn_max=1800,
                             savgol_window=SAVGOL_WINDOW, polyorder=POLYORDER):
    out = _load_cluster_and_cube(sample, cluster_dir, cube_dir)
    if out is None:
        return None
    labels, indices, cube = out
    cube = _ensure_expected_bands(cube, expected_bands, sample)
    Y, X, Z = cube.shape

    sel_clusters = selected_clusters.get(sample, [])
    if len(sel_clusters) == 0:
        logger.warning("No selected clusters for %s", sample)
        return None

    flat_indices = _flat_indices_from_pixel_indices(indices, X)
    use = np.isin(labels, sel_clusters)
    selected_indices = flat_indices[use]
    if selected_indices.size == 0:
        logger.warning("No pixels in selected clusters for %s", sample)
        return None

    flat_cube = cube.reshape(-1, Z)
    pix = flat_cube[selected_indices, :]
    pix = pix[np.all(np.isfinite(pix), axis=1)]
    if pix.size == 0:
        logger.warning("No valid pixels for %s", sample)
        return None

    sec_deriv = savgol_filter(pix, window_length=savgol_window,
                              polyorder=polyorder, deriv=2, axis=1)
    mean_spec = np.nanmean(sec_deriv, axis=0)
    wn = np.linspace(wn_min, wn_max, expected_bands)
    return mean_spec, wn


# =============================================================================
# max-T permutation
# =============================================================================
def maxT_permutation(groupA, groupB, B=5000, strata=None, stat="diff", rng_seed=SEED):
    rng = np.random.default_rng(rng_seed)
    X = np.vstack([groupA, groupB])
    nA, nB = groupA.shape[0], groupB.shape[0]
    idxA = np.arange(nA)
    idxB = np.arange(nA, nA + nB)

    def compute_stat(A, B):
        if stat == "welch":
            mA, mB = A.mean(0), B.mean(0)
            vA, vB = A.var(0, ddof=1), B.var(0, ddof=1)
            denom = np.sqrt(vA / A.shape[0] + vB / B.shape[0])
            denom[denom == 0] = np.nan
            return (mA - mB) / denom
        return A.mean(0) - B.mean(0)

    obs = compute_stat(X[idxA], X[idxB])

    if strata is None:
        strata = np.zeros(X.shape[0], dtype=int)
    unique_batches = np.unique(strata)

    max_stats = np.empty(B, dtype=float)
    for b in range(B):
        perm_idxA = []
        perm_idxB = []
        for bt in unique_batches:
            in_batch = np.where(strata == bt)[0]
            nA_bt = np.sum(np.isin(in_batch, idxA))
            perm = rng.permutation(in_batch)
            perm_idxA.extend(perm[:nA_bt])
            perm_idxB.extend(perm[nA_bt:])
        Aperm = X[np.array(perm_idxA)]
        Bperm = X[np.array(perm_idxB)]
        stat_perm = compute_stat(Aperm, Bperm)
        max_stats[b] = np.nanmax(np.abs(stat_perm))

    p_maxT = (np.sum(max_stats[:, None] >= np.abs(obs)[None, :], axis=0) + 1) / (B + 1)
    return obs, p_maxT


# =============================================================================
# Top-k significant *bands* + plot with colored spans + band names (NO legend)
# =============================================================================
def top_k_bands(pvals, k=15, min_sep=1):
    """
    Pick top-k most significant *single bands* (smallest pvals).
    min_sep: minimum index separation between picked bands (1 allows adjacent).
    Returns list of indices.
    """
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    if not np.any(ok):
        return []

    order = np.argsort(p[ok])
    cand = np.where(ok)[0][order]

    picked = []
    for i in cand:
        if any(abs(int(i) - j) < int(min_sep) for j in picked):
            continue
        picked.append(int(i))
        if len(picked) == k:
            break
    return picked


def _place_band_label(ax, x, text, y_frac=0.98, rotation=90, fontsize=8):
    """
    Place label in axes coords for stable layout (no legend).
    """
    ymin, ymax = ax.get_ylim()
    y = ymin + (ymax - ymin) * y_frac
    ax.text(x, y, text, rotation=rotation, va="top", ha="center",
            fontsize=fontsize, clip_on=True)


def plot_maxT_topbands_with_known(
    wn, obs, pvals,
    title: str,
    out_base: str,
    k_top=15,
    halfwidth_cm1=15.0,
    min_sep=1,
    known_bands=KNOWN_BANDS,
    tol_match_cm1=15.0,
    alpha_known=0.12,
    alpha_top=0.25,
):
    """
    1) Plot obs (difference curve)
    2) Shade known bands (±halfwidth) with category color, label their names
    3) Shade top-k bands (±halfwidth), colored by matched category if possible,
       label the band name (known if matched else "Top @ ####").
    NO legend.
    Returns list of dicts: {wn, p, idx, name, category}.
    """
    apply_pub_style()

    wn = np.asarray(wn, float)
    obs = np.asarray(obs, float)
    pvals = np.asarray(pvals, float)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(wn, obs, lw=1)
    ax.axhline(0, ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Mean diff (KO − WT)")
    ax.set_title(title)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Known bands first (lighter) ---
    # To avoid clutter, only label if within wn range.
    wn_min, wn_max = float(np.min(wn)), float(np.max(wn))
    for b in known_bands:
        c = float(b["center"])
        if c < wn_min or c > wn_max:
            continue
        cat = b.get("category", "Other")
        col = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["Other"])
        ax.axvspan(c - halfwidth_cm1, c + halfwidth_cm1, alpha=alpha_known, color=col, lw=0)
        _place_band_label(ax, c, b["name"], y_frac=0.98, rotation=90, fontsize=8)

    # --- Top-k bands (stronger) ---
    top_idx = top_k_bands(pvals, k=k_top, min_sep=min_sep)
    top_info = []
    for idx in top_idx:
        x = float(wn[idx])
        p = float(pvals[idx])
        hit = match_known_band(x, known_bands=known_bands, tol_cm1=tol_match_cm1)

        if hit is not None:
            name = hit["name"]
            cat = hit.get("category", "Other")
        else:
            name = f"Top @ {x:.0f}"
            cat = "Other"

        col = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["Other"])
        ax.axvspan(x - halfwidth_cm1, x + halfwidth_cm1, alpha=alpha_top, color=col, lw=0)
        _place_band_label(ax, x, name, y_frac=0.80, rotation=90, fontsize=8)

        top_info.append({"idx": int(idx), "wn": x, "p": p, "name": name, "category": cat})

    fig.tight_layout()
    save_figure(fig, out_base, dpi=600, save_pdf=True)
    plt.close(fig)
    return top_info


# =============================================================================
# Ratio strip plots (same as before; unchanged)
# =============================================================================
def ratio_stripplots(
    sample_names, groups,
    cluster_dir, cube_dir, selected_clusters,
    out_dir: str,
    mode: str = "raw",
    expected_bands: int = 426,
    wn_min=950, wn_max=1800,
    band_halfwidth=25,
    sec_smooth_win=7,
    sec_flip_sign=True,
    max_points_per_group=30_000,
    max_points_per_sample=2_500,
    dot_alpha=0.08,
    dot_size=6,
    jitter_scale=0.06,
    use_violin_background=True,
    trim_for_plotting_only=True,
    fixed_ylim_ch2ch3=None,
    fixed_ylim_ai_aii=None,
    use_log_y_ai=False,
    force_nonneg_bands=True,
    denom_min_pct=3.0,
    num_min_pct=5.0,
    denom_abs_floor=None,
    num_abs_floor=None,
    ratio_hard_cap=10.0,
    save_pdf=True,
    seed=SEED,
):
    mode = mode.lower().strip()
    assert mode in ("raw", "secderiv")

    # mode-specific defaults
    if denom_abs_floor is None:
        denom_abs_floor = 1e-2 if mode == "raw" else 1e-6
    if num_abs_floor is None:
        num_abs_floor = 1e-5 if mode == "raw" else 1e-7

    selected_clusters = _clean_selected_clusters(selected_clusters, n_clusters=8)
    os.makedirs(out_dir, exist_ok=True)

    def load_cluster(sample):
        f = os.path.join(cluster_dir, f"{sample}_umap_kmeans.npz")
        if not os.path.exists(f):
            logger.warning("Missing cluster file for %s", sample)
            return None
        z = np.load(f)
        return z["cluster_labels"], z["pixel_indices"]

    def load_cube(sample):
        f = os.path.join(cube_dir, f"masked_cube_{sample}.npz")
        if not os.path.exists(f):
            logger.warning("Missing cube file for %s", sample)
            return None
        cube = np.load(f)["data"]  # (H,W,Z)
        H, W, Z = cube.shape
        if Z < expected_bands:
            missing = expected_bands - Z
            logger.warning("%s: missing %d bands – padding with front edge values", sample, missing)
            edge = cube[:, :, :1]
            pad = np.repeat(edge, missing, axis=2)
            cube = np.concatenate([pad, cube], axis=2)
        elif Z > expected_bands:
            logger.warning("%s: has %d extra bands – trimming from start", sample, Z - expected_bands)
            cube = cube[:, :, -expected_bands:]
        return cube

    def selected_mask(H, W, labels, indices, sel_lst):
        if indices.ndim == 2 and indices.shape[1] == 2:
            rows = indices[:, 0]; cols = indices[:, 1]
            use = np.isin(labels, sel_lst)
            r = rows[use]; c = cols[use]
            m = np.zeros((H, W), dtype=bool)
            m[r, c] = True
            return m
        else:
            flat_idx = indices
            use = np.isin(labels, sel_lst)
            flat_use = flat_idx[use]
            m = np.zeros((H * W,), dtype=bool)
            m[flat_use] = True
            return m.reshape(H, W)

    def smooth_along_spectrum(cube, win=7):
        win = max(3, int(win) | 1)
        k = np.ones(win, dtype=np.float32) / float(win)
        pad = win // 2
        cube_pad = np.pad(cube, ((0, 0), (0, 0), (pad, pad)), mode="reflect")
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis=2, arr=cube_pad)
        return out

    def second_deriv_cube(cube, wns, smooth_win=7, flip_sign=True):
        sm = smooth_along_spectrum(cube, win=smooth_win)
        dnu = np.median(np.diff(wns))
        d1 = np.gradient(sm, dnu, axis=2)
        d2 = np.gradient(d1, dnu, axis=2)
        return (-d2 if flip_sign else d2)

    def band_value(cube_like, wns, center, halfwidth):
        m = (wns >= (center - halfwidth)) & (wns <= (center + halfwidth))
        if not np.any(m):
            H, W = cube_like.shape[:2]
            return np.full((H, W), np.nan, dtype=float)
        out = np.nanmean(cube_like[:, :, m], axis=2)
        if force_nonneg_bands:
            out = np.clip(out, 0, None)
        return out

    def denom_guard(den):
        d = den[np.isfinite(den)]
        if d.size == 0:
            return np.full_like(den, False, dtype=bool), denom_abs_floor
        thr = max(np.nanpercentile(d, denom_min_pct), denom_abs_floor)
        return (den > thr), thr

    def num_guard(num):
        d = num[np.isfinite(num)]
        if d.size == 0:
            return np.full_like(num, False, dtype=bool), num_abs_floor
        thr = max(np.nanpercentile(d, num_min_pct), num_abs_floor)
        return (num > thr), thr

    def collect_ratios_for_sample(sample):
        out = load_cluster(sample)
        if out is None:
            return None, None
        labels, indices = out

        cube_raw = load_cube(sample)
        if cube_raw is None:
            return None, None
        H, W, Z = cube_raw.shape
        wns = np.linspace(wn_min, wn_max, Z)

        sel = selected_clusters.get(sample, [])
        if len(sel) == 0:
            logger.warning("No selected clusters for %s", sample)
            return None, None

        mask2d = selected_mask(H, W, labels, indices, sel)
        if not np.any(mask2d):
            logger.warning("No pixels in selected clusters for %s", sample)
            return None, None

        cube = cube_raw
        if mode == "secderiv":
            cube = second_deriv_cube(cube_raw, wns, smooth_win=sec_smooth_win, flip_sign=sec_flip_sign)

        ch2    = band_value(cube, wns, 1464, band_halfwidth)
        ch3    = band_value(cube, wns, 1375, band_halfwidth)
        amideI = band_value(cube, wns, 1655, band_halfwidth)
        amideII= band_value(cube, wns, 1545, band_halfwidth)

        vR1_den, thr_ch3 = denom_guard(ch3)
        nR1_num, thr_ch2 = num_guard(ch2)
        vR2_den, thr_aii = denom_guard(amideII)
        nR2_num, thr_ai  = num_guard(amideI)

        logger.info("[%s] %s guards: CH2>%.3g & CH3>%.3g ; AmideI>%.3g & AmideII>%.3g",
                    mode, sample, thr_ch2, thr_ch3, thr_ai, thr_aii)

        valid1 = np.isfinite(ch2) & np.isfinite(ch3) & nR1_num & vR1_den
        valid2 = np.isfinite(amideI) & np.isfinite(amideII) & nR2_num & vR2_den

        r1 = np.full_like(ch2, np.nan, dtype=float)
        r2 = np.full_like(amideI, np.nan, dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            r1[valid1] = ch2[valid1] / ch3[valid1]
            r2[valid2] = amideI[valid2] / amideII[valid2]

        r1 = np.clip(r1, 0, ratio_hard_cap)
        r2 = np.clip(r2, 0, ratio_hard_cap)

        r1 = r1[mask2d].ravel()
        r2 = r2[mask2d].ravel()

        ok = np.isfinite(r1) & np.isfinite(r2) & (r1 > 0) & (r2 > 0)
        r1, r2 = r1[ok], r2[ok]

        if len(r1) > max_points_per_sample:
            rng = np.random.default_rng(abs(hash(sample)) % (2**32))
            take = rng.choice(len(r1), size=max_points_per_sample, replace=False)
            r1, r2 = r1[take], r2[take]

        return r1, r2

    def concat_cap(lst, cap, seed_local):
        if not lst:
            return np.array([])
        arr = np.concatenate(lst)
        if len(arr) <= cap:
            return arr
        rng = np.random.default_rng(seed_local)
        return arr[rng.choice(len(arr), size=cap, replace=False)]

    # gather
    WT_r1, WT_r2, KO_r1, KO_r2 = [], [], [], []
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    with LogTimer(f"Collect per-pixel ratios ({mode})"):
        for s in sample_names:
            g = sample_to_group.get(s)
            if g is None:
                continue
            r1, r2 = collect_ratios_for_sample(s)
            if r1 is None:
                continue
            if g == "WT":
                WT_r1.append(r1); WT_r2.append(r2)
            else:
                KO_r1.append(r1); KO_r2.append(r2)

    WT_ch2ch3 = concat_cap(WT_r1, max_points_per_group, seed_local=1)
    WT_ai_aii = concat_cap(WT_r2, max_points_per_group, seed_local=2)
    KO_ch2ch3 = concat_cap(KO_r1, max_points_per_group, seed_local=3)
    KO_ai_aii = concat_cap(KO_r2, max_points_per_group, seed_local=4)

    def auto_ylim(a, b, log=False, pad_frac=0.01):
        vals = np.r_[a, b]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        lo, hi = np.nanpercentile(vals, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return None
        pad = (hi - lo) * pad_frac
        lo2, hi2 = lo - pad, hi + pad
        if log:
            lo2 = max(lo2, 1e-6)
        return (lo2, hi2)

    def mask_ylim(arr, ylim):
        if ylim is None:
            return arr
        lo, hi = ylim
        return arr[(arr >= lo) & (arr <= hi) & np.isfinite(arr)]

    def stripplot(WT, KO, ylabel, title, out_base, log_y=False, ylim=None):
        apply_pub_style()
        if len(WT) == 0 or len(KO) == 0:
            logger.warning("Nothing to plot for %s", title)
            return

        WTv, KOv = WT, KO
        if trim_for_plotting_only and ylim is not None:
            WTv = mask_ylim(WT, ylim)
            KOv = mask_ylim(KO, ylim)

        fig, ax = plt.subplots(figsize=(11, 4))
        rng = np.random.default_rng(seed)

        colors = {"WT": "#2ca02c", "KO": "#d62728"}

        if use_violin_background:
            parts = ax.violinplot([WTv, KOv], positions=[0, 1], widths=0.6,
                                  showmeans=False, showmedians=False, showextrema=False)
            for i, b in enumerate(parts["bodies"]):
                b.set_facecolor(colors["WT"] if i == 0 else colors["KO"])
                b.set_alpha(0.18)
                b.set_edgecolor("none")

        x_wt = 0 + rng.normal(0, jitter_scale, size=len(WTv))
        x_ko = 1 + rng.normal(0, jitter_scale, size=len(KOv))
        ax.scatter(x_wt, WTv, s=dot_size, alpha=dot_alpha, c=colors["WT"], edgecolors="none", rasterized=True)
        ax.scatter(x_ko, KOv, s=dot_size, alpha=dot_alpha, c=colors["KO"], edgecolors="none", rasterized=True)

        def median_whiskers(x, xpos):
            if len(x) == 0:
                return
            med = np.median(x)
            q1, q3 = np.quantile(x, [0.05, 0.95])
            ax.scatter([xpos], [med], s=36, c="white", edgecolors="black", linewidths=0.7, zorder=3)
            ax.vlines(xpos, q1, q3, colors="black", linewidth=1.2, zorder=3)

        median_whiskers(WTv, 0)
        median_whiskers(KOv, 1)

        ax.set_xticks([0, 1], ["WT", "KO"])
        ax.set_ylabel(ylabel)
        if log_y:
            ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(max(ylim[0], 1e-6), ylim[1])
        elif ylim is not None:
            ax.set_ylim(*ylim)

        ax.set_title(title)
        ax.yaxis.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        save_figure(fig, out_base, dpi=600, save_pdf=save_pdf)
        plt.close(fig)

    ylim_ch2ch3 = fixed_ylim_ch2ch3 or auto_ylim(WT_ch2ch3, KO_ch2ch3, log=False)
    ylim_ai_aii = fixed_ylim_ai_aii or auto_ylim(WT_ai_aii, KO_ai_aii, log=use_log_y_ai)

    suffix = "raw" if mode == "raw" else "secderiv"
    stripplot(
        WT_ch2ch3, KO_ch2ch3,
        ylabel=f"CH$_2$/CH$_3$ ratio (per pixel; {suffix})",
        title=f"CH$_2$/CH$_3$ Ratio – WT vs KO ({suffix})",
        out_base=os.path.join(out_dir, f"ratios_CH2_CH3_strip_{suffix}"),
        log_y=False,
        ylim=ylim_ch2ch3
    )
    stripplot(
        WT_ai_aii, KO_ai_aii,
        ylabel=f"Amide I / Amide II ratio (per pixel; {suffix})",
        title=f"Amide I / Amide II Ratio – WT vs KO ({suffix})",
        out_base=os.path.join(out_dir, f"ratios_AmideI_AmideII_strip_{suffix}"),
        log_y=use_log_y_ai,
        ylim=ylim_ai_aii
    )

    logger.info("Saved ratio strip plots (%s) into %s", suffix, out_dir)
    return (WT_ch2ch3, KO_ch2ch3), (WT_ai_aii, KO_ai_aii)


# =============================================================================
# MAIN
# =============================================================================
def main():
    # ------------------------------ #
    # Paths
    # ------------------------------ #
    input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"
    save_dir = os.path.join(input_dir, "UMAP_clustering_test")
    os.makedirs(save_dir, exist_ok=True)
    out_prefix = os.path.join(save_dir, "similarity_scatter_pub1")

    # Similarity NPZ files
    sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                    "T10","T11","T12","T13","T14","T15","T22"]
    sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

    groups = {
        "WT": ["T1","T3","T6","T17","T19","T20","T21"],
        "KO": ["T10","T11","T12","T13","T14","T15","T22"]
    }

    selected_clusters = {
        "T1" :[0,1,3,4,6,7],
        "T2" : [0,1,2],
        "T3" : [0,1,4,5],
        "T19": [0,2,3.6,7],
        "T20": [0,1,2,4,5,7],
        "T21": [1,2,3,4],
        "T17": [0,1,2,3,5,6],
        "T10": [0,2,4,5,6,7],
        "T11": [0,1,2,3,6,7],
        "T12": [0,1,2,3,5,7],
        "T13": [1,2,5,6,7],
        "T14": [0,1,4,5,7],
        "T15": [0,2,3,4,5,7],
        "T22": [0,1,4,5,6],
    }
    selected_clusters = _clean_selected_clusters(selected_clusters, n_clusters=8)

    # ------------------------------ #
    # Similarity matrices
    # ------------------------------ #
    with LogTimer("Compute similarity matrices (raw & 2nd-deriv)"):
        raw_matrix, raw_sims, raw_pairs = build_similarity_matrices(
            sample_paths, sample_names, groups, selected_clusters,
            second_derivative=False,
            subsample_size=SUBSAMPLE_SIZE,
            normalize_spectra=True,
            fix_derivative_sign=False,
            abs_cosine=False,
        )
        sec_matrix, sec_sims, sec_pairs = build_similarity_matrices(
            sample_paths, sample_names, groups, selected_clusters,
            second_derivative=True,
            subsample_size=SUBSAMPLE_SIZE,
            normalize_spectra=True,
            fix_derivative_sign=True,
            abs_cosine=False,
        )
        logger.info("Shapes: raw=%s | sec=%s", raw_matrix.shape, sec_matrix.shape)

    # ------------------------------ #
    # Signed 2nd-deriv centroids + leaning
    # ------------------------------ #
    with LogTimer("Build signed sec-deriv matrices & centroid leaning"):
        X = build_signed_secderiv_matrices(sample_paths, sample_names, selected_clusters)
        KO_like, WT_like, ang = centroid_leaning_barplot(X, sample_names, groups, out_prefix)

    # ------------------------------ #
    # Meta-hue scatter plots
    # ------------------------------ #
    with LogTimer("Meta-hue scatter plots (pub saved)"):
        plot_meta_color_scatter(
            raw_pairs, sec_pairs, groups,
            out_base=out_prefix,
            fraction=0.6,
            random_state=SEED,
            figsize=(10, 9),
            font_family="Arial",
            base_fontsize=11,
            MAX_PER_PAIR=5000,
            MAX_GLOBAL=50_000_000,
            y_lim_raw=(-1.0, 1.0),
            y_lim_sec=(0.6, 1.0),
        )

    # ------------------------------ #
    # Heatmaps
    # ------------------------------ #
    with LogTimer("Heatmaps (median pairwise similarities)"):
        M_raw = pair_median_matrix(raw_pairs, sample_names)
        M_sec = pair_median_matrix(sec_pairs, sample_names)

        plot_similarity_heatmap(
            M_raw, sample_names, groups,
            title="Raw Spectra – Pairwise Median Similarities",
            out_base=out_prefix + "_raw",
            cluster=True
        )
        plot_similarity_heatmap(
            M_sec, sample_names, groups,
            title="2nd Derivative – Pairwise Median Similarities",
            out_base=out_prefix + "_sec",
            cluster=True
        )

    # ------------------------------ #
    # RAW + 2nd-deriv means from cubes + max-T (KO − WT)
    # ------------------------------ #
    cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L\UMAP_clustering_8Cluster"
    cube_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
    plot_dir    = os.path.join(cube_dir, "WT_KO_pubplot1")
    os.makedirs(plot_dir, exist_ok=True)

    # build sample mean spectra for BOTH modes
    with LogTimer("Build sample means (RAW + 2nd-deriv)"):
        means_raw = {}
        means_sec = {}
        wn_shared = None

        for n in sample_names:
            outR = get_sample_mean_raw(n, cluster_dir, cube_dir, selected_clusters)
            if outR is not None:
                mR, wnR = outR
                means_raw[n] = mR
                if wn_shared is None:
                    wn_shared = wnR

            outS = get_sample_mean_2nd_deriv(n, cluster_dir, cube_dir, selected_clusters)
            if outS is not None:
                mS, wnS = outS
                means_sec[n] = mS
                if wn_shared is None:
                    wn_shared = wnS

        logger.info("Built RAW means for %d samples; SEC means for %d samples",
                    len(means_raw), len(means_sec))

    def stack_from(means_dict, names):
        arr = [means_dict[n] for n in names if n in means_dict and np.all(np.isfinite(means_dict[n]))]
        return np.vstack(arr) if len(arr) >= 1 else None

    # KO-vs-WT: A=KO, B=WT so obs = mean(KO) - mean(WT)
    WT_raw = stack_from(means_raw, groups["WT"])
    KO_raw = stack_from(means_raw, groups["KO"])

    WT_sec = stack_from(means_sec, groups["WT"])
    KO_sec = stack_from(means_sec, groups["KO"])

    # Also optional splits (KO-A / KO-B) from centroid leaning (secderiv-based split)
    KOA_raw = stack_from(means_raw, KO_like)
    KOB_raw = stack_from(means_raw, WT_like)

    KOA_sec = stack_from(means_sec, KO_like)
    KOB_sec = stack_from(means_sec, WT_like)

    comparisons = [
        # RAW
        ("RAW: KO − WT",         KO_raw, WT_raw, os.path.join(plot_dir, "maxT_RAW_KO_minus_WT_top15bands")),
        ("RAW: KO-A − WT",       KOA_raw, WT_raw, os.path.join(plot_dir, "maxT_RAW_KOA_minus_WT_top15bands")),
        ("RAW: KO-B − WT",       KOB_raw, WT_raw, os.path.join(plot_dir, "maxT_RAW_KOB_minus_WT_top15bands")),
        # 2nd deriv
        ("2nd-deriv: KO − WT",   KO_sec, WT_sec, os.path.join(plot_dir, "maxT_SEC_KO_minus_WT_top15bands")),
        ("2nd-deriv: KO-A − WT", KOA_sec, WT_sec, os.path.join(plot_dir, "maxT_SEC_KOA_minus_WT_top15bands")),
        ("2nd-deriv: KO-B − WT", KOB_sec, WT_sec, os.path.join(plot_dir, "maxT_SEC_KOB_minus_WT_top15bands")),
    ]

    for title, A, B, out_base in comparisons:
        if wn_shared is None:
            logger.warning("Skipping '%s' (no wavenumber axis).", title)
            continue
        if A is None or B is None or A.shape[0] < 2 or B.shape[0] < 2:
            logger.warning("Skipping '%s' (insufficient samples): A=%s B=%s",
                           title, None if A is None else A.shape, None if B is None else B.shape)
            continue

        with LogTimer(f"Permutation test: {title} (B={BPERM})"):
            obs, p = maxT_permutation(A, B, B=BPERM, stat="diff", rng_seed=SEED)

            top_info = plot_maxT_topbands_with_known(
                wn_shared, obs, p,
                title=title + " | Top-15 bands + known FTIR bands (±15 cm⁻¹)",
                out_base=out_base,
                k_top=15,
                halfwidth_cm1=15.0,
                min_sep=1,
                known_bands=KNOWN_BANDS,
                tol_match_cm1=15.0,
            )

            for i, d in enumerate(top_info, 1):
                logger.info("Top%02d: %7.1f cm^-1 | p*=%.3e | %s | %s",
                            i, d["wn"], d["p"], d["name"], d["category"])

    # ------------------------------ #
    # Ratio strip plots (RAW + 2nd-deriv)
    # ------------------------------ #
    with LogTimer("Ratio strip plots (RAW)"):
        ratio_stripplots(
            sample_names, groups,
            cluster_dir=cluster_dir, cube_dir=cube_dir, selected_clusters=selected_clusters,
            out_dir=plot_dir,
            mode="raw",
            band_halfwidth=30,
            max_points_per_group=30_000,
            max_points_per_sample=2_500,
            fixed_ylim_ch2ch3=(0.5, 2.25),
            fixed_ylim_ai_aii=(0.8, 2.5),
            denom_min_pct=3.0, num_min_pct=5.0,
            denom_abs_floor=1e-2, num_abs_floor=1e-5,
        )

    with LogTimer("Ratio strip plots (2nd-deriv)"):
        ratio_stripplots(
            sample_names, groups,
            cluster_dir=cluster_dir, cube_dir=cube_dir, selected_clusters=selected_clusters,
            out_dir=plot_dir,
            mode="secderiv",
            band_halfwidth=25,
            max_points_per_group=40_000,
            max_points_per_sample=5_500,
            fixed_ylim_ch2ch3=None,
            fixed_ylim_ai_aii=None,
            denom_min_pct=1.0, num_min_pct=1.0,
            denom_abs_floor=1e-6, num_abs_floor=1e-7,
            sec_smooth_win=7,
            sec_flip_sign=True,
        )


if __name__ == "__main__":
    main()
