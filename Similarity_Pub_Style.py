# -*- coding: utf-8 -*-
"""
Similarity-only analysis (FAST) — FULL SPECTRA + SELECTED BANDS (separate results)

This version runs TWO separate similarity pipelines:
1) FULL_SPECTRA:
   - Uses the full spectral vector from each selected pixel
2) SELECTED_BANDS_ONLY:
   - Uses only the FTIR band windows you specified

Both branches run separately and save separate:
- Cliff's delta / MWU CSVs
- cosine scatter plots
- log10(1-cos) scatter plots
- heatmaps

Optional preprocessing:
- background subtraction before similarity calculation
- optional 2nd derivative branch in each analysis

Notes:
- Similarity is cosine similarity on either the full spectrum or the reduced band-window vector.
- For band-only mode, the windows are concatenated into one reduced feature vector.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu

# ------------------------------ #
# Paths
# ------------------------------ #
BASE_DIR = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"
umap_dir = os.path.join(BASE_DIR, "UMAP_clustering_test")
save_dir = os.path.join(BASE_DIR, "Similarity_Right")
os.makedirs(save_dir, exist_ok=True)

# Background subtraction
APPLY_BG_SUBTRACTION = True
SKIP_BG_IF_MISSING = True
bg_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\Area_Right_B\BG_First300ROI_MeanSpectrum"
BG_PATTERNS = [
    "bg_first300_mean_spectrum_{sid}.npz",
    "bg_first30_mean_spectrum_{sid}.npz",
    "bg_mean_spectrum_{sid}.npz",
    "bg_{sid}.npz",
]
DROP_BG_FIRST_5_WHEN_426_TO_421 = True

# Spectral axis assumption (for band-only slicing)
ASSUME_SAMPLE_MIN = 960.0
ASSUME_SAMPLE_MAX = 1800.0

# Savitzky–Golay (2nd-deriv) params
savgol_window = 15
polyorder = 3

# Performance knobs
subsample_size = 2000
max_pairs_dist = 40000
rng_master_seed = 10
EPS_DIST = 1e-6
SAVE_PDF_ALSO = True

# ------------------------------ #
# Heatmap display choices
# ------------------------------ #
HEATMAP_VMIN = 0.97
HEATMAP_VMAX = 1.00
HEATMAP_TICKS = [0.97, 0.98, 0.99, 1.00]
HEATMAP_CMAP = "Reds"

# ------------------------------ #
# Group colors (WT/KO)
# ------------------------------ #
BASE_WT = "#2ca02c"
BASE_KO = "#d62728"
GROUP_LUT = {"WT": BASE_WT, "KO": BASE_KO}

# ------------------------------ #
# Pub-style
# ------------------------------ #
PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11
FS = {
    "base": BASE_FONTSIZE,
    "axes": BASE_FONTSIZE + 1,
    "title": BASE_FONTSIZE + 3,
    "tick": BASE_FONTSIZE,
    "legend": BASE_FONTSIZE,
}
DOT_SIZE = 7
DOT_ALPHA = 0.35
GRID_ALPHA = 0.30
GRID_LW = 0.6
SPINE_LW = 1.0

# ------------------------------ #
# Selected FTIR bands to keep in BAND-ONLY mode
# ------------------------------ #
window = 30
BAND_1734_CENTER    = 1734.0
BAND_1734_HALFWIDTH = 13
BAND_AMIDE_I        = 1655.0
BAND_AMIDE_II       = 1545.0
BAND_CH2            = 1464.0
BAND_CH3            = 1375.0
BAND_PO2_SYM        = 1080.0
BAND_PO2_ASYM       = 1235.0
BAND_CARB_1030      = 1030.0
BAND_CARB_1155      = 1155.0
BAND_CARB_HALFWIDTH = 15
BAND_PO2_HALFWIDTH  = 15

BAND_LIBRARY = {
    "1734":       (BAND_1734_CENTER, BAND_1734_HALFWIDTH),
    "amideI":     (BAND_AMIDE_I,     window),
    "amideII":    (BAND_AMIDE_II,    window),
    "ch2":        (BAND_CH2,         window),
    "ch3":        (BAND_CH3,         window),
    "po2_1080":   (BAND_PO2_SYM,     BAND_PO2_HALFWIDTH),
    "po2_1235":   (BAND_PO2_ASYM,    BAND_PO2_HALFWIDTH),
    "carb_1030":  (BAND_CARB_1030,   BAND_CARB_HALFWIDTH),
    "carb_1155":  (BAND_CARB_1155,   BAND_CARB_HALFWIDTH),
}


def apply_pub_style():
    fam = FONT_FAMILY or "DejaVu Sans"
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": PUB_DPI,
        "font.family": fam,
        "font.size": FS["base"],
        "axes.labelsize": FS["axes"],
        "axes.titlesize": FS["title"],
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "legend.fontsize": FS["legend"],
        "axes.linewidth": SPINE_LW,
        "grid.alpha": GRID_ALPHA,
        "grid.linestyle": "--",
        "grid.linewidth": GRID_LW,
        "mathtext.default": "regular",
    })


def _savefig(fig, path_prefix):
    fig.savefig(path_prefix + ".png", bbox_inches="tight", dpi=PUB_DPI)
    if SAVE_PDF_ALSO:
        fig.savefig(path_prefix + ".pdf", bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)


def blend_colors(color1, color2):
    rgb1 = np.array(mcolors.to_rgb(color1))
    rgb2 = np.array(mcolors.to_rgb(color2))
    return tuple((rgb1 + rgb2) / 2)


def _clean_selected_clusters(raw_map, n_clusters=8):
    out = {}
    for name, vals in raw_map.items():
        ints = []
        for v in vals:
            try:
                vi = int(v)
            except Exception:
                continue
            if 0 <= vi < n_clusters:
                ints.append(vi)
        out[name] = sorted(set(ints))
    return out


def _orient_sign(spectra, ref_vec):
    if spectra.size == 0:
        return spectra
    ref_norm = np.linalg.norm(ref_vec)
    if not np.isfinite(ref_norm) or ref_norm == 0:
        return spectra
    ref = ref_vec / ref_norm
    dots = spectra @ ref
    flip = dots < 0
    if np.any(flip):
        spectra[flip] *= -1.0
    return spectra


def balance_sims(sims_dict, random_state=50):
    rng = np.random.default_rng(random_state)
    nonempty = [len(v) for v in sims_dict.values() if len(v) > 0]
    if len(nonempty) == 0:
        return {k: np.asarray(v) for k, v in sims_dict.items()}
    min_len = min(nonempty)
    balanced = {}
    for k, v in sims_dict.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            balanced[k] = v
        elif v.size > min_len:
            balanced[k] = rng.choice(v, size=min_len, replace=False)
        else:
            balanced[k] = v
    return balanced

# ------------------------------ #
# Background subtraction helpers
# ------------------------------ #
def _find_bg_file(sample):
    for pat in BG_PATTERNS:
        p = os.path.join(bg_dir, pat.format(sid=sample))
        if os.path.exists(p):
            return p
    return None


def _load_bg_vector(sample):
    p = _find_bg_file(sample)
    if p is None:
        return None, None
    z = np.load(p, allow_pickle=True)
    for k in z.files:
        a = np.asarray(z[k])
        if np.issubdtype(a.dtype, np.number) and a.ndim == 1 and a.size > 50:
            return a.astype(np.float64).ravel(), p
    raise ValueError(f"{sample}: BG file has no 1D numeric vector. keys={list(z.files)}")


def _align_bg_to_Z(bg, Z_src):
    bg = np.asarray(bg, dtype=np.float64).ravel()
    if bg.size == Z_src:
        return bg
    if DROP_BG_FIRST_5_WHEN_426_TO_421 and (bg.size == 426 and Z_src == 421):
        bg2 = bg[5:]
        if bg2.size >= Z_src:
            return bg2[:Z_src]
        out = np.empty(Z_src, float)
        out[:bg2.size] = bg2
        out[bg2.size:] = bg2[-1]
        return out
    L = min(bg.size, Z_src)
    out = np.empty(Z_src, dtype=np.float64)
    out[:L] = bg[:L]
    if L < Z_src:
        out[L:] = out[L-1]
    return out


def _apply_bg_subtraction(spec_sel, sample):
    if not APPLY_BG_SUBTRACTION:
        return spec_sel
    bg, _ = _load_bg_vector(sample)
    if bg is None:
        if SKIP_BG_IF_MISSING:
            print(f"Warning: BG missing for {sample} -> BG subtraction skipped.")
            return spec_sel
        raise FileNotFoundError(f"BG missing for {sample} in {bg_dir}")
    bgZ = _align_bg_to_Z(bg, spec_sel.shape[1])
    return spec_sel.astype(np.float64, copy=False) - bgZ[None, :]

# ---------- Cliff's delta helpers ----------
def cliffs_delta_1d(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    a_sort = np.sort(a)
    b_sort = np.sort(b)
    less = np.searchsorted(b_sort, a_sort, side='left')
    greater = (b_sort.size - np.searchsorted(b_sort, a_sort, side='right'))
    d = (less.sum() - greater.sum()) / (a_sort.size * b_sort.size)
    return float(d)


def bootstrap_cliffs_ci(a, b, n_boot=2000, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, (np.nan, np.nan)
    obs = cliffs_delta_1d(a, b)
    if np.isnan(obs):
        return np.nan, (np.nan, np.nan)
    boots = np.empty(n_boot, float)
    na, nb = a.size, b.size
    for i in range(n_boot):
        aa = a[rng.integers(0, na, na)]
        bb = b[rng.integers(0, nb, nb)]
        boots[i] = cliffs_delta_1d(aa, bb)
    lo = np.quantile(boots, (1-ci)/2)
    hi = np.quantile(boots, 1-(1-ci)/2)
    return obs, (lo, hi)


def effect_bin(di):
    if not np.isfinite(di):
        return "nan"
    ad = abs(di)
    if ad < 0.147:
        lab = "negligible"
    elif ad < 0.33:
        lab = "small"
    elif ad < 0.474:
        lab = "medium"
    else:
        lab = "large"
    arrow = "A>B" if di > 0 else ("A<B" if di < 0 else "A≈B")
    return f"{lab} ({arrow})"

# ---------- Fast math helpers ----------
def _unit_row_normalize(X):
    if X.size == 0:
        return X.astype(np.float32)
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def _sample_repr(X):
    if X.size == 0:
        return None
    return _unit_row_normalize(X).mean(axis=0).astype(np.float32, copy=False)


def sample_pairwise_cosines(A, B, max_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    if A.size == 0 or B.size == 0:
        return np.array([], dtype=np.float32)
    na, nb = A.shape[0], B.shape[0]
    P = min(max_pairs, na * nb)
    ia = rng.integers(0, na, P, dtype=np.int32)
    ib = rng.integers(0, nb, P, dtype=np.int32)
    return np.einsum('ij,ij->i', A[ia], B[ib], dtype=np.float32)

# ------------------------------ #
# Band-only reduction
# ------------------------------ #
def _build_band_feature_matrix(X):
    if X.size == 0:
        return X
    Z = X.shape[1]
    wns = np.linspace(ASSUME_SAMPLE_MIN, ASSUME_SAMPLE_MAX, Z)
    chunks = []
    for _, (center, halfwidth) in BAND_LIBRARY.items():
        m = (wns >= (center - halfwidth)) & (wns <= (center + halfwidth))
        if np.any(m):
            chunks.append(X[:, m])
    if len(chunks) == 0:
        return np.empty((X.shape[0], 0), dtype=np.float32)
    return np.concatenate(chunks, axis=1).astype(np.float32, copy=False)

# ------------------------------ #
# Core similarity builder
# ------------------------------ #
def build_similarity_matrices_custom(
    sample_paths,
    sample_names,
    groups,
    selected_clusters,
    feature_mode="full_spectra",
    second_derivative=False,
    subsample_size=2000,
    random_state=50,
    normalize_spectra=True,
    fix_derivative_sign=True,
    abs_cosine=False,
    max_pairs_dist=100000,
    exclude_self_from_distributions=True,
):
    rng = np.random.default_rng(random_state)
    staged = {}
    spectra_all = {}

    for path, name in zip(sample_paths, sample_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        data = np.load(path, mmap_mode='r')
        spec = data["spectra"]
        labels = data["cluster_labels"]

        clusters = selected_clusters.get(name, [])
        if not clusters:
            staged[name] = np.empty((0, spec.shape[1]), dtype=np.float32)
            continue

        mask = np.isin(labels, clusters)
        spec_sel = spec[mask]
        if spec_sel.size == 0:
            staged[name] = np.empty((0, spec.shape[1]), dtype=np.float32)
            continue

        spec_sel = spec_sel[np.all(np.isfinite(spec_sel), axis=1)]

        if len(spec_sel) > subsample_size:
            idx = rng.choice(len(spec_sel), size=subsample_size, replace=False)
            spec_sel = spec_sel[idx]

        spec_sel = _apply_bg_subtraction(spec_sel, name)
        spec_sel = np.asarray(spec_sel, dtype=np.float32, order='C')

        if second_derivative and spec_sel.size:
            spec_sel = savgol_filter(
                spec_sel, window_length=savgol_window,
                polyorder=polyorder, deriv=2, axis=1
            )

        if feature_mode == "band_only":
            spec_sel = _build_band_feature_matrix(spec_sel)
        elif feature_mode != "full_spectra":
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        staged[name] = np.asarray(spec_sel, dtype=np.float32, order='C')

    if fix_derivative_sign and second_derivative:
        nonempty = [v for v in staged.values() if v.size > 0]
        if len(nonempty) > 0:
            all_for_ref = np.vstack(nonempty)
            ref_vec = np.median(all_for_ref, axis=0) if all_for_ref.shape[0] >= 5 else np.mean(all_for_ref, axis=0)
            if np.all(np.isfinite(ref_vec)):
                for name, spec_sel in staged.items():
                    if spec_sel.size:
                        staged[name] = _orient_sign(spec_sel.copy(), ref_vec)

    for name, spec_sel in staged.items():
        if normalize_spectra and spec_sel.size:
            spec_sel = _unit_row_normalize(spec_sel)
        spectra_all[name] = spec_sel

    repr_vecs = {name: _sample_repr(spectra_all[name]) for name in sample_names}

    n = len(sample_names)
    matrix = np.full((n, n), np.nan, dtype=np.float32)
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}
    sim_pairs = {"WT-WT": [], "KO-KO": [], "WT-KO": []}

    for i, name1 in enumerate(sample_names):
        A = spectra_all[name1]
        v1 = repr_vecs[name1]
        if A.size == 0 or v1 is None:
            continue

        for j in range(i, n):
            name2 = sample_names[j]
            B = spectra_all[name2]
            v2 = repr_vecs[name2]
            if B.size == 0 or v2 is None:
                continue

            mean_ij = float(np.dot(v1, v2))
            matrix[i, j] = matrix[j, i] = mean_ij

            if exclude_self_from_distributions and (i == j):
                continue

            sims_pair = sample_pairwise_cosines(
                A, B, max_pairs=max_pairs_dist,
                seed=(hash((name1, name2, random_state, feature_mode, second_derivative)) & 0xffffffff)
            )
            if abs_cosine:
                sims_pair = np.abs(sims_pair)

            g1 = "WT" if name1 in groups["WT"] else "KO"
            g2 = "WT" if name2 in groups["WT"] else "KO"
            key = f"{g1}-{g2}" if g1 == g2 else "WT-KO"
            sims[key].extend(sims_pair.tolist())
            sim_pairs[key].append((name1, name2, sims_pair.tolist()))

    return matrix, sample_names, sims, sim_pairs

# ------------------------------ #
# Stats summaries
# ------------------------------ #
def summarize_delta(title, sims_dict, transform=None, balance=True, seed=123, csv_path=None):
    transform = transform or (lambda x: x)
    dct = balance_sims(sims_dict, random_state=seed) if balance else sims_dict
    comps = [("WT-WT", "KO-KO"), ("WT-WT", "WT-KO"), ("KO-KO", "WT-KO")]
    rows = []
    print(f"\n=== Cliff's delta + Mann–Whitney: {title} ===")

    for A, B in comps:
        a = np.asarray(dct.get(A, []), float); a = a[np.isfinite(a)]
        b = np.asarray(dct.get(B, []), float); b = b[np.isfinite(b)]
        a = transform(a); b = transform(b)
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        b = np.asarray(b, float); b = b[np.isfinite(b)]

        if a.size < 2 or b.size < 2:
            msg = f"{A} vs {B}: insufficient data (|A|={a.size}, |B|={b.size})"
            print(msg)
            rows.append([A, B, a.size, b.size, np.nan, np.nan, np.nan, np.nan, msg])
            continue

        d, (lo, hi) = bootstrap_cliffs_ci(a, b, n_boot=2000, seed=seed, ci=0.95)
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        except TypeError:
            _, p = mannwhitneyu(a, b, alternative="two-sided")

        note = effect_bin(d)
        print(f"{A:>5s} vs {B:<5s} | nA={a.size:6d}, nB={b.size:6d} | "
              f"Cliff's d = {d:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}] -> {note} | "
              f"MWU p = {p:.3e}")
        rows.append([A, B, a.size, b.size, d, lo, hi, p, ""])

    if csv_path is not None:
        hdr = "groupA,groupB,nA,nB,cliffs_d,ci_lo,ci_hi,mannwhitney_p,note"
        np.savetxt(csv_path, np.array(rows, dtype=object), fmt="%s",
                   delimiter=",", header=hdr, comments="")
        print("Saved:", csv_path)

# ------------------------------ #
# Scatter plot
# ------------------------------ #
def plot_meta_color_scatter(sim_pairs, groups, y_mode="cosine", fraction=0.003,
                            random_state=50, figsize=(8.5, 7.5), out_path=None, title=None):
    apply_pub_style()
    rng = np.random.default_rng(random_state)

    wt_colors = sns.color_palette("Greens", n_colors=len(groups["WT"]))
    ko_colors = sns.color_palette("Reds",   n_colors=len(groups["KO"]))
    sample_color = {name: col for name, col in zip(groups["WT"], wt_colors)}
    sample_color.update({name: col for name, col in zip(groups["KO"], ko_colors)})

    positions = {"WT-WT": 1, "KO-KO": 2, "WT-KO": 3}
    jitter_scale = 0.06

    def transform_y(y):
        y = np.asarray(y, float)
        if y_mode == "cosine":
            return y
        d = 1.0 - y
        d = np.clip(d, EPS_DIST, None)
        return np.log10(d)

    fig, ax = plt.subplots(figsize=figsize)
    for key, pairs in sim_pairs.items():
        pos = positions[key]
        for n1, n2, sims_list in pairs:
            sims = np.array(sims_list, dtype=float)
            sims = sims[np.isfinite(sims)]
            if sims.size == 0:
                continue
            subsample_n = max(20, int(len(sims) * fraction))
            subsample_n = min(subsample_n, sims.size)
            subsampled = rng.choice(sims, size=subsample_n, replace=False)
            yy = transform_y(subsampled)
            x = rng.normal(pos, jitter_scale, size=subsample_n)
            cpair = blend_colors(sample_color.get(n1, (0.4, 0.4, 0.4)),
                                 sample_color.get(n2, (0.4, 0.4, 0.4)))
            ax.scatter(x, yy, color=cpair, alpha=DOT_ALPHA, s=DOT_SIZE,
                       edgecolors="none", rasterized=True)

    ax.set_xlim(0.4, 3.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if y_mode == "cosine":
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(0.75, 1.0)
    else:
        ax.set_ylabel(r"log10(1 − cosine + eps)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if out_path:
        _savefig(fig, out_path)
        print("Saved:", out_path + ".[png|pdf]")
    else:
        plt.show()
        plt.close(fig)

# ------------------------------ #
# Heatmap helpers
# ------------------------------ #
def pair_median_matrix(sim_pairs, sample_names):
    name_to_idx = {n: i for i, n in enumerate(sample_names)}
    n = len(sample_names)
    M = np.full((n, n), np.nan, dtype=float)
    for _, pairs in sim_pairs.items():
        for n1, n2, sims in pairs:
            v = np.asarray(sims, float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            m = float(np.median(v))
            i, j = name_to_idx[n1], name_to_idx[n2]
            M[i, j] = M[j, i] = m
    np.fill_diagonal(M, 1.0)
    return M


def plot_similarity_heatmap_with_groupbars(M, sample_names, groups, title, out_path_prefix,
                                          cluster=False, cmap=HEATMAP_CMAP):
    apply_pub_style()
    group_label = ["WT" if n in groups["WT"] else "KO" for n in sample_names]
    handles = [
        mpatches.Patch(color=GROUP_LUT["WT"], label="WT"),
        mpatches.Patch(color=GROUP_LUT["KO"], label="KO"),
    ]

    if cluster:
        row_colors = [GROUP_LUT[g] for g in group_label]
        col_colors = [GROUP_LUT[g] for g in group_label]
        cg = sns.clustermap(
            M, method="average", metric="euclidean", cmap=cmap,
            vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
            xticklabels=sample_names, yticklabels=sample_names,
            row_colors=row_colors, col_colors=col_colors,
            linewidths=0.2, linecolor="white",
            figsize=(12.8, 8.0), dendrogram_ratio=(0.14, 0.14),
            colors_ratio=(0.03, 0.03), cbar_pos=(0.84, 0.22, 0.02, 0.55),
            cbar_kws={"ticks": HEATMAP_TICKS},
        )
        cg.fig.subplots_adjust(top=0.90, right=0.82)
        cg.ax_heatmap.set_title("")
        cg.fig.suptitle(title, y=0.96, fontsize=FS["title"] + 6)
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90, ha="center")
        plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0)
        cg.ax_heatmap.tick_params(axis="both", labelsize=FS["tick"] + 3)
        cg.cax.set_ylabel("Median cosine (sampled pairs)", fontsize=FS["axes"] + 5, rotation=90, labelpad=18)
        cg.cax.tick_params(labelsize=FS["tick"] + 3, width=1.2, length=6)
        cg.cax.set_position([0.84, 0.22, 0.02, 0.55])
        cg.fig.legend(handles=handles, title="Group", frameon=False,
                      loc="upper left", bbox_to_anchor=(0.88, 0.92),
                      fontsize=FS["legend"] + 6, title_fontsize=FS["title"] + 4)
        cg.savefig(out_path_prefix + "_clustered_heatmap.png", dpi=PUB_DPI, bbox_inches="tight")
        if SAVE_PDF_ALSO:
            cg.savefig(out_path_prefix + "_clustered_heatmap.pdf", bbox_inches="tight", dpi=PUB_DPI)
        plt.close(cg.fig)
        return

    wt_idx = [i for i, n in enumerate(sample_names) if n in groups["WT"]]
    ko_idx = [i for i, n in enumerate(sample_names) if n in groups["KO"]]
    wt_sorted = sorted(wt_idx, key=lambda i: np.nanmean(M[i]))
    ko_sorted = sorted(ko_idx, key=lambda i: np.nanmean(M[i]))
    order = wt_sorted + ko_sorted

    M_ord = M[np.ix_(order, order)]
    names_ord = [sample_names[i] for i in order]
    glab_ord = [("WT" if n in groups["WT"] else "KO") for n in names_ord]
    grgb_ord = np.array([mcolors.to_rgb(GROUP_LUT[g]) for g in glab_ord], dtype=float)

    fig = plt.figure(figsize=(10.8, 8.2))
    gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[0.35, 8.0], width_ratios=[0.35, 8.0], hspace=0.05, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_hm = fig.add_subplot(gs[1, 1])

    ax_top.imshow(np.array([grgb_ord], dtype=float), aspect="auto")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_xlim(-0.5, len(names_ord) - 0.5)

    ax_left.imshow(np.array([grgb_ord]).transpose(1, 0, 2), aspect="auto")
    ax_left.set_xticks([]); ax_left.set_yticks([])
    ax_left.set_ylim(len(names_ord) - 0.5, -0.5)

    hm = sns.heatmap(
        M_ord, ax=ax_hm, cmap=cmap, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX,
        xticklabels=names_ord, yticklabels=names_ord, square=True, cbar=True,
        cbar_kws={"label": "Median cosine (sampled pairs)", "ticks": HEATMAP_TICKS, "shrink": 0.92, "pad": 0.02, "aspect": 30},
        linewidths=0.2, linecolor="white"
    )
    ax_hm.set_title(title, pad=18)
    plt.setp(ax_hm.get_xticklabels(), rotation=90)
    ax_hm.tick_params(axis="both", labelsize=FS["tick"] + 3)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=FS["tick"] + 3, width=1.2, length=6)
    cbar.set_label("Median cosine (sampled pairs)", fontsize=FS["axes"] + 5)

    fig.legend(handles=handles, title="Group", frameon=False,
               loc="upper right", bbox_to_anchor=(1.22, 0.92),
               fontsize=FS["legend"] + 6, title_fontsize=FS["title"] + 4)
    _savefig(fig, out_path_prefix + "_heatmap")

# ------------------------------ #
# INPUTS
# ------------------------------ #
sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

#Right
selected_clusters_raw = {
    "T1" :[0,1,3,4,6,7],
    "T6" : [0,1,6,7],
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
selected_clusters = _clean_selected_clusters(selected_clusters_raw, n_clusters=8)

sample_paths = [os.path.join(umap_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

# ------------------------------ #
# Run one branch
# ------------------------------ #
def run_similarity_branch(branch_name, feature_mode, seed_offset=0):
    branch_prefix = os.path.join(save_dir, f"similarity_{branch_name}")

    raw_matrix, _, raw_sims, raw_pairs = build_similarity_matrices_custom(
        sample_paths, sample_names, groups, selected_clusters,
        feature_mode=feature_mode,
        second_derivative=False,
        subsample_size=subsample_size,
        random_state=rng_master_seed + seed_offset,
        normalize_spectra=True,
        fix_derivative_sign=False,
        abs_cosine=False,
        max_pairs_dist=max_pairs_dist,
        exclude_self_from_distributions=True
    )

    sec_matrix, _, sec_sims, sec_pairs = build_similarity_matrices_custom(
        sample_paths, sample_names, groups, selected_clusters,
        feature_mode=feature_mode,
        second_derivative=True,
        subsample_size=subsample_size,
        random_state=rng_master_seed + 1 + seed_offset,
        normalize_spectra=True,
        fix_derivative_sign=True,
        abs_cosine=False,
        max_pairs_dist=max_pairs_dist,
        exclude_self_from_distributions=True
    )

    summarize_delta(
        f"{branch_name} RAW cosine similarity",
        raw_sims, balance=True, seed=123,
        csv_path=branch_prefix + "_RAW_cliffs_cosine.csv"
    )
    summarize_delta(
        f"{branch_name} SEC 2nd-derivative cosine similarity",
        sec_sims, balance=True, seed=123,
        csv_path=branch_prefix + "_SEC_cliffs_cosine.csv"
    )

    log1m = lambda x: np.log10(np.clip(1.0 - x, EPS_DIST, None))
    summarize_delta(
        f"{branch_name} RAW log10(1 - cosine + eps)",
        raw_sims, transform=log1m, balance=True, seed=123,
        csv_path=branch_prefix + "_RAW_cliffs_log1mcos.csv"
    )
    summarize_delta(
        f"{branch_name} SEC log10(1 - cosine + eps)",
        sec_sims, transform=log1m, balance=True, seed=123,
        csv_path=branch_prefix + "_SEC_cliffs_log1mcos.csv"
    )

    plot_meta_color_scatter(
        raw_pairs, groups, y_mode="cosine", fraction=0.003,
        random_state=rng_master_seed + seed_offset,
        figsize=(8.5, 7.5),
        out_path=branch_prefix + "_RAW_meta_cosine",
        title=f"{branch_name} — RAW cosine similarities"
    )
    plot_meta_color_scatter(
        sec_pairs, groups, y_mode="cosine", fraction=0.003,
        random_state=rng_master_seed + 1 + seed_offset,
        figsize=(8.5, 7.5),
        out_path=branch_prefix + "_SEC_meta_cosine",
        title=f"{branch_name} — 2nd-derivative cosine similarities"
    )
    plot_meta_color_scatter(
        raw_pairs, groups, y_mode="log1mcos", fraction=0.003,
        random_state=rng_master_seed + seed_offset,
        figsize=(8.5, 7.5),
        out_path=branch_prefix + "_RAW_meta_log1mcos",
        title=f"{branch_name} — RAW log10(1 − cosine)"
    )
    plot_meta_color_scatter(
        sec_pairs, groups, y_mode="log1mcos", fraction=0.003,
        random_state=rng_master_seed + 1 + seed_offset,
        figsize=(8.5, 7.5),
        out_path=branch_prefix + "_SEC_meta_log1mcos",
        title=f"{branch_name} — 2nd-derivative log10(1 − cosine)"
    )

    M_raw = pair_median_matrix(raw_pairs, sample_names)
    M_sec = pair_median_matrix(sec_pairs, sample_names)
    plot_similarity_heatmap_with_groupbars(
        M_raw, sample_names, groups,
        title=f"{branch_name} — RAW pairwise median cosine",
        out_path_prefix=branch_prefix + "_raw",
        cluster=False,
        cmap=HEATMAP_CMAP
    )
    plot_similarity_heatmap_with_groupbars(
        M_sec, sample_names, groups,
        title=f"{branch_name} — 2nd-derivative pairwise median cosine",
        out_path_prefix=branch_prefix + "_sec",
        cluster=False,
        cmap=HEATMAP_CMAP
    )
    plot_similarity_heatmap_with_groupbars(
        M_sec, sample_names, groups,
        title=f"{branch_name} — 2nd-derivative clustered median cosine",
        out_path_prefix=branch_prefix + "_sec",
        cluster=True,
        cmap=HEATMAP_CMAP
    )

# ------------------------------ #
# MAIN
# ------------------------------ #
if __name__ == "__main__":
    print("Running FULL_SPECTRA branch...")
    run_similarity_branch("FULL_SPECTRA", feature_mode="full_spectra", seed_offset=0)

    print("Running SELECTED_BANDS_ONLY branch...")
    run_similarity_branch("SELECTED_BANDS_ONLY", feature_mode="band_only", seed_offset=100)

    print("Done. Separate results saved under:")
    print(save_dir)
