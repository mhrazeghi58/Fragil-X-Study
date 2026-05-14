#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from contextlib import contextmanager
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
from scipy import stats   # used for rankdata

# ============================== PATHS / CONFIG ==============================
cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L\UMAP_clustering_8Cluster"
cube_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
save_dir    = os.path.join(cube_dir, "WT_KO_bands")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "WT_KO")

sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

# ---------------- Cluster selection (Right) ----------------
#selected_clusters = {
#    "T1":[1,2,3,4,5,6],
#    "T2":[1,2,3,4,5,6],
#    "T3":[1,2,3,4,5,6,7],
#    "T6":[0,1,3,5,6,7],
#    "T17":[1,3,5,6,7],
#    "T19":[1,2,3,4.6,7],
#    "T20":[0,2,3,5,7],
#    "T21":[0,1,4,6,7],
#    "T10":[0,1,3,4,5,7],
#    "T11":[0,1,3,4,5,6],
#    "T12":[0,1,2,4,5,7],
#    "T13":[0,1,2,4,6,7],
#    "T14":[0,1,3,4,5,6],
#    "T15":[0,1,3,4,5,6,7],
#    "T22":[0,2,4,6,7],
#}

#Left
selected_clusters = {
    "T1" :[0,1,3,4,6,7],
    "T2" : [0,1,2],
    "T3" : [0,1,4,5],
    "T6" : [0,1,3,4,6],
    "T19": [0,2,3.6,7],
   "T20": [0,1,2,4,5,7],
   "T21": [1,2,3,4],
    "T17": [0,1,2,3,5,6],
    "T10": [0,2,4,5,6,7],
    "T11": [0,1,2,3,6,7],
    "T12": [0,1,2,3,5,7],
    "T13": [1,2,5,6,7],
    "T14": [0,1,4,5,7],# 
    "T15": [0,2,3,4,5,7],
    "T22": [0,1,4,5,6],
}

# --------- Spectral / band settings ---------
savgol_window = 11
polyorder     = 3
try:
    window
except NameError:
    window = 30

# Use “top-k peaks in window” integration instead of mean over the whole window
TOPK_PER_BAND = 1  # take the mean of the top-k values inside the window (per pixel)

# ---- Optional per-pixel normalization to remove overall intensity effects ----
#   "none"     : no normalization
#   "l2"       : divide each pixel spectrum by its L2 norm (default)
#   "l1"       : divide by L1 norm (sum of abs intensities)
#   "ref_1655" : divide by the Amide I (1655 cm-1) band value (top-k mean)
NORM_MODE = "l2"   # <— change here if you prefer "none" or "ref_1655"
NORM_EPS  = 1e-12

BAND_1734_CENTER    = 1734.0
BAND_1734_HALFWIDTH = 13
BAND_AMIDE_I        = 1655.0
BAND_AMIDE_II       = 1545.0

# ===================== MANUAL AXES & PIXEL CAPS =======================
USE_MANUAL_YLIMS = True
MANUAL_YLIMS = {
    "ch2ch3":         (0.5, 2.25),
    "amideI_amideII": (0.8, 2.5),
    "1734_amideI":    (0.00, 0.80),
    "1734_amideII":   (0.00, 1.00),
    "single_band":    None,
    "amideI":         None,
    "amideII":        None,
}
USE_LOG_Y_FOR = {
    "amideI_amideII": False,
    "1734_amideI":    False,
    "1734_amideII":   False,
    "amideI":         False,
    "amideII":        False,
}

PIXEL_CAP_PER_GROUP = {  # used for stats (concatenated), plots use per-sample cap
    "ch2ch3":         20_000,
    "amideI_amideII": 20_000,
    "1734_amideI":    20_000,
    "1734_amideII":   20_000,
    "single_band":    20_000,
    "amideI":         20_000,
    "amideII":        20_000,
}
MAX_POINTS_PER_SAMPLE = 2_000

# Annotate figure with Cliff's δ and win percentages
ANNOTATE_STATS_ON_PLOTS = True

# Random seed
SEED = 1
rng_global = np.random.default_rng(SEED)

# ========================== LOGGING & UTILS ==========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WT_KO")

@contextmanager
def LogTimer(msg):
    t0 = time.time(); logger.info(msg + " ...")
    try:
        yield
    finally:
        logger.info("%s done in %.2fs", msg, time.time() - t0)

# ========================== CLEAN selected_clusters ===================
def _clean_selected_clusters(raw, n_clusters=8):
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

selected_clusters = _clean_selected_clusters(selected_clusters, n_clusters=8)

# ========================== STYLE & CONSTANTS =========================
# Base colors by GROUP; per-sample colors are shades of these.
BASE_WT = "#2ca02c"
BASE_KO = "#d62728"

DOT_SIZE     = 5
JITTER_SCALE = 0.06
USE_VIOLIN_BACKGROUND = True
SAVE_PDF_ALSO = True

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11

# Group violin fill (soft)
VIOLIN_WT_COLOR = (0.3, 0.7, 0.3, 0.12)  # light greenish w/ alpha
VIOLIN_KO_COLOR = (0.9, 0.3, 0.3, 0.12)  # light reddish w/ alpha

TRIM_FOR_PLOTTING_ONLY = True
FORCE_NONNEG_BANDS   = False
DENOM_MIN_PCT        = 3.0
DENOM_ABS_FLOOR      = 1e-2
NUM_MIN_PCT          = 15.0
NUM_ABS_FLOOR        = 1e-4
APPLY_RATIO_HARD_CAP = True
RATIO_HARD_CAP       = 10.0

def _apply_pub_style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": PUB_DPI,
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE + 1,
        "axes.titlesize": BASE_FONTSIZE + 3,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "axes.linewidth": 0.8,
        "grid.color": "#9aa0a6",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "mathtext.default": "regular",
    })

# ========================== DATA LOADING / NORM ======================
def _load_cluster_labels_indices(sample):
    f = os.path.join(cluster_dir, f"{sample}_umap_kmeans.npz")
    if not os.path.exists(f):
        logger.warning("Missing cluster file for %s", sample); return None
    z = np.load(f)
    return z["cluster_labels"], z["pixel_indices"]

def _load_padded_cube(sample, expected_bands=426):
    f = os.path.join(cube_dir, f"masked_cube_{sample}.npz")
    if not os.path.exists(f):
        logger.warning("Missing RAW cube for %s", sample); return None
    cube = np.load(f)["data"]  # (H,W,Z)
    H, W, Z = cube.shape
    if Z < expected_bands:
        missing = expected_bands - Z
        logger.warning("%s: missing %d bands – padding with front edge values", sample, missing)
        edge = cube[:, :, :1]
        pad  = np.repeat(edge, missing, axis=2)
        cube = np.concatenate([pad, cube], axis=2)
    elif Z > expected_bands:
        logger.warning("%s: has %d extra bands – trimming from start", sample, Z - expected_bands)
        cube = cube[:, :, -expected_bands:]
    return cube

def _selected_mask_from_clusters(H, W, labels, indices, sel_lst):
    if indices.ndim == 2 and indices.shape[1] == 2:
        rows = indices[:, 0]; cols = indices[:, 1]
        use = np.isin(labels, sel_lst); r = rows[use]; c = cols[use]
        m = np.zeros((H, W), dtype=bool); m[r, c] = True
        return m
    else:
        flat_idx = indices
        use = np.isin(labels, sel_lst); flat_use = flat_idx[use]
        m = np.zeros((H*W,), dtype=bool); m[flat_use] = True
        return m.reshape(H, W)

def _maybe_normalize_cube(cube, wns, mode=NORM_MODE):
    """Return a normalized copy (or the original) according to `mode`."""
    if mode is None or mode.lower() == "none":
        return cube
    C = cube.astype(np.float64, copy=True)
    if mode.lower() == "l2":
        norm = np.sqrt(np.sum(C * C, axis=2, keepdims=True))
        C = C / (norm + NORM_EPS)
    elif mode.lower() == "l1":
        denom = np.sum(np.abs(C), axis=2, keepdims=True)
        C = C / (denom + NORM_EPS)
    elif mode.lower() == "ref_1655":
        ref = _band_value_topk_mean(C, wns, BAND_AMIDE_I, window, k=TOPK_PER_BAND)
        ref = np.clip(ref, NORM_EPS, None)
        C = C / ref[:, :, None]
    else:
        return C
    return C

# --- top-k-mean extractor over a band window ---
def _band_value_topk_mean(cube, wns, center, halfwidth, k=TOPK_PER_BAND):
    """For each pixel, within [center±halfwidth], take the mean of the top-k values."""
    if halfwidth is None or halfwidth <= 0:
        halfwidth = 10
    m = (wns >= (center - halfwidth)) & (wns <= (center + halfwidth))
    if not np.any(m):
        H, W = cube.shape[:2]
        return np.full((H, W), np.nan, dtype=float)

    slab = cube[:, :, m]  # (H,W,nwin)
    if FORCE_NONNEG_BANDS:
        slab = np.clip(slab, 0, None)

    nwin = slab.shape[2]
    kk = int(max(1, min(k, nwin)))

    H, W = slab.shape[:2]
    flat = slab.reshape(-1, nwin)
    idx = np.argpartition(flat, -kk, axis=1)[:, -kk:]
    topk = np.take_along_axis(flat, idx, axis=1)
    out_flat = np.nanmean(topk, axis=1)
    return out_flat.reshape(H, W)

# alias used throughout
_band_value = _band_value_topk_mean

# ========================== PLOTTING HELPERS ==========================
AUTO_PAD_FRAC = 0.01

def _auto_ylim(a, b, log=False, pad_frac=AUTO_PAD_FRAC):
    vals = np.r_[a, b]; vals = vals[np.isfinite(vals)]
    if vals.size == 0: return None
    lo, hi = np.nanpercentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi: return None
    pad = (hi - lo) * pad_frac
    lo2, hi2 = lo - pad, hi + pad
    if log: lo2 = max(lo2, 1e-6)
    return (lo2, hi2)

def _mask_for_ylim(arr, ylim):
    if ylim is None: return arr
    lo, hi = ylim
    return arr[(arr >= lo) & (arr <= hi) & np.isfinite(arr)]

def _denom_guard(den, min_pct=DENOM_MIN_PCT, abs_floor=DENOM_ABS_FLOOR):
    d = den[np.isfinite(den)]
    if d.size == 0:
        return np.full_like(den, False, dtype=bool), abs_floor
    thr_q = np.nanpercentile(d, min_pct)
    thr = max(thr_q, abs_floor)
    return (den > thr), thr

def _num_guard(num, min_pct=NUM_MIN_PCT, abs_floor=NUM_ABS_FLOOR):
    d = num[np.isfinite(num)]
    if d.size == 0:
        return np.full_like(num, False, dtype=bool), abs_floor
    thr_q = np.nanpercentile(d, min_pct)
    thr = max(thr_q, abs_floor)
    return (num > thr), thr

def _annotate(ax, text):
    if not ANNOTATE_STATS_ON_PLOTS or not text:
        return
    ylim = ax.get_ylim()
    y = ylim[1] - (ylim[1]-ylim[0]) * 0.06
    ax.text(0.5, y, text, ha="center", va="top", transform=ax.get_xaxis_transform(), fontsize=10)

# --- Per-group sample color maps (shades of green/red) ---
def _group_sample_colors(wt_keys, ko_keys):
    import colorsys, matplotlib.colors as mcolors
    def ramp(hex_color, n, v_min=0.45, v_max=0.95, s=0.85):
        r,g,b = mcolors.to_rgb(hex_color)
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        vs = np.linspace(v_min, v_max, max(n,1))
        return [colorsys.hsv_to_rgb(h, s, v) for v in vs]
    wt_cols = ramp(BASE_WT, len(wt_keys))
    ko_cols = ramp(BASE_KO, len(ko_keys))
    return {k: wt_cols[i] for i,k in enumerate(wt_keys)}, {k: ko_cols[i] for i,k in enumerate(ko_keys)}

def _strip_plot_by_sample(WT_dict, KO_dict, ylabel, title, out_png, log_y=False, ylim=None, stats_text=None):
    """
    WT_dict / KO_dict: dict {sample_name -> 1D array}
    Each WT sample gets a distinct green hue; each KO sample a distinct red hue.
    """
    _apply_pub_style()

    # Flatten for violin & medians
    WT_all = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    if len(WT_all) == 0 or len(KO_all) == 0:
        logger.warning("Nothing to plot for %s", title); return

    # Trim for display if manual ylim provided
    WTv = WT_all; KOv = KO_all
    if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
        WTv = _mask_for_ylim(WT_all, ylim)
        KOv = _mask_for_ylim(KO_all, ylim)

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    rng = np.random.default_rng(0)

    # Violin background per group (neutral alpha)
    if USE_VIOLIN_BACKGROUND:
        parts = ax.violinplot([WTv, KOv], positions=[0, 1], widths=0.6,
                               showmeans=False, showmedians=False, showextrema=False)
        parts["bodies"][0].set_facecolor(VIOLIN_WT_COLOR); parts["bodies"][0].set_edgecolor("none")
        parts["bodies"][1].set_facecolor(VIOLIN_KO_COLOR); parts["bodies"][1].set_edgecolor("none")

    # Distinct colors per sample within each group
    wt_keys = list(WT_dict.keys()); ko_keys = list(KO_dict.keys())
    wt_col_map, ko_col_map = _group_sample_colors(wt_keys, ko_keys)

    handles = []

    # WT at x=0
    for s, arr in WT_dict.items():
        if len(arr) == 0: continue
        arrv = _mask_for_ylim(arr, ylim) if (ylim is not None and TRIM_FOR_PLOTTING_ONLY) else arr
        xs = 0 + rng.normal(0, JITTER_SCALE, size=len(arrv))
        col = wt_col_map.get(s)
        ax.scatter(xs, arrv, s=DOT_SIZE, alpha=0.5, c=[col], edgecolors="none", rasterized=True)
        handles.append(mpatches.Patch(color=col, label=s))
    # KO at x=1
    for s, arr in KO_dict.items():
        if len(arr) == 0: continue
        arrv = _mask_for_ylim(arr, ylim) if (ylim is not None and TRIM_FOR_PLOTTING_ONLY) else arr
        xs = 1 + rng.normal(0, JITTER_SCALE, size=len(arrv))
        col = ko_col_map.get(s)
        ax.scatter(xs, arrv, s=DOT_SIZE, alpha=0.5, c=[col], edgecolors="none", rasterized=True)
        handles.append(mpatches.Patch(color=col, label=s))

    # Group medians + 5–95% whiskers
    def _median_iqr(x, xpos):
        if len(x) == 0: return
        med = np.median(x); q1, q3 = np.quantile(x, [0.05, 0.95])
        ax.scatter([xpos], [med], s=38, c="white", edgecolors="black", linewidths=0.7, zorder=3)
        ax.vlines(xpos, q1, q3, colors="black", linewidth=1.2, zorder=3)
    _median_iqr(WTv, 0); _median_iqr(KOv, 1)

    ax.set_xticks([0, 1], ["WT", "KO"])
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
        if ylim is not None: ax.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.yaxis.grid(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # Legend outside (one entry per sample)
    if handles:
        ax.legend(handles=handles, title="Samples", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, ncol=1)

    _annotate(ax, stats_text)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    if SAVE_PDF_ALSO:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ========================== STATISTICS (Cliff's only) =================
def _fast_mwu_U(x, y):
    x = np.asarray(x); y = np.asarray(y)
    ranks = stats.rankdata(np.concatenate([x, y]))
    rx = ranks[:len(x)]
    U = rx.sum() - len(x) * (len(x) + 1) / 2.0
    return float(U)

def cliffs_delta(WT, KO):
    WT = np.asarray(WT); KO = np.asarray(KO)
    nx, ny = len(WT), len(KO)
    if nx == 0 or ny == 0:
        return np.nan
    U = _fast_mwu_U(WT, KO)
    return float((2.0 * U) / (nx * ny) - 1.0)

def bootstrap_ci(stat_fn, x, y, nboot=3000, alpha=0.05, rng=None):
    if rng is None: rng = rng_global
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return (np.nan, np.nan)
    vals = []
    for _ in range(nboot):
        xb = x[rng.integers(0, nx, nx)]
        yb = y[rng.integers(0, ny, ny)]
        vals.append(stat_fn(xb, yb))
    lo, hi = np.quantile(vals, [alpha/2, 1 - alpha/2])
    return float(lo), float(hi)

def pairwise_preference(WT, KO, max_pairs=200_000, rng=None):
    if rng is None: rng = rng_global
    WT = np.asarray(WT); KO = np.asarray(KO)
    nW, nK = len(WT), len(KO)
    if nW == 0 or nK == 0:
        return (np.nan, np.nan, np.nan)
    P = min(max_pairs, nW * nK)
    i = rng.integers(0, nW, P); j = rng.integers(0, nK, P)
    a = WT[i]; b = KO[j]
    p_wt = np.mean(a > b); p_ko = np.mean(b > a)
    p_tie = 1.0 - p_wt - p_ko
    return float(p_wt), float(p_ko), float(p_tie)

def cliffs_only_stats(name, WT, KO):
    WT = np.asarray(WT); KO = np.asarray(KO)
    n_wt, n_ko = len(WT), len(KO)
    if n_wt == 0 or n_ko == 0:
        return {
            "metric": name, "n_wt": n_wt, "n_ko": n_ko,
            "mean_wt": np.nan, "mean_ko": np.nan,
            "median_wt": np.nan, "median_ko": np.nan,
            "sd_wt": np.nan, "sd_ko": np.nan,
            "cliffs_delta": np.nan, "cliffs_delta_lo": np.nan, "cliffs_delta_hi": np.nan,
            "median_diff": np.nan, "median_diff_lo": np.nan, "median_diff_hi": np.nan,
            "p_wt_gt_ko": np.nan, "p_ko_gt_wt": np.nan, "p_tie": np.nan
        }
    delta = cliffs_delta(WT, KO)
    d_lo, d_hi = bootstrap_ci(cliffs_delta, WT, KO, nboot=3000)
    med_diff_fn = lambda a,b: float(np.median(a) - np.median(b))
    mdiff = med_diff_fn(WT, KO)
    md_lo, md_hi = bootstrap_ci(med_diff_fn, WT, KO, nboot=3000)
    p_wt, p_ko, p_tie = pairwise_preference(WT, KO, max_pairs=200_000)
    return {
        "metric": name,
        "n_wt": n_wt, "n_ko": n_ko,
        "mean_wt": float(np.mean(WT)), "mean_ko": float(np.mean(KO)),
        "median_wt": float(np.median(WT)), "median_ko": float(np.median(KO)),
        "sd_wt": float(np.std(WT, ddof=1)) if n_wt>1 else np.nan,
        "sd_ko": float(np.std(KO, ddof=1)) if n_ko>1 else np.nan,
        "cliffs_delta": float(delta), "cliffs_delta_lo": float(d_lo), "cliffs_delta_hi": float(d_hi),
        "median_diff": float(mdiff), "median_diff_lo": float(md_lo), "median_diff_hi": float(md_hi),
        "p_wt_gt_ko": float(p_wt), "p_ko_gt_wt": float(p_ko), "p_tie": float(p_tie)
    }

def save_stats_csv(rows, path):
    headers = ["metric","n_wt","n_ko","mean_wt","mean_ko","median_wt","median_ko","sd_wt","sd_ko",
               "cliffs_delta","cliffs_delta_lo","cliffs_delta_hi",
               "median_diff","median_diff_lo","median_diff_hi",
               "p_wt_gt_ko","p_ko_gt_wt","p_tie"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    logger.info("Saved stats table: %s", path)

def format_stats_for_title(row):
    d  = row["cliffs_delta"]
    pK = row["p_ko_gt_wt"] * 100.0
    pW = row["p_wt_gt_ko"] * 100.0
    return f"Cliff's δ={d:+.2f}; KO>WT≈{pK:.0f}% vs WT>KO≈{pW:.0f}%"

# ========================== GATHER HELPERS ============================
def _collect_ratios_for_sample(sample):
    out = _load_cluster_labels_indices(sample)
    if out is None: return None, None
    labels, indices = out
    cube = _load_padded_cube(sample)
    if cube is None: return None, None
    H, W, Z = cube.shape

    sel = selected_clusters.get(sample, [])
    if len(sel) == 0:
        logger.warning("No selected clusters for %s; skipping", sample); return None, None

    mask2d = _selected_mask_from_clusters(H, W, labels, indices, sel)
    if not np.any(mask2d):
        logger.warning("No pixels in selected clusters for %s; skipping", sample); return None, None

    wns    = np.linspace(950, 1800, Z)
    # intensity normalization (per pixel)
    cube   = _maybe_normalize_cube(cube, wns, mode=NORM_MODE)

    ch2    = _band_value(cube, wns, 1464, window)
    ch3    = _band_value(cube, wns, 1375, window)
    amideI = _band_value(cube, wns, BAND_AMIDE_I, window)
    amideII= _band_value(cube, wns, BAND_AMIDE_II, window)

    vR1_den, _ = _denom_guard(ch3)
    nR1_num, _ = _num_guard(ch2)
    vR2_den, _ = _denom_guard(amideII)
    nR2_num, _ = _num_guard(amideI)

    valid1 = np.isfinite(ch2)    & np.isfinite(ch3)    & nR1_num & vR1_den
    valid2 = np.isfinite(amideI) & np.isfinite(amideII)& nR2_num & vR2_den

    r1 = np.full_like(ch2,     np.nan, dtype=float)
    r2 = np.full_like(amideI,  np.nan, dtype=float)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        r1[valid1] = ch2[valid1]     / ch3[valid1]
        r2[valid2] = amideI[valid2]  / amideII[valid2]
    if APPLY_RATIO_HARD_CAP:
        r1 = np.clip(r1, 0, RATIO_HARD_CAP)
        r2 = np.clip(r2, 0, RATIO_HARD_CAP)

    r1 = r1[mask2d].ravel(); r2 = r2[mask2d].ravel()
    ok = np.isfinite(r1) & np.isfinite(r2) & (r1 > 0) & (r2 > 0)
    r1, r2 = r1[ok], r2[ok]

    if len(r1) > MAX_POINTS_PER_SAMPLE:
        rng = np.random.default_rng(abs(hash(sample)) % (2**32))
        take = rng.choice(len(r1), size=MAX_POINTS_PER_SAMPLE, replace=False)
        r1, r2 = r1[take], r2[take]
    return r1, r2

def _collect_ratios_for_sample_ext(sample, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH):
    out = _load_cluster_labels_indices(sample)
    if out is None: return (None, None, None, None)
    labels, indices = out
    cube = _load_padded_cube(sample)
    if cube is None: return (None, None, None, None)
    H, W, Z = cube.shape

    sel = selected_clusters.get(sample, [])
    if len(sel) == 0:
        logger.warning("No selected clusters for %s; skipping", sample)
        return (None, None, None, None)

    mask2d = _selected_mask_from_clusters(H, W, labels, indices, sel)
    if not np.any(mask2d):
        logger.warning("No pixels in selected clusters for %s; skipping", sample)
        return (None, None, None, None)

    wns    = np.linspace(950, 1800, Z)
    cube   = _maybe_normalize_cube(cube, wns, mode=NORM_MODE)

    ch2    = _band_value(cube, wns, 1464, window)
    ch3    = _band_value(cube, wns, 1375, window)
    amideI = _band_value(cube, wns, BAND_AMIDE_I, window)
    amideII= _band_value(cube, wns, BAND_AMIDE_II, window)
    b1734  = _band_value(cube, wns, float(center_1734), halfwidth_1734 if halfwidth_1734 is not None else window)

    vR1_den, _  = _denom_guard(ch3)
    nR1_num, _  = _num_guard(ch2)
    vR2_den, _  = _denom_guard(amideII)
    nR2_num, _  = _num_guard(amideI)
    nR3_num, _  = _num_guard(b1734)

    valid1 = np.isfinite(ch2)    & np.isfinite(ch3)     & nR1_num & vR1_den
    valid2 = np.isfinite(amideI) & np.isfinite(amideII) & nR2_num & vR2_den
    valid3 = np.isfinite(b1734)  & np.isfinite(amideI)  & nR3_num & nR2_num
    valid4 = np.isfinite(b1734)  & np.isfinite(amideII) & nR3_num & vR2_den

    r1 = np.full_like(ch2,     np.nan, dtype=float)
    r2 = np.full_like(amideI,  np.nan, dtype=float)
    r3 = np.full_like(amideI,  np.nan, dtype=float)
    r4 = np.full_like(amideII, np.nan, dtype=float)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        r1[valid1] = ch2[valid1]     / ch3[valid1]
        r2[valid2] = amideI[valid2]  / amideII[valid2]
        r3[valid3] = b1734[valid3]   / amideI[valid3]
        r4[valid4] = b1734[valid4]   / amideII[valid4]

    if APPLY_RATIO_HARD_CAP:
        r1 = np.clip(r1, 0, RATIO_HARD_CAP)
        r2 = np.clip(r2, 0, RATIO_HARD_CAP)
        r3 = np.clip(r3, 0, RATIO_HARD_CAP)
        r4 = np.clip(r4, 0, RATIO_HARD_CAP)

    r1 = r1[mask2d].ravel(); r2 = r2[mask2d].ravel(); r3 = r3[mask2d].ravel(); r4 = r4[mask2d].ravel()
    ok1 = np.isfinite(r1) & (r1 > 0)
    ok2 = np.isfinite(r2) & (r2 > 0)
    ok3 = np.isfinite(r3) & (r3 > 0)
    ok4 = np.isfinite(r4) & (r4 > 0)
    r1, r2, r3, r4 = r1[ok1], r2[ok2], r3[ok3], r4[ok4]

    def _cap(arr, seed):
        if len(arr) <= MAX_POINTS_PER_SAMPLE: return arr
        rng = np.random.default_rng(abs(hash((sample, seed))) % (2**32))
        take = rng.choice(len(arr), size=MAX_POINTS_PER_SAMPLE, replace=False)
        return arr[take]
    r1 = _cap(r1, ("ch2ch3", center_1734))
    r2 = _cap(r2, ("ai_aii", center_1734))
    r3 = _cap(r3, ("1734_ai", center_1734))
    r4 = _cap(r4, ("1734_aii", center_1734))
    return r1, r2, r3, r4

def _collect_band_for_sample(sample, center_cm1, halfwidth=None):
    out = _load_cluster_labels_indices(sample)
    if out is None: return None
    labels, indices = out
    cube = _load_padded_cube(sample)
    if cube is None: return None
    H, W, Z = cube.shape

    sel = selected_clusters.get(sample, [])
    if len(sel) == 0:
        logger.warning("No selected clusters for %s; skipping", sample); return None

    mask2d = _selected_mask_from_clusters(H, W, labels, indices, sel)
    if not np.any(mask2d):
        logger.warning("No pixels in selected clusters for %s; skipping", sample); return None

    wns = np.linspace(950, 1800, Z)
    cube = _maybe_normalize_cube(cube, wns, mode=NORM_MODE)
    band_img = _band_value(cube, wns, float(center_cm1), halfwidth if halfwidth is not None else window)
    vals = band_img[mask2d].ravel()
    ok = np.isfinite(vals) & (vals > 0)
    vals = vals[ok]
    if len(vals) > MAX_POINTS_PER_SAMPLE:
        rng = np.random.default_rng(abs(hash((sample, center_cm1))) % (2**32))
        take = rng.choice(len(vals), size=MAX_POINTS_PER_SAMPLE, replace=False)
        vals = vals[take]
    return vals

# ---- Gather for STATS (concatenated) ----
def _cap_group(arr, metric_key, seed):
    cap = PIXEL_CAP_PER_GROUP.get(metric_key, 30_000)
    if len(arr) <= cap: return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(len(arr), size=cap, replace=False)]

def _gather_by_group_for_stats(sample_names, groups):
    WT_vals = {"ch2ch3": [], "ai_aii": []}
    KO_vals = {"ch2ch3": [], "ai_aii": []}
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    with LogTimer("Collect per-pixel RAW ratios for WT/KO (stats)"):
        for n in sample_names:
            g = sample_to_group.get(n)
            if g is None: continue
            r1, r2 = _collect_ratios_for_sample(n)
            if r1 is None: continue
            (WT_vals if g=="WT" else KO_vals)["ch2ch3"].append(r1)
            (WT_vals if g=="WT" else KO_vals)["ai_aii"].append(r2)

    def _concat(lst): return np.concatenate(lst) if lst else np.array([])
    WT_ch2ch3 = _concat(WT_vals["ch2ch3"]); WT_ai_aii = _concat(WT_vals["ai_aii"])
    KO_ch2ch3 = _concat(KO_vals["ch2ch3"]); KO_ai_aii = _concat(KO_vals["ai_aii"])

    WT_ch2ch3 = _cap_group(WT_ch2ch3, "ch2ch3",         seed=1)
    WT_ai_aii = _cap_group(WT_ai_aii, "amideI_amideII", seed=2)
    KO_ch2ch3 = _cap_group(KO_ch2ch3, "ch2ch3",         seed=3)
    KO_ai_aii = _cap_group(KO_ai_aii, "amideI_amideII", seed=4)
    return (WT_ch2ch3, KO_ch2ch3), (WT_ai_aii, KO_ai_aii)

def _gather_by_group_ext_for_stats(sample_names, groups, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH):
    WT_vals = {"ch2ch3": [], "ai_aii": [], "1734_ai": [], "1734_aii": []}
    KO_vals = {"ch2ch3": [], "ai_aii": [], "1734_ai": [], "1734_aii": []}
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    with LogTimer("Collect per-pixel RAW ratios incl. 1734-band (stats)"):
        for n in sample_names:
            g = sample_to_group.get(n)
            if g is None: continue
            r1, r2, r3, r4 = _collect_ratios_for_sample_ext(n, center_1734=center_1734, halfwidth_1734=halfwidth_1734)
            if r1 is None: continue
            tgt = WT_vals if g == "WT" else KO_vals
            tgt["ch2ch3"].append(r1);   tgt["ai_aii"].append(r2)
            tgt["1734_ai"].append(r3);  tgt["1734_aii"].append(r4)

    def _concat(lst): return np.concatenate(lst) if lst else np.array([])
    out = {
        "ch2ch3":   (_concat(WT_vals["ch2ch3"]),   _concat(KO_vals["ch2ch3"])),
        "ai_aii":   (_concat(WT_vals["ai_aii"]),   _concat(KO_vals["ai_aii"])),
        "1734_ai":  (_concat(WT_vals["1734_ai"]),  _concat(KO_vals["1734_ai"])),
        "1734_aii": (_concat(WT_vals["1734_aii"]), _concat(KO_vals["1734_aii"])),
    }
    out["ch2ch3"]   = (_cap_group(out["ch2ch3"][0],   "ch2ch3",         301),
                       _cap_group(out["ch2ch3"][1],   "ch2ch3",         302))
    out["ai_aii"]   = (_cap_group(out["ai_aii"][0],   "amideI_amideII", 401),
                       _cap_group(out["ai_aii"][1],   "amideI_amideII", 402))
    out["1734_ai"]  = (_cap_group(out["1734_ai"][0],  "1734_amideI",    501),
                       _cap_group(out["1734_ai"][1],  "1734_amideI",    502))
    out["1734_aii"] = (_cap_group(out["1734_aii"][0], "1734_amideII",   601),
                       _cap_group(out["1734_aii"][1], "1734_amideII",   602))
    return out

# ---- Gather for PLOTS (per-sample) ----
def _gather_per_sample_basic(sample_names, groups):
    """Return WT_dict, KO_dict for CH2/CH3 and AI/AII as two dict pairs."""
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})
    WT_ch, KO_ch = {}, {}
    WT_ai, KO_ai = {}, {}
    with LogTimer("Collect per-sample ratios for plotting (CH2/CH3, AmideI/II)"):
        for n in sample_names:
            r1, r2 = _collect_ratios_for_sample(n)
            if r1 is None: continue
            if sample_to_group.get(n) == "WT":
                WT_ch[n] = r1; WT_ai[n] = r2
            else:
                KO_ch[n] = r1; KO_ai[n] = r2
    return (WT_ch, KO_ch), (WT_ai, KO_ai)

def _gather_per_sample_ext(sample_names, groups, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH):
    """Return dict pairs per ratio for plotting."""
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})
    WT_1734_ai, KO_1734_ai = {}, {}
    WT_1734_aii, KO_1734_aii = {}, {}
    with LogTimer("Collect per-sample 1734 ratios for plotting"):
        for n in sample_names:
            r1, r2, r3, r4 = _collect_ratios_for_sample_ext(n, center_1734=center_1734, halfwidth_1734=halfwidth_1734)
            if r1 is None: continue
            if sample_to_group.get(n) == "WT":
                WT_1734_ai[n]  = r3; WT_1734_aii[n] = r4
            else:
                KO_1734_ai[n]  = r3; KO_1734_aii[n] = r4
    return (WT_1734_ai, KO_1734_ai), (WT_1734_aii, KO_1734_aii)

def _gather_per_sample_band(sample_names, groups, center_cm1, halfwidth=None):
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})
    WT_b, KO_b = {}, {}
    with LogTimer(f"Collect per-sample band @{center_cm1:.1f} for plotting"):
        for n in sample_names:
            v = _collect_band_for_sample(n, center_cm1, halfwidth=halfwidth)
            if v is None: continue
            if sample_to_group.get(n) == "WT":
                WT_b[n] = v
            else:
                KO_b[n] = v
    return WT_b, KO_b

# ===================== SIGNED 2nd-DERIV & CENTROID LEANING ============
def build_signed_secderiv_matrices(sample_paths, sample_names, selected_clusters,
                                   window=savgol_window, poly=polyorder,
                                   subsample_size=2000, random_state=SEED,
                                   do_normalize=True):
    rng = np.random.default_rng(random_state)
    staged = {}
    for path, name in zip(sample_paths, sample_names):
        if not os.path.exists(path): raise FileNotFoundError(path)
        data = np.load(path)
        spec = data["spectra"]
        spec = savgol_filter(spec, window_length=window, polyorder=poly, deriv=2, axis=1)
        mask = np.isin(data["cluster_labels"], selected_clusters.get(name, []))
        spec = spec[mask]
        spec = spec[np.all(np.isfinite(spec), axis=1)]
        if len(spec) > subsample_size:
            spec = spec[rng.choice(len(spec), size=subsample_size, replace=False)]
        staged[name] = spec

    all_for_ref = np.vstack([v for v in staged.values() if len(v) > 0]) if any(len(v)>0 for v in staged.values()) else None
    ref = None
    if all_for_ref is not None:
        ref_vec = np.median(all_for_ref, axis=0) if all_for_ref.shape[0] >= 5 else np.mean(all_for_ref, axis=0)
        ref = ref_vec / (np.linalg.norm(ref_vec) + 1e-12)

    X = {}
    for name, S in staged.items():
        if S.size == 0: X[name] = S; continue
        if ref is not None:
            dots = S @ ref
            S[dots < 0] *= -1.0
        if do_normalize:
            S = normalize(S, axis=1)
        X[name] = S
    return X

def _unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _sample_vector(S, reducer="median"):
    if S is None or S.size == 0: return None
    v = (np.median(S, axis=0) if reducer == "median" else np.mean(S, axis=0)).astype(np.float64)
    return _unit(v)

def centroid_angles_and_margin(sample_paths, sample_names, groups, selected_clusters,
                               subsample_size=2000, reducer="median"):
    with LogTimer("Build signed sec-deriv matrices & centroid leaning"):
        X = build_signed_secderiv_matrices(
            sample_paths, sample_names, selected_clusters,
            window=savgol_window, poly=polyorder,
            subsample_size=subsample_size, random_state=SEED, do_normalize=True
        )
        V = {n: _sample_vector(X[n], reducer=reducer) for n in sample_names}
        WT_vecs = [_unit(V[n]) for n in groups["WT"] if V.get(n) is not None]
        KO_vecs = [_unit(V[n]) for n in groups["KO"] if V.get(n) is not None]
        mu_WT   = _unit(np.mean(WT_vecs, axis=0)) if WT_vecs else None
        mu_KO   = _unit(np.mean(KO_vecs, axis=0)) if KO_vecs else None

        def _angle(u, v):
            if u is None or v is None: return np.nan
            c = float(np.clip(np.dot(u, v), -1.0, 1.0))
            return np.arccos(c)

        angles = {}
        for n in sample_names:
            th_wt = _angle(V.get(n), mu_WT)
            th_ko = _angle(V.get(n), mu_KO)
            margin_mrad = (th_ko - th_wt) * 1e3
            angles[n] = (th_wt * 1e3, th_ko * 1e3, margin_mrad)

        _apply_pub_style()
        bar = sorted(
            [(n, ("WT" if n in groups["WT"] else "KO"), *angles[n]) for n in sample_names if np.isfinite(angles[n][2])],
            key=lambda r: r[3], reverse=True
        )
        plt.figure(figsize=(8.8, 4.5))
        x = np.arange(len(bar))
        margins = [mm for _,_,_,mm in bar]
        cols = ["#2ca02c" if g == "WT" else "#d62728" for _, g, *_ in bar]
        plt.axhline(0, ls="--", lw=1)
        plt.bar(x, margins, color=cols, edgecolor="k", alpha=0.85)
        plt.xticks(x, [n for n, *_ in bar], rotation=90)
        plt.ylabel("Angle margin (mrad): θKO − θWT\n(+ ⇒ WT-like, − ⇒ KO-like)")
        plt.title("Centroid leaning (2nd-deriv, sign-corrected)")
        plt.tight_layout()
        path = out_prefix + "_centroid_margins.png"
        plt.savefig(path, dpi=600, bbox_inches="tight")
        if SAVE_PDF_ALSO:
            plt.savefig(out_prefix + "_centroid_margins.pdf", bbox_inches="tight")
        plt.show()
        logger.info("Saved: %s", path)
        return angles

# ========================== MAIN RUN ==================================
if __name__ == "__main__":
    stats_rows = []

    # ---------- RAW ratios (stats + per-sample hue plots) ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        (ch2ch3_WT_s, ch2ch3_KO_s), (ai_aii_WT_s, ai_aii_KO_s) = _gather_by_group_for_stats(sample_names, groups)

        row_ch = cliffs_only_stats("CH2/CH3", ch2ch3_WT_s, ch2ch3_KO_s); stats_rows.append(row_ch)
        row_ai = cliffs_only_stats("AmideI/AmideII", ai_aii_WT_s, ai_aii_KO_s); stats_rows.append(row_ai)

        (WT_ch_dict, KO_ch_dict), (WT_ai_dict, KO_ai_dict) = _gather_per_sample_basic(sample_names, groups)

        out1 = os.path.join(save_dir, "ratios_CH2_CH3_strip_by_sample_hues.png")
        out2 = os.path.join(save_dir, "ratios_AmideI_AmideII_strip_by_sample_hues.png")

        ylim_ch2ch3 = MANUAL_YLIMS["ch2ch3"] if USE_MANUAL_YLIMS else _auto_ylim(ch2ch3_WT_s, ch2ch3_KO_s, log=False)
        ylim_ai_aii = MANUAL_YLIMS["amideI_amideII"] if USE_MANUAL_YLIMS else _auto_ylim(ai_aii_WT_s, ai_aii_KO_s, log=USE_LOG_Y_FOR["amideI_amideII"])

        _strip_plot_by_sample(
            WT_ch_dict, KO_ch_dict,
            ylabel="CH$_2$/CH$_3$ ratio (per pixel; selected clusters; top-k; normalized: %s)" % NORM_MODE,
            title="CH$_2$/CH$_3$ Ratio – WT vs KO (distinct green/red hues per sample)",
            out_png=out1, log_y=False, ylim=ylim_ch2ch3,
            stats_text=format_stats_for_title(row_ch)
        )
        _strip_plot_by_sample(
            WT_ai_dict, KO_ai_dict,
            ylabel="Amide I / Amide II ratio (per pixel; selected clusters; top-k; normalized: %s)" % NORM_MODE,
            title="Amide I / Amide II – WT vs KO (distinct green/red hues per sample)",
            out_png=out2, log_y=USE_LOG_Y_FOR["amideI_amideII"], ylim=ylim_ai_aii,
            stats_text=format_stats_for_title(row_ai)
        )
        logger.info("Saved ratio plots: %s ; %s", out1, out2)
    else:
        logger.warning("Skipping RAW ratios (missing config).")

    # ---------- 1734-band ratios ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        ext_stats = _gather_by_group_ext_for_stats(sample_names, groups, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH)
        r_1734_ai_WT_s,  r_1734_ai_KO_s  = ext_stats["1734_ai"]
        r_1734_aii_WT_s, r_1734_aii_KO_s = ext_stats["1734_aii"]

        row_1734_ai  = cliffs_only_stats("1734/AmideI",  r_1734_ai_WT_s,  r_1734_ai_KO_s);  stats_rows.append(row_1734_ai)
        row_1734_aii = cliffs_only_stats("1734/AmideII", r_1734_aii_WT_s, r_1734_aii_KO_s); stats_rows.append(row_1734_aii)

        (WT_1734_ai, KO_1734_ai), (WT_1734_aii, KO_1734_aii) = _gather_per_sample_ext(sample_names, groups, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH)

        ylim_1734_ai  = MANUAL_YLIMS["1734_amideI"]  if USE_MANUAL_YLIMS else _auto_ylim(r_1734_ai_WT_s,  r_1734_ai_KO_s,  log=USE_LOG_Y_FOR["1734_amideI"])
        ylim_1734_aii = MANUAL_YLIMS["1734_amideII"] if USE_MANUAL_YLIMS else _auto_ylim(r_1734_aii_WT_s, r_1734_aii_KO_s, log=USE_LOG_Y_FOR["1734_amideII"])

        out_1734_ai  = os.path.join(save_dir, "ratios_1734_over_AmideI_by_sample_hues.png")
        out_1734_aii = os.path.join(save_dir, "ratios_1734_over_AmideII_by_sample_hues.png")

        _strip_plot_by_sample(
            WT_1734_ai, KO_1734_ai,
            ylabel="1734 / Amide I ratio (per pixel; top-k; normalized: %s)" % NORM_MODE,
            title=f"1734 / Amide I – WT vs KO (distinct hues) @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=out_1734_ai, log_y=USE_LOG_Y_FOR["1734_amideI"], ylim=ylim_1734_ai,
            stats_text=format_stats_for_title(row_1734_ai)
        )
        _strip_plot_by_sample(
            WT_1734_aii, KO_1734_aii,
            ylabel="1734 / Amide II ratio (per pixel; top-k; normalized: %s)" % NORM_MODE,
            title=f"1734 / Amide II – WT vs KO (distinct hues) @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=out_1734_aii, log_y=USE_LOG_Y_FOR["1734_amideII"], ylim=ylim_1734_aii,
            stats_text=format_stats_for_title(row_1734_aii)
        )
        logger.info("Saved 1734-band ratio plots: %s ; %s", out_1734_ai, out_1734_aii)
    else:
        logger.warning("Skipping 1734-band ratios (missing config).")

    # ---------- SINGLE-BAND intensity: 1734 + Amide I + Amide II ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        # 1734
        center_band = BAND_1734_CENTER
        halfwidth   = window

        WT_band_s = []; KO_band_s = []
        for n in groups["WT"]:
            v = _collect_band_for_sample(n, center_band, halfwidth=halfwidth)
            if v is not None: WT_band_s.append(v)
        for n in groups["KO"]:
            v = _collect_band_for_sample(n, center_band, halfwidth=halfwidth)
            if v is not None: KO_band_s.append(v)
        WT_band_s = np.concatenate(WT_band_s) if WT_band_s else np.array([])
        KO_band_s = np.concatenate(KO_band_s) if KO_band_s else np.array([])
        WT_band_s = _cap_group(WT_band_s, "single_band", seed=11)
        KO_band_s = _cap_group(KO_band_s, "single_band", seed=12)

        row_band = cliffs_only_stats(f"Band@{center_band:.1f}", WT_band_s, KO_band_s); stats_rows.append(row_band)

        WT_band_dict, KO_band_dict = _gather_per_sample_band(sample_names, groups, center_cm1=center_band, halfwidth=halfwidth)

        ylim_band = MANUAL_YLIMS["single_band"] if USE_MANUAL_YLIMS else _auto_ylim(WT_band_s, KO_band_s, log=False)
        out_band = os.path.join(save_dir, f"band_{int(center_band)}cm-1_by_sample_hues.png")
        _strip_plot_by_sample(
            WT_band_dict, KO_band_dict,
            ylabel=f"Absorbance at {center_band:.1f} cm$^{{-1}}$ (top-k; normalized: {NORM_MODE})",
            title=f"Single-band intensity — WT vs KO (distinct hues) @ {center_band:.1f} cm$^{{-1}}$",
            out_png=out_band, log_y=False, ylim=ylim_band,
            stats_text=format_stats_for_title(row_band)
        )
        logger.info("Saved single-band 1734 plot: %s", out_band)

        # Amide I (1655) and Amide II (1545)
        for label, center in [("amideI", BAND_AMIDE_I), ("amideII", BAND_AMIDE_II)]:
            WT_s = []; KO_s = []
            for n in groups["WT"]:
                v = _collect_band_for_sample(n, center, halfwidth=window)
                if v is not None: WT_s.append(v)
            for n in groups["KO"]:
                v = _collect_band_for_sample(n, center, halfwidth=window)
                if v is not None: KO_s.append(v)
            WT_s = np.concatenate(WT_s) if WT_s else np.array([])
            KO_s = np.concatenate(KO_s) if KO_s else np.array([])
            WT_s = _cap_group(WT_s, label, seed=21)
            KO_s = _cap_group(KO_s, label, seed=22)

            row = cliffs_only_stats(f"{label}@{center:.1f}", WT_s, KO_s); stats_rows.append(row)

            WT_d, KO_d = _gather_per_sample_band(sample_names, groups, center_cm1=center, halfwidth=window)
            ylim_sb = MANUAL_YLIMS[label] if USE_MANUAL_YLIMS else _auto_ylim(WT_s, KO_s, log=USE_LOG_Y_FOR[label])
            out_sb = os.path.join(save_dir, f"band_{label}_{int(center)}cm-1_by_sample_hues.png")
            _strip_plot_by_sample(
                WT_d, KO_d,
                ylabel=f"{label.replace('amide','Amide ')} intensity (top-k; normalized: {NORM_MODE})",
                title=f"{label.replace('amide','Amide ').title()} — WT vs KO (distinct hues) @ {center:.1f} cm$^{{-1}}$",
                out_png=out_sb, log_y=USE_LOG_Y_FOR[label], ylim=ylim_sb,
                stats_text=format_stats_for_title(row)
            )
            logger.info("Saved %s single-band plot: %s", label, out_sb)
    else:
        logger.warning("Skipping single-band plots (missing config).")

    # ---------- Save table (Cliff's only) ----------
    if stats_rows:
        for r in stats_rows:
            logger.info("%s: n_wt=%d n_ko=%d  δ=%+.3f (CI[%.3f, %.3f])  medΔ=%.4g  KO>WT≈%.1f%% WT>KO≈%.1f%%",
                        r["metric"], r["n_wt"], r["n_ko"], r["cliffs_delta"],
                        r["cliffs_delta_lo"], r["cliffs_delta_hi"],
                        r["median_diff"], r["p_ko_gt_wt"]*100, r["p_wt_gt_ko"]*100)
        save_stats_csv(stats_rows, out_prefix + "_WT_vs_KO_cliffs_only.csv")

    # ---------- (Optional) CENTROID leaning ----------
    sample_paths = [
        # r"...\per-sample\T1.npz", r"...\per-sample\T3.npz", ...
    ]
    if sample_paths and len(sample_paths) == len(sample_names) and groups.get("WT") and groups.get("KO") and selected_clusters:
        _ = centroid_angles_and_margin(
            sample_paths, sample_names, groups, selected_clusters,
            subsample_size=2000, reducer="median"
        )
    else:
        logger.warning("Skipping centroid-angle step (missing per-sample sec-deriv input paths or config).")
