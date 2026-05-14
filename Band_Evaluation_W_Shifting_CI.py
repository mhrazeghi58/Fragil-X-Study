# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 19:30:49 2025

@author: hrazeghikondela
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 03:09:00 2025

@author: hrazeghikondela
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging, time, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from contextlib import contextmanager
from scipy.signal import savgol_filter
from scipy import stats   # used for rankdata
from scipy.stats import mannwhitneyu

# ============================== PATHS / CONFIG ==============================
cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L\UMAP_clustering_8Cluster"
cube_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
save_dir    = os.path.join(cube_dir, "WT_KO_bands_wCI")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "WT_KO")

sample_names = ["T1","T2","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

# ---------------------------------------------------------------------------
# ### ANIMAL-LEVEL: map each section/sample (e.g., "T1") to an animal ID
#     (edit if multiple sections belong to the same animal)
SAMPLE_TO_ANIMAL = {
    "T1":"WT_01","T2":"WT_08","T3":"WT_02","T6":"WT_03","T17":"WT_04","T19":"WT_05","T20":"WT_06","T21":"WT_07",
    "T10":"KO_01","T11":"KO_02","T12":"KO_03","T13":"KO_04","T14":"KO_05","T15":"KO_06","T22":"KO_07",
}
ANIMAL_TO_GROUP = {aid: ("WT" if aid.startswith("WT_") else "KO") for aid in SAMPLE_TO_ANIMAL.values()}
# ---------------------------------------------------------------------------

# ---------------- Cluster selection (CA) ----------------
#selected_clusters = {
#    "T1":[0,1,2,3,4,5,6,7],
#    "T2":[0,1,2,3,4,7],
 #   "T3":[0,1,2,3,5,6,7],
#    "T6":[0,1,3,5,6,7],
#    "T17":[0,1,2,3,5,6,7],
#    "T19":[0,1,3,4.5,7],
#    "T20":[0,1,2,3,4,5,7],
#    "T21":[1,2,4,5,6,7],
#    "T10":[1,2,3,6,5,7],
#    "T11":[0,1,2,5,6,7],
#    "T12":[0,1,2,4,6,7],
#    "T13":[0,1,3,4,5,6,7],
#    "T14":[0,1,2,4,5,6,7],
#    "T15":[0,2,3,4,5,6,7],
#    "T22":[0,1,2,3,4,6],
#}

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


# ---------------- Cluster selection (Left) ----------------
#selected_clusters = {
#    "T1" :[0,1,3,4,6,7],
#    "T2" : [0,1,2],
#    "T3" : [0,1,4,5],
#    "T6" : [0,1,3,4,6],
#    "T19": [0,2,3.6,7],
#   "T20": [0,1,2,4,5,7],
#   "T21": [1,2,3,4],
#    "T17": [0,1,2,3,5,6],
#    "T10": [0,2,4,5,6,7],
#    "T11": [0,1,2,3,6,7],
#    "T12": [0,1,2,3,5,7],
#    "T13": [1,2,5,6,7],
#    "T14": [0,1,4,5,7],# 
#    "T15": [0,2,3,4,5,7],
#    "T22": [0,1,4,5,6],
#}
# --------- Spectral / band settings ---------
savgol_window = 11
polyorder     = 3
try:
    window
except NameError:
    window = 30

# Use “top-k peaks in window” integration
TOPK_PER_BAND = 1  # mean of the top-k values inside the window (per pixel)

# ---- Preprocess policy: shift ONLY (no L2 normalization) ----
NORM_MODE = "none"       # <— L2 normalization removed
SHIFT_MIN_FLOOR = 0.2
NORM_EPS  = 1e-12  # kept for compatibility (unused when NORM_MODE="none")

# Bands (cm-1)
BAND_1734_CENTER    = 1734.0
BAND_1734_HALFWIDTH = 13
BAND_AMIDE_I        = 1655.0
BAND_AMIDE_II       = 1545.0
BAND_CH2            = 1464.0
BAND_CH3            = 1375.0

# Which ratios to compute/plot
RATIOS_TO_COMPUTE = {
    "CH2/CH3":       (BAND_CH2,     BAND_CH3),
    "1734/AmideI":   (BAND_1734_CENTER, BAND_AMIDE_I),
    "1734/AmideII":  (BAND_1734_CENTER, BAND_AMIDE_II),
    "CH2/AmideI":    (BAND_CH2,     BAND_AMIDE_I),
    "CH3/AmideI":    (BAND_CH3,     BAND_AMIDE_I),
}

# ===================== MANUAL AXES & PIXEL CAPS =======================
USE_MANUAL_YLIMS = True
MANUAL_YLIMS = {
    "CH2/CH3":         None,
    "AmideI/AmideII":  None,
    "1734/AmideI":     None,
    "1734/AmideII":    None,
    "CH2/AmideI":      None,
    "CH3/AmideI":      None,
    "single_band":     None,
    "amideI":          None,
    "amideII":         None,
    "1734":            None,
}
USE_LOG_Y_FOR = {
    "AmideI/AmideII": False,
    "1734/AmideI":    False,
    "1734/AmideII":   False,
    "CH2/AmideI":     False,
    "CH3/AmideI":     False,
    "amideI":         False,
    "amideII":        False,
    "1734":           False,
}

PIXEL_CAP_PER_GROUP = {
    "CH2/CH3":         10_000,
    "AmideI/AmideII":  10_000,
    "1734/AmideI":     10_000,
    "1734/AmideII":    10_000,
    "CH2/AmideI":      10_000,
    "CH3/AmideI":      10_000,
    "single_band":     10_000,
    "amideI":          10_000,
    "amideII":         10_000,
    "1734":            10_000,
}
MAX_POINTS_PER_SAMPLE = 500

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
BASE_WT = "#2ca02c"
BASE_KO = "#d62728"

DOT_SIZE     = 5
JITTER_SCALE = 0.06
USE_VIOLIN_BACKGROUND = True
SAVE_PDF_ALSO = True
SHOW_SAMPLE_LEGEND = False   # <— hide per-sample hue legend

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11

VIOLIN_WT_COLOR = (0.3, 0.7, 0.3, 0.12)
VIOLIN_KO_COLOR = (0.9, 0.3, 0.3, 0.12)

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

# ========================== DATA LOADING / PREPROCESS ==================
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

# ----- Shift (min≥floor); NO L2 normalization -----
def _shift_spectra_min_floor(cube, floor=SHIFT_MIN_FLOOR):
    C = cube.astype(np.float64, copy=True)
    mins = np.min(C, axis=2, keepdims=True)
    add = np.clip(floor - mins, 0.0, None)
    return C + add

def _preprocess_cube(cube, wns):
    """
    Shift spectra upward so per-pixel min >= SHIFT_MIN_FLOOR.
    (L2 normalization removed.)
    """
    C = _shift_spectra_min_floor(cube, floor=SHIFT_MIN_FLOOR)
    logger.debug("Applied per-pixel shift (min floor=%.3g). No L2 normalization.", SHIFT_MIN_FLOOR)
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
    nwin = slab.shape[2]
    kk = int(max(1, min(k, nwin)))

    H, W = slab.shape[:2]
    flat = slab.reshape(-1, nwin)
    idx = np.argpartition(flat, -kk, axis=1)[:, -kk:]
    topk = np.take_along_axis(flat, idx, axis=1)
    out_flat = np.nanmean(topk, axis=1)
    return out_flat.reshape(H, W)

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
    ax.text(0.5, 0.94, text, ha="center", va="top", transform=ax.transAxes, fontsize=10)

# --- Per-group sample color maps ---
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
    _apply_pub_style()

    WT_all = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    if WT_all.size == 0 and KO_all.size == 0:
        logger.warning("Nothing to plot for %s (both WT and KO empty)", title)
        return

    if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
        WTv = _mask_for_ylim(WT_all, ylim)
        KOv = _mask_for_ylim(KO_all, ylim)
    else:
        WTv, KOv = WT_all, KO_all

    if WTv.size == 0 and KOv.size == 0:
        logger.warning("All values fell outside ylim for %s; skipping plot.", title)
        return

    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    rng = np.random.default_rng(0)

    draw_violins = USE_VIOLIN_BACKGROUND and (WTv.size > 0) and (KOv.size > 0)
    if draw_violins:
        parts = ax.violinplot([WTv, KOv], positions=[0, 1], widths=0.6,
                               showmeans=False, showmedians=False, showextrema=False)
        parts["bodies"][0].set_facecolor(VIOLIN_WT_COLOR); parts["bodies"][0].set_edgecolor("none")
        parts["bodies"][1].set_facecolor(VIOLIN_KO_COLOR); parts["bodies"][1].set_edgecolor("none")

    wt_keys = list(WT_dict.keys()); ko_keys = list(KO_dict.keys())
    wt_col_map, ko_col_map = _group_sample_colors(wt_keys, ko_keys)

    for s, arr in WT_dict.items():
        if arr.size == 0: continue
        arrv = _mask_for_ylim(arr, ylim) if (ylim is not None and TRIM_FOR_PLOTTING_ONLY) else arr
        if arrv.size == 0: continue
        xs = 0 + rng.normal(0, JITTER_SCALE, size=arrv.size)
        ax.scatter(xs, arrv, s=DOT_SIZE, alpha=0.6, c=[wt_col_map.get(s)], edgecolors="none", rasterized=True)

    for s, arr in KO_dict.items():
        if arr.size == 0: continue
        arrv = _mask_for_ylim(arr, ylim) if (ylim is not None and TRIM_FOR_PLOTTING_ONLY) else arr
        if arrv.size == 0: continue
        xs = 1 + rng.normal(0, JITTER_SCALE, size=arrv.size)
        ax.scatter(xs, arrv, s=DOT_SIZE, alpha=0.6, c=[ko_col_map.get(s)], edgecolors="none", rasterized=True)

    def _median_iqr(x, xpos):
        if x.size == 0: return
        med = np.median(x); q1, q3 = np.quantile(x, [0.05, 0.95])
        ax.scatter([xpos], [med], s=38, c="white", edgecolors="black", linewidths=0.7, zorder=3)
        ax.vlines(xpos, q1, q3, colors="black", linewidth=1.2, zorder=3)
    _median_iqr(WTv, 0); _median_iqr(KOv, 1)

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
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _annotate(ax, stats_text)

    if False:  # SHOW_SAMPLE_LEGEND is False by default
        handles = []
        for s in wt_keys:
            handles.append(mpatches.Patch(color=wt_col_map[s], label=s))
        for s in ko_keys:
            handles.append(mpatches.Patch(color=ko_col_map[s], label=s))
        if handles:
            ax.legend(handles=handles, title="Samples",
                      bbox_to_anchor=(1.02, 1), loc="upper left",
                      frameon=False, ncol=1)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    if SAVE_PDF_ALSO:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".pdf", bbox_inches="tight")
    plt.close(fig)

# ========================== STATISTICS (Cliff's + MWU) =================
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

def save_stats_csv(rows, path):
    headers = ["metric","n_wt","n_ko","mean_wt","mean_ko","median_wt","median_ko","sd_wt","sd_ko",
               "cliffs_delta","cliffs_delta_lo","cliffs_delta_hi",
               "median_diff","median_diff_lo","median_diff_hi",
               "p_wt_gt_ko","p_ko_gt_wt","p_tie","mwu_U","mwu_p_two_sided"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for r in rows:
            w.writerow([r.get(h,"") for h in headers])
    logger.info("Saved stats table: %s", path)

def format_stats_for_title(row):
    d  = row["cliffs_delta"]
    pK = row["p_ko_gt_wt"] * 100.0
    pW = row["p_wt_gt_ko"] * 100.0
    return f"Cliff's δ={d:+.2f}; KO>WT≈{pK:.0f}% vs WT>KO≈{pW:.0f}%"

# ========================== GATHER HELPERS ============================
def _collect_per_pixel_values(sample, bands_to_get):
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
    cube = _preprocess_cube(cube, wns)

    per_band = {}
    for key, (center, halfw) in bands_to_get.items():
        band_img = _band_value(cube, wns, float(center), halfw if halfw is not None else window)
        vals = band_img[mask2d].ravel()
        ok = np.isfinite(vals) & (vals > 0)
        vals = vals[ok]
        if len(vals) > MAX_POINTS_PER_SAMPLE:
            rng = np.random.default_rng(abs(hash((sample, key))) % (2**32))
            take = rng.choice(len(vals), size=MAX_POINTS_PER_SAMPLE, replace=False)
            vals = vals[take]
        per_band[key] = vals
    return per_band

def _collect_ratios_for_sample(sample):
    bands = _collect_per_pixel_values(sample, {
        "ch2":   (BAND_CH2,     window),
        "ch3":   (BAND_CH3,     window),
        "amideI":(BAND_AMIDE_I, window),
        "amideII":(BAND_AMIDE_II, window),
    })
    if bands is None: return None, None
    ch2, ch3, amideI, amideII = bands["ch2"], bands["ch3"], bands["amideI"], bands["amideII"]

    vR1_den, _ = _denom_guard(ch3)
    nR1_num, _ = _num_guard(ch2)
    vR2_den, _ = _denom_guard(amideII)
    nR2_num, _ = _num_guard(amideI)

    L = min(len(ch2), len(ch3), len(amideI), len(amideII))
    if L == 0: return None, None
    ch2 = ch2[:L]; ch3 = ch3[:L]; amideI = amideI[:L]; amideII = amideII[:L]

    valid1 = np.isfinite(ch2) & np.isfinite(ch3) & (ch2>0) & (ch3>0)
    valid2 = np.isfinite(amideI) & np.isfinite(amideII) & (amideI>0) & (amideII>0)

    r1 = np.full(L, np.nan); r2 = np.full(L, np.nan)
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        r1[valid1] = ch2[valid1] / ch3[valid1]
        r2[valid2] = amideI[valid2] / amideII[valid2]
    if APPLY_RATIO_HARD_CAP:
        r1 = np.clip(r1, 0, RATIO_HARD_CAP)
        r2 = np.clip(r2, 0, RATIO_HARD_CAP)
    r1 = r1[np.isfinite(r1) & (r1>0)]
    r2 = r2[np.isfinite(r2) & (r2>0)]
    return r1, r2

def _collect_ratios_for_sample_ext(sample, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH):
    per = _collect_per_pixel_values(sample, {
        "ch2":   (BAND_CH2,     window),
        "ch3":   (BAND_CH3,     window),
        "amideI":(BAND_AMIDE_I, window),
        "amideII":(BAND_AMIDE_II, window),
        "b1734": (center_1734,  halfwidth_1734),
    })
    if per is None: return (None, None, None, None)
    ch2, ch3, amideI, amideII, b1734 = per["ch2"], per["ch3"], per["amideI"], per["amideII"], per["b1734"]

    L = min(len(ch2), len(ch3), len(amideI), len(amideII), len(b1734))
    if L == 0: return (None, None, None, None)
    ch2, ch3, amideI, amideII, b1734 = ch2[:L], ch3[:L], amideI[:L], amideII[:L], b1734[:L]

    r1 = ch2 / ch3
    r2 = amideI / amideII
    r3 = b1734 / amideI
    r4 = b1734 / amideII

    if APPLY_RATIO_HARD_CAP:
        r1 = np.clip(r1, 0, RATIO_HARD_CAP)
        r2 = np.clip(r2, 0, RATIO_HARD_CAP)
        r3 = np.clip(r3, 0, RATIO_HARD_CAP)
        r4 = np.clip(r4, 0, RATIO_HARD_CAP)

    r1 = r1[np.isfinite(r1) & (r1>0)]
    r2 = r2[np.isfinite(r2) & (r2>0)]
    r3 = r3[np.isfinite(r3) & (r3>0)]
    r4 = r4[np.isfinite(r4) & (r4>0)]
    return r1, r2, r3, r4

def _collect_band_for_sample(sample, center_cm1, halfwidth=None):
    per = _collect_per_pixel_values(sample, {"band": (center_cm1, halfwidth if halfwidth is not None else window)})
    if per is None: return None
    vals = per["band"]
    return vals

# ---- Helpers to gather for STATS (concatenated) ----
def _cap_group(arr, metric_key, seed):
    cap = PIXEL_CAP_PER_GROUP.get(metric_key, 30_000)
    if len(arr) <= cap: return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(len(arr), size=cap, replace=False)]

def _concat(lst): return np.concatenate(lst) if lst else np.array([])

# ===================== SIGNED 2nd-DERIV & CENTROID LEANING ============
def build_signed_secderiv_matrices(sample_paths, sample_names, selected_clusters,
                                   window=savgol_window, poly=polyorder,
                                   subsample_size=2000, random_state=SEED,
                                   do_normalize=False):
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
            subsample_size=subsample_size, random_state=SEED, do_normalize=False
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

# ========================== NEW: ANIMAL-LEVEL HELPERS ==================
def _sample_median_for_metric(sample, metric_key):
    """
    Return per-sample median (across selected pixels) for a named metric.
    metric_key in:
      'band_1734', 'band_amideI', 'band_amideII',
      'CH2/CH3', 'AmideI/AmideII', '1734/AmideI', '1734/AmideII', 'CH2/AmideI', 'CH3/AmideI'
    """
    if metric_key == "band_1734":
        v = _collect_band_for_sample(sample, BAND_1734_CENTER, halfwidth=BAND_1734_HALFWIDTH)
        return np.nanmedian(v) if (v is not None and v.size) else np.nan
    if metric_key == "band_amideI":
        v = _collect_band_for_sample(sample, BAND_AMIDE_I, halfwidth=window)
        return np.nanmedian(v) if (v is not None and v.size) else np.nan
    if metric_key == "band_amideII":
        v = _collect_band_for_sample(sample, BAND_AMIDE_II, halfwidth=window)
        return np.nanmedian(v) if (v is not None and v.size) else np.nan

    r1, r2, r3, r4 = _collect_ratios_for_sample_ext(sample)
    # r1=CH2/CH3, r2=AmideI/AmideII, r3=1734/AmideI, r4=1734/AmideII
    if metric_key == "CH2/CH3":        return np.nanmedian(r1) if (r1 is not None and r1.size) else np.nan
    if metric_key == "AmideI/AmideII": return np.nanmedian(r2) if (r2 is not None and r2.size) else np.nan
    if metric_key == "1734/AmideI":    return np.nanmedian(r3) if (r3 is not None and r3.size) else np.nan
    if metric_key == "1734/AmideII":   return np.nanmedian(r4) if (r4 is not None and r4.size) else np.nan

    # CH2/AmideI and CH3/AmideI computed directly
    per = _collect_per_pixel_values(sample, {
        "ch2": (BAND_CH2, window), "ch3": (BAND_CH3, window), "amideI": (BAND_AMIDE_I, window)
    })
    if per is None or min(map(len, per.values())) == 0:
        return np.nan
    L = min(len(per["ch2"]), len(per["ch3"]), len(per["amideI"]))
    ch2_ai = per["ch2"][:L] / per["amideI"][:L]
    ch3_ai = per["ch3"][:L] / per["amideI"][:L]
    ch2_ai = ch2_ai[np.isfinite(ch2_ai) & (ch2_ai>0)]
    ch3_ai = ch3_ai[np.isfinite(ch3_ai) & (ch3_ai>0)]
    if metric_key == "CH2/AmideI": return np.nanmedian(ch2_ai) if ch2_ai.size else np.nan
    if metric_key == "CH3/AmideI": return np.nanmedian(ch3_ai) if ch3_ai.size else np.nan
    return np.nan

def compute_animal_level_medians(sample_names, sample_to_animal, animal_to_group):
    """
    Returns:
      per_animal_rows: list of dicts: {animal, group, metric, animal_median}
      stats_rows: list of dicts: per-metric stats at animal level
    """
    metrics = [
        "band_1734","band_amideI","band_amideII",
        "CH2/CH3","AmideI/AmideII","1734/AmideI","1734/AmideII","CH2/AmideI","CH3/AmideI"
    ]
    # gather per-sample medians
    per_animal_sections = {}  # metric -> animal -> list of section medians
    with LogTimer("Compute per-sample medians for animal-level aggregation"):
        for s in sample_names:
            animal = sample_to_animal.get(s)
            if animal is None:
                logger.warning("Sample %s not mapped to animal; skipping.", s)
                continue
            for m in metrics:
                sm = _sample_median_for_metric(s, m)
                if not np.isfinite(sm):  # skip if no data
                    continue
                per_animal_sections.setdefault(m, {}).setdefault(animal, []).append(float(sm))

    # collapse to per-animal medians
    per_animal_rows = []
    for m, d in per_animal_sections.items():
        for animal, section_meds in d.items():
            if not section_meds: continue
            g = animal_to_group.get(animal)
            if g not in ("WT","KO"):
                continue
            per_animal_rows.append({
                "animal": animal,
                "group": g,
                "metric": m,
                "animal_median": float(np.median(section_meds))
            })

    # stats per metric across animals
    stats_rows = []
    metrics_pretty = {
        "band_1734": f"1734@{BAND_1734_CENTER:.1f}",
        "band_amideI": f"AmideI@{BAND_AMIDE_I:.1f}",
        "band_amideII": f"AmideII@{BAND_AMIDE_II:.1f}",
        "CH2/CH3": "CH2/CH3",
        "AmideI/AmideII": "AmideI/AmideII",
        "1734/AmideI": "1734/AmideI",
        "1734/AmideII": "1734/AmideII",
        "CH2/AmideI": "CH2/AmideI",
        "CH3/AmideI": "CH3/AmideI",
    }

    with LogTimer("Animal-level stats (Cliff's δ CI + Mann–Whitney)"):
        for m in metrics:
            vals = [r for r in per_animal_rows if r["metric"] == m]
            if not vals: 
                continue
            WT = np.array([r["animal_median"] for r in vals if r["group"]=="WT"])
            KO = np.array([r["animal_median"] for r in vals if r["group"]=="KO"])
            if WT.size==0 or KO.size==0:
                continue

            delta = cliffs_delta(WT, KO)
            d_lo, d_hi = bootstrap_ci(cliffs_delta, WT, KO, nboot=10000, rng=rng_global)
            p_wt, p_ko, p_tie = pairwise_preference(WT, KO, max_pairs=200_000, rng=rng_global)
            U, p = mannwhitneyu(WT, KO, alternative="two-sided")

            stats_rows.append({
                "metric": metrics_pretty[m],
                "n_wt": int(WT.size), "n_ko": int(KO.size),
                "mean_wt": float(np.mean(WT)), "mean_ko": float(np.mean(KO)),
                "median_wt": float(np.median(WT)), "median_ko": float(np.median(KO)),
                "sd_wt": float(np.std(WT, ddof=1)) if WT.size>1 else np.nan,
                "sd_ko": float(np.std(KO, ddof=1)) if KO.size>1 else np.nan,
                "cliffs_delta": float(delta), "cliffs_delta_lo": float(d_lo), "cliffs_delta_hi": float(d_hi),
                "median_diff": float(np.median(WT) - np.median(KO)),
                "median_diff_lo": np.nan, "median_diff_hi": np.nan,  # (optional: add BCa CI)
                "p_wt_gt_ko": float(p_wt), "p_ko_gt_wt": float(p_ko), "p_tie": float(p_tie),
                "mwu_U": float(U), "mwu_p_two_sided": float(p),
            })

    return per_animal_rows, stats_rows

# ========================== MAIN RUN ==================================
if __name__ == "__main__":
    logger.info("Preprocess: shift floor=%.3g; normalization=%s",
                SHIFT_MIN_FLOOR, ("NONE" if NORM_MODE.lower()=="none" else NORM_MODE))
    stats_rows = []

    # ---------- Base ratios (CH2/CH3 & AmideI/II) ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        # (existing pixel-level gathering & plots omitted here to save space in this message;
        #  your original sections remain unchanged above in your own file.)

        # ---------- SINGLE-BAND/EXTENDED (existing plotting code remains) ----------
        # ... (keep your existing plotting code as-is) ...

        logger.info("Pixel-level plotting sections executed (if kept).")
    else:
        logger.warning("Skipping base ratios (missing config).")

    # ---------- OPTIONAL: CENTROID leaning (unchanged) ----------
    sample_paths = []  # keep empty unless you have per-sample sec-deriv inputs
    if sample_paths and len(sample_paths) == len(sample_names) and groups.get("WT") and groups.get("KO") and selected_clusters:
        _ = centroid_angles_and_margin(
            sample_paths, sample_names, groups, selected_clusters,
            subsample_size=2000, reducer="median"
        )
    else:
        logger.warning("Skipping centroid-angle step (missing per-sample sec-deriv input paths or config).")

    # ======================= NEW: ANIMAL-LEVEL ==========================
    per_animal_rows, animal_stats = compute_animal_level_medians(
        sample_names, SAMPLE_TO_ANIMAL, ANIMAL_TO_GROUP
    )

    # Save per-animal medians (one row per animal×metric)
    per_animal_csv = out_prefix + "_per_animal_medians.csv"
    with open(per_animal_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["animal","group","metric","animal_median"])
        for r in sorted(per_animal_rows, key=lambda x: (x["metric"], x["group"], x["animal"])): 
            w.writerow([r["animal"], r["group"], r["metric"], f"{r['animal_median']:.6g}"])
    logger.info("Saved per-animal medians: %s", per_animal_csv)

    # Save animal-level stats with CI + MWU p
    animal_stats_csv = out_prefix + "_animal_level_stats.csv"
    save_stats_csv(animal_stats, animal_stats_csv)

    # Console summary
    for r in animal_stats:
        logger.info("%s  nWT=%d nKO=%d  δ=%+.3f (95%% CI %.3f..%.3f)  MWU p=%.4f  KO>WT≈%.0f%%  WT>KO≈%.0f%%",
                    r["metric"], r["n_wt"], r["n_ko"], r["cliffs_delta"],
                    r["cliffs_delta_lo"], r["cliffs_delta_hi"],
                    r["mwu_p_two_sided"], r["p_ko_gt_wt"]*100, r["p_wt_gt_ko"]*100)
# ================== PLOT: per-animal medians with p + δ CI ==================

import pandas as pd
animal_rows_df = pd.read_csv(per_animal_csv)         # columns: animal, group, metric, animal_median
stats_df       = pd.read_csv(animal_stats_csv)       # columns include: metric, mwu_p_two_sided, cliffs_delta, ...

# (2) Small helpers for Cliff's δ on animal-level medians
def _delta(a, b):
    """Cliff's delta on two 1D arrays (wrapper over cliffs_delta defined earlier)."""
    return float(cliffs_delta(np.asarray(a, float), np.asarray(b, float)))

def _bootstrap_ci_delta(a, b, nboot=10_000, alpha=0.05, rng=np.random.default_rng(0)):
    """Percentile bootstrap CI for Cliff's delta."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return (np.nan, np.nan)
    vals = []
    for _ in range(nboot):
        aa = a[rng.integers(0, na, na)]
        bb = b[rng.integers(0, nb, nb)]
        vals.append(_delta(aa, bb))
    lo, hi = np.percentile(vals, [100*alpha/2, 100*(1-alpha/2)])
    return float(lo), float(hi)

# (3) Plot function (Option-B annotation; robust animal codes)
# ================== PLOT: per-animal medians with ONLY p-value ==================
# Place this AFTER you save:
#   per_animal_csv = out_prefix + "_per_animal_medians.csv"
#   animal_stats_csv = out_prefix + "_animal_level_stats.csv"

import pandas as pd


# Load the just-saved tables
animal_rows_df = pd.read_csv(per_animal_csv)   # columns: animal, group, metric, animal_median
stats_df       = pd.read_csv(animal_stats_csv) # includes: metric, mwu_p_two_sided, ...

def _plot_animal_medians_one(metric_key, nice_name, animal_rows_df, stats_df, outdir):
    """Box + jittered scatter of per-animal medians (WT vs KO) with ONLY p-value."""
    _apply_pub_style()

    sub = animal_rows_df[animal_rows_df["metric"] == metric_key].copy()
    if sub.empty:
        logger.warning("No animal-level rows for %s", metric_key)
        return

    wt = sub.loc[sub["group"] == "WT", "animal_median"].astype(float).to_numpy()
    ko = sub.loc[sub["group"] == "KO", "animal_median"].astype(float).to_numpy()
    if wt.size == 0 or ko.size == 0:
        logger.warning("[skip] %s: nWT=%d, nKO=%d", metric_key, wt.size, ko.size)
        return

    # Prefer precomputed p from stats_df; otherwise compute
    row = stats_df.loc[stats_df["metric"] == nice_name]
    if len(row) == 1 and np.isfinite(row["mwu_p_two_sided"].iloc[0]):
        p = float(row["mwu_p_two_sided"].iloc[0])
    else:
        try:
            _, p = mannwhitneyu(wt, ko, alternative="two-sided")
            p = float(p)
        except Exception:
            p = np.nan

    fig, ax = plt.subplots(figsize=(5.0, 4.6))

    # Boxplots (no fliers)
    bp = ax.boxplot([wt, ko], positions=[0, 1], widths=0.5,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", linewidth=1.2),
                    boxprops=dict(linewidth=1.1, edgecolor="#4B5563"),
                    whiskerprops=dict(linewidth=1.0, color="#4B5563"),
                    capprops=dict(linewidth=1.0, color="#4B5563"))
    bp["boxes"][0].set_facecolor((0.3, 0.7, 0.3, 0.20))  # WT
    bp["boxes"][1].set_facecolor((0.9, 0.3, 0.3, 0.20))  # KO

    # Jittered scatter (alpha=0.30), NO labels
    rng = np.random.default_rng(abs(hash(metric_key)) % (2**32))
    j = 0.20
    ax.scatter(0 + rng.uniform(-j, j, wt.size), wt, s=36, color=BASE_WT, alpha=0.90, edgecolors="none")
    ax.scatter(1 + rng.uniform(-j, j, ko.size), ko, s=36, color=BASE_KO, alpha=0.90, edgecolors="none")

    # Cosmetics
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1], ["WT", "KO"])
    ax.set_title(nice_name, pad=12)
    ax.set_ylabel(nice_name)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, ls="--", lw=0.5, alpha=0.4)

    # Annotation: ONLY p-value (Option B: pinned top-left in axes)
    p_txt = ("n/a" if (p is None or not np.isfinite(p)) else f"{p:.3f}")
    ax.text(0.9, 0.99, f"MWU p = {p_txt}",
            ha="right", va="top", transform=ax.transAxes,
            fontsize=BASE_FONTSIZE-1, color="#374151",
            bbox=dict(facecolor="white", edgecolor="#D1D5DB",
                      boxstyle="round,pad=0.25", alpha=0.9))

    fig.tight_layout()
    fname = os.path.join(outdir, f"animal_level_{metric_key.replace('/','-')}_box_scatter.png")
    fig.savefig(fname, dpi=600, bbox_inches="tight")
    if SAVE_PDF_ALSO:
        fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved animal-level box+scatter: %s", fname)

# Pretty names to match your stats CSV "metric" column
pretty_names = {
    "band_1734":   f"1734@{BAND_1734_CENTER:.1f}",
    "band_amideI": f"AmideI@{BAND_AMIDE_I:.1f}",
    "band_amideII":f"AmideII@{BAND_AMIDE_II:.1f}",
    "CH2/CH3":     "CH2/CH3",
    "AmideI/AmideII": "AmideI/AmideII",
    "1734/AmideI": "1734/AmideI",
    "1734/AmideII":"1734/AmideII",
    "CH2/AmideI":  "CH2/AmideI",
    "CH3/AmideI":  "CH3/AmideI",
}

# Plot for all metrics present in the per-animal table
_present = set(animal_rows_df["metric"].unique())
for mk, nn in pretty_names.items():
    if mk in _present:
        _plot_animal_medians_one(mk, nn, animal_rows_df, stats_df, save_dir)
