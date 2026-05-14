# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 03:09:00 2025
@author: hrazeghikondela
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, logging, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from contextlib import contextmanager
from scipy.signal import savgol_filter
from scipy import stats   # used for rankdata

# ============================== PATHS / CONFIG ==============================
cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R\UMAP_clustering_test"
cube_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"
save_dir    = os.path.join(cube_dir, "WT_KO_bands_sub")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "WT_KO")

sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

# ---------------- Cluster selection (CA-1) ----------------
#selected_clusters = {
#    "T1":[0,1,2,3,4,5,6,7],
#    "T2":[0,1,2,3,4,7],
#    "T3":[0,1,2,3,5,6,7],
#    "T6":[0,1,3,5,6,7],
#    "T17":[0,1,2,3,5,6,7],
#    "T19":[0,1,3,4,5,7],   # fixed stray '4.5' -> '4'
#    "T20":[0,1,2,3,4,5,7],
#    "T21":[1,2,4,5,6,7],
#    "T10":[1,2,3,5,6,7],   # order normalized
#    "T11":[0,1,2,5,6,7],
#    "T12":[0,1,2,4,6,7],
#    "T13":[0,1,3,4,5,6,7],
#    "T14":[0,1,2,4,5,6,7],
#    "T15":[0,2,3,4,5,6,7],
#    "T22":[0,1,2,3,4,6],
#}

# ---------------- Cluster selection (Right) ----------------
selected_clusters = {
    "T1":[1,2,3,4,5,6],
    "T2":[1,2,3,4,5,6],
    "T3":[1,2,3,4,5,6,7],
    "T6":[0,1,3,5,6,7],
    "T17":[1,3,5,6,7],
    "T19":[1,2,3,4.6,7],
    "T20":[0,2,3,5,7],
    "T21":[0,1,4,6,7],
    "T10":[0,1,3,4,5,7],
    "T11":[0,1,3,4,5,6],
    "T12":[0,1,2,4,5,7],
    "T13":[0,1,2,4,6,7],
    "T14":[0,1,3,4,5,6],
   "T15":[0,1,3,4,5,6,7],
    "T22":[0,2,4,6,7],
}

#Left
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

TOPK_PER_BAND = 1  # mean of the top-k values inside the window (per pixel)

# ---- Preprocess: shift ONLY (no L2 normalization) ----
NORM_MODE = "none"
SHIFT_MIN_FLOOR = 0.2
NORM_EPS  = 1e-12  # unused here

# Bands (cm-1)
BAND_1734_CENTER    = 1734.0
BAND_1734_HALFWIDTH = 13
BAND_AMIDE_I        = 1655.0
BAND_AMIDE_II       = 1545.0
BAND_CH2            = 1464.0
BAND_CH3            = 1375.0

# ===================== MANUAL AXES & PIXEL CAPS =======================
USE_MANUAL_YLIMS = True
MANUAL_YLIMS = {
    "CH2/CH3":         (0.2, 3.5),
    "AmideI/AmideII":  (0.2, 4.0),
    "1734/AmideI":     (0.1, 1.25),
    "1734/AmideII":    (0.1, 1.5),
    "CH2/AmideI":      (0.15, 1.25),
    "CH3/AmideI":      (0.15, 1.25),
    "single_band":     None,
    "amideI":          (0.25, 1.75),
    "amideII":         (0.2, 1.4),
    "1734":            (0.2, 0.7),
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

ANNOTATE_STATS_ON_PLOTS = True
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

# Larger points per your request
DOT_SIZE     = 10
JITTER_SCALE = 0.06
USE_VIOLIN_BACKGROUND = True
SAVE_PDF_ALSO = True
SHOW_SAMPLE_LEGEND = False

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11

# numeric-only font sizes (avoids "undefined font size")
FS = {
    "base":   int(BASE_FONTSIZE),
    "axes":   int(BASE_FONTSIZE + 1),
    "title":  int(BASE_FONTSIZE + 3),
    "tick":   int(BASE_FONTSIZE),
    "annot":  10,
    "legend": int(BASE_FONTSIZE),
}

VIOLIN_WT_COLOR = (0.3, 0.7, 0.3, 0.18)  # slightly more opaque for overlay clarity
VIOLIN_KO_COLOR = (0.9, 0.3, 0.3, 0.22)

TRIM_FOR_PLOTTING_ONLY = True
FORCE_NONNEG_BANDS   = False
DENOM_MIN_PCT        = 3.0
DENOM_ABS_FLOOR      = 1e-2
NUM_MIN_PCT          = 15.0
NUM_ABS_FLOOR        = 1e-4
APPLY_RATIO_HARD_CAP = True
RATIO_HARD_CAP       = 10.0

def _apply_pub_style():
    # fallback if Arial unavailable
    try:
        fam = FONT_FAMILY
    except Exception:
        fam = "DejaVu Sans"
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": PUB_DPI,
        "font.family": fam,
        "font.size":   FS["base"],
        "axes.labelsize": FS["axes"],
        "axes.titlesize": FS["title"],
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
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
    C = _shift_spectra_min_floor(cube, floor=SHIFT_MIN_FLOOR)
    logger.debug("Applied per-pixel shift (min floor=%.3g). No L2 normalization.", SHIFT_MIN_FLOOR)
    return C

# --- top-k-mean extractor over a band window ---
def _band_value_topk_mean(cube, wns, center, halfwidth, k=TOPK_PER_BAND):
    if halfwidth is None or halfwidth <= 0:
        halfwidth = 10
    m = (wns >= (center - halfwidth)) & (wns <= (center + halfwidth))
    if not np.any(m):
        H, W = cube.shape[:2]
        return np.full((H, W), np.nan, dtype=float)
    slab = cube[:, :, m]
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
    ax.text(0.5, 0.94, text, ha="center", va="top",
            transform=ax.transAxes, fontsize=FS["annot"])

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

# ------------- Original strip-plot (kept as-is, but bigger points) -------------
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

    # optional violin background
    draw_violins = USE_VIOLIN_BACKGROUND and (WTv.size > 0) and (KOv.size > 0)
    if draw_violins:
        parts = ax.violinplot([WTv, KOv], positions=[0, 1], widths=0.6,
                               showmeans=False, showmedians=False, showextrema=False)
        parts["bodies"][0].set_facecolor(VIOLIN_WT_COLOR); parts["bodies"][0].set_edgecolor("none")
        parts["bodies"][1].set_facecolor(VIOLIN_KO_COLOR); parts["bodies"][1].set_edgecolor("none")

    # per-sample colors (legend suppressed)
    wt_keys = list(WT_dict.keys()); ko_keys = list(KO_dict.keys())
    wt_col_map, ko_col_map = _group_sample_colors(wt_keys, ko_keys)

    # WT at x=0
    for s, arr in WT_dict.items():
        if arr.size == 0: continue
        arrv = _mask_for_ylim(arr, ylim) if (ylim is not None and TRIM_FOR_PLOTTING_ONLY) else arr
        if arrv.size == 0: continue
        xs = 0 + rng.normal(0, JITTER_SCALE, size=arrv.size)
        ax.scatter(xs, arrv, s=DOT_SIZE, alpha=0.6, c=[wt_col_map.get(s)], edgecolors="none", rasterized=True)

    # KO at x=1
    for s, arr in KO_dict.items():
        if arr.size == 0: continue
        arrv = _mask_for_ylim(arr, ylim) if (ylim is not None and TRIM_FOR_PLOTTING_ONLY) else arr
        if arrv.size == 0: continue
        xs = 1 + rng.normal(0, JITTER_SCALE, size=arrv.size)
        ax.scatter(xs, arrv, s=DOT_SIZE, alpha=0.6, c=[ko_col_map.get(s)], edgecolors="none", rasterized=True)

    # group medians + whiskers
    def _median_iqr(x, xpos):
        if x.size == 0: return
        med = np.median(x); q1, q3 = np.quantile(x, [0.05, 0.95])
        ax.scatter([xpos], [med], s=42, c="white", edgecolors="black", linewidths=0.8, zorder=3)
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

    if SHOW_SAMPLE_LEGEND:
        handles = []
        for s in wt_keys:
            handles.append(mpatches.Patch(color=wt_col_map[s], label=s))
        for s in ko_keys:
            handles.append(mpatches.Patch(color=ko_col_map[s], label=s))
        if handles:
            ax.legend(handles=handles, title="Samples",
                      bbox_to_anchor=(1.02, 1), loc="upper left",
                      frameon=False, ncol=1, fontsize=FS["legend"])

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=PUB_DPI)
    if SAVE_PDF_ALSO:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".pdf", bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)

# ------------- NEW: separate overlay violin figure (WT over KO) -------------
def _overlay_violin(WT_dict, KO_dict, ylabel, title, out_png, log_y=False, ylim=None, stats_text=None):
    """
    Draw KO violin first (background), then WT violin on top at same x positions.
    No scatter dots here—just clean overlay to compare distributions.
    """
    _apply_pub_style()

    WT_all = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    if WT_all.size == 0 and KO_all.size == 0:
        logger.warning("Nothing to overlay for %s", title)
        return

    if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
        WTv = _mask_for_ylim(WT_all, ylim)
        KOv = _mask_for_ylim(KO_all, ylim)
    else:
        WTv, KOv = WT_all, KO_all

    if WTv.size == 0 and KOv.size == 0:
        logger.warning("All overlay values fell outside ylim for %s; skipping.", title)
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    # KO background violin at x=0 and x=1? -> we want the same 2 groups as original
    # We'll overlay both groups: WT over KO at each position (0 for WT, 1 for KO)
    # For more direct contrast, many users prefer overlay at each group. Here:
    # x=0: KO then WT, x=1: KO then WT (still informative of overall spread).
    # But you asked "overlay WT on KO": draw KO first, then WT, at both positions.

    # position 0: WT vs KO pooled around WT? We'll keep conventional labels.
    # Draw KO then WT at positions [0,1].
    # KO (background)
    if KOv.size > 0:
        parts_ko = ax.violinplot([KOv, KOv], positions=[0, 1], widths=0.7,
                                 showmeans=False, showmedians=False, showextrema=False)
        for b in parts_ko["bodies"]:
            b.set_facecolor(VIOLIN_KO_COLOR); b.set_edgecolor("none"); b.set_zorder(1)

    # WT (foreground)
    if WTv.size > 0:
        parts_wt = ax.violinplot([WTv, WTv], positions=[0, 1], widths=0.7,
                                 showmeans=False, showmedians=False, showextrema=False)
        for b in parts_wt["bodies"]:
            b.set_facecolor(VIOLIN_WT_COLOR); b.set_edgecolor("none"); b.set_zorder(2)

    # Add medians to aid reading (using pooled subsampled values already)
    def _median_iqr(x, xpos, z=3):
        if x.size == 0: return
        med = np.median(x); q1, q3 = np.quantile(x, [0.25, 0.75])
        ax.scatter([xpos], [med], s=46, c="white", edgecolors="black", linewidths=0.9, zorder=z)
        ax.vlines(xpos, q1, q3, colors="black", linewidth=1.2, zorder=z)

    _median_iqr(KOv, 0, z=2.5); _median_iqr(WTv, 0, z=3.0)
    _median_iqr(KOv, 1, z=2.5); _median_iqr(WTv, 1, z=3.0)

    ax.set_xticks([0, 1], ["WT", "KO"])
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title + " (Overlay violins: WT on KO)")
    ax.yaxis.grid(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    _annotate(ax, stats_text)

    # mini legend
    handles = [mpatches.Patch(color=VIOLIN_KO_COLOR, label="KO (background)"),
               mpatches.Patch(color=VIOLIN_WT_COLOR, label="WT (foreground)")]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=FS["legend"])

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=PUB_DPI)
    if SAVE_PDF_ALSO:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".pdf", bbox_inches="tight", dpi=PUB_DPI)
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

    L = min(len(ch2), len(ch3), len(amideI), len(amideII))
    if L == 0: return None, None
    ch2 = ch2[:L]; ch3 = ch3[:L]; amideI = amideI[:L]; amideII = amideII[:L]

    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        r1 = ch2 / ch3
        r2 = amideI / amideII
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

def _gather_by_group_for_stats(sample_names, groups):
    WT_vals = {"CH2/CH3": [], "AmideI/AmideII": []}
    KO_vals = {"CH2/CH3": [], "AmideI/AmideII": []}
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    with LogTimer("Collect per-pixel ratios for WT/KO (stats)"):
        for n in sample_names:
            g = sample_to_group.get(n)
            if g is None: continue
            r1, r2 = _collect_ratios_for_sample(n)
            if r1 is None: continue
            (WT_vals if g=="WT" else KO_vals)["CH2/CH3"].append(r1)
            (WT_vals if g=="WT" else KO_vals)["AmideI/AmideII"].append(r2)

    WT_ch2ch3 = _concat(WT_vals["CH2/CH3"]); WT_ai_aii = _concat(WT_vals["AmideI/AmideII"])
    KO_ch2ch3 = _concat(KO_vals["CH2/CH3"]); KO_ai_aii = _concat(KO_vals["AmideI/AmideII"])

    WT_ch2ch3 = _cap_group(WT_ch2ch3, "CH2/CH3",         seed=1)
    WT_ai_aii = _cap_group(WT_ai_aii, "AmideI/AmideII",  seed=2)
    KO_ch2ch3 = _cap_group(KO_ch2ch3, "CH2/CH3",         seed=3)
    KO_ai_aii = _cap_group(KO_ai_aii, "AmideI/AmideII",  seed=4)
    return (WT_ch2ch3, KO_ch2ch3), (WT_ai_aii, KO_ai_aii)

def _gather_by_group_ext_for_stats(sample_names, groups, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH):
    WT_vals = {"CH2/CH3": [], "AmideI/AmideII": [], "1734/AmideI": [], "1734/AmideII": [], "CH2/AmideI": [], "CH3/AmideI": []}
    KO_vals = {"CH2/CH3": [], "AmideI/AmideII": [], "1734/AmideI": [], "1734/AmideII": [], "CH2/AmideI": [], "CH3/AmideI": []}
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    with LogTimer("Collect per-pixel extended ratios incl. 1734 (stats)"):
        for n in sample_names:
            g = sample_to_group.get(n)
            if g is None: continue
            r1, r2, r3, r4 = _collect_ratios_for_sample_ext(n, center_1734=center_1734, halfwidth_1734=halfwidth_1734)
            if r1 is None: continue
            tgt = WT_vals if g == "WT" else KO_vals
            tgt["CH2/CH3"].append(r1);   tgt["AmideI/AmideII"].append(r2)
            tgt["1734/AmideI"].append(r3);  tgt["1734/AmideII"].append(r4)

            per = _collect_per_pixel_values(n, {
                "ch2": (BAND_CH2, window), "ch3": (BAND_CH3, window), "amideI": (BAND_AMIDE_I, window)
            })
            if per is not None and len(per["amideI"]) and len(per["ch2"]) and len(per["ch3"]):
                L = min(len(per["amideI"]), len(per["ch2"]), len(per["ch3"]))
                ch2 = per["ch2"][:L]; ch3 = per["ch3"][:L]; ai = per["amideI"][:L]
                ch2_ai = ch2 / ai; ch3_ai = ch3 / ai
                ch2_ai = ch2_ai[np.isfinite(ch2_ai) & (ch2_ai>0)]
                ch3_ai = ch3_ai[np.isfinite(ch3_ai) & (ch3_ai>0)]
                tgt["CH2/AmideI"].append(ch2_ai)
                tgt["CH3/AmideI"].append(ch3_ai)

    def _pack(key, s1, s2, capkey):
        wt = _concat(WT_vals[key]); ko = _concat(KO_vals[key])
        wt = _cap_group(wt, capkey, s1); ko = _cap_group(ko, capkey, s2)
        return wt, ko

    out = {
        "CH2/CH3":        _pack("CH2/CH3",        301, 302, "CH2/CH3"),
        "AmideI/AmideII": _pack("AmideI/AmideII", 401, 402, "AmideI/AmideII"),
        "1734/AmideI":    _pack("1734/AmideI",    501, 502, "1734/AmideI"),
        "1734/AmideII":   _pack("1734/AmideII",   601, 602, "1734/AmideII"),
        "CH2/AmideI":     _pack("CH2/AmideI",     701, 702, "CH2/AmideI"),
        "CH3/AmideI":     _pack("CH3/AmideI",     801, 802, "CH3/AmideI"),
    }
    return out

# ---- Gather for PLOTS (per-sample) ----
def _gather_per_sample_basic(sample_names, groups):
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
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})
    WT_1734_ai, KO_1734_ai = {}, {}
    WT_1734_aii, KO_1734_aii = {}, {}
    WT_ch2_ai, KO_ch2_ai = {}, {}
    WT_ch3_ai, KO_ch3_ai = {}, {}

    with LogTimer("Collect per-sample extended ratios for plotting"):
        for n in sample_names:
            r1, r2, r3, r4 = _collect_ratios_for_sample_ext(n, center_1734=center_1734, halfwidth_1734=halfwidth_1734)
            if r1 is None: continue
            per = _collect_per_pixel_values(n, {
                "ch2": (BAND_CH2, window), "ch3": (BAND_CH3, window), "amideI": (BAND_AMIDE_I, window)
            })
            if per is None: continue
            L = min(len(per["ch2"]), len(per["ch3"]), len(per["amideI"]))

            ch2_ai = (per["ch2"][:L] / per["amideI"][:L]); ch2_ai = ch2_ai[np.isfinite(ch2_ai) & (ch2_ai>0)]
            ch3_ai = (per["ch3"][:L] / per["amideI"][:L]); ch3_ai = ch3_ai[np.isfinite(ch3_ai) & (ch3_ai>0)]

            if sample_to_group.get(n) == "WT":
                WT_1734_ai[n]  = r3; WT_1734_aii[n] = r4
                WT_ch2_ai[n]   = ch2_ai; WT_ch3_ai[n] = ch3_ai
            else:
                KO_1734_ai[n]  = r3; KO_1734_aii[n] = r4
                KO_ch2_ai[n]   = ch2_ai; KO_ch3_ai[n] = ch3_ai
    return (WT_1734_ai, KO_1734_ai), (WT_1734_aii, KO_1734_aii), (WT_ch2_ai, KO_ch2_ai), (WT_ch3_ai, KO_ch3_ai)

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
        plt.savefig(path, dpi=PUB_DPI, bbox_inches="tight")
        if SAVE_PDF_ALSO:
            plt.savefig(out_prefix + "_centroid_margins.pdf", bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", path)
        return angles

# ========================== MAIN RUN ==================================
if __name__ == "__main__":
    logger.info("Preprocess: shift floor=%.3g; normalization=%s",
                SHIFT_MIN_FLOOR, ("NONE" if NORM_MODE.lower()=="none" else NORM_MODE))
    stats_rows = []

    # ---------- Base ratios (CH2/CH3 & AmideI/II) ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        (ch2ch3_WT_s, ch2ch3_KO_s), (ai_aii_WT_s, ai_aii_KO_s) = _gather_by_group_for_stats(sample_names, groups)

        row_ch = cliffs_only_stats("CH2/CH3", ch2ch3_WT_s, ch2ch3_KO_s); stats_rows.append(row_ch)
        row_ai = cliffs_only_stats("AmideI/AmideII", ai_aii_WT_s, ai_aii_KO_s); stats_rows.append(row_ai)

        (WT_ch_dict, KO_ch_dict), (WT_ai_dict, KO_ai_dict) = _gather_per_sample_basic(sample_names, groups)

        out1 = os.path.join(save_dir, "ratios_CH2_CH3_strip_by_sample_hues.png")
        out2 = os.path.join(save_dir, "ratios_AmideI_AmideII_strip_by_sample_hues.png")

        ylim_ch2ch3 = MANUAL_YLIMS["CH2/CH3"] if USE_MANUAL_YLIMS else _auto_ylim(ch2ch3_WT_s, ch2ch3_KO_s, log=False)
        ylim_ai_aii = MANUAL_YLIMS["AmideI/AmideII"] if USE_MANUAL_YLIMS else _auto_ylim(ai_aii_WT_s, ai_aii_KO_s, log=USE_LOG_Y_FOR["AmideI/AmideII"])

        # Original plots (unchanged structure)
        _strip_plot_by_sample(
            WT_ch_dict, KO_ch_dict,
            ylabel="CH$_2$/CH$_3$ ratio (per pixel; top-k in window; shift-only)",
            title="CH$_2$/CH$_3$ Ratio – WT vs KO (distinct green/red hues per sample)",
            out_png=out1, log_y=False, ylim=ylim_ch2ch3,
            stats_text=format_stats_for_title(row_ch)
        )
        _strip_plot_by_sample(
            WT_ai_dict, KO_ai_dict,
            ylabel="Amide I / Amide II ratio (per pixel; top-k in window; shift-only)",
            title="Amide I / Amide II – WT vs KO (distinct green/red hues per sample)",
            out_png=out2, log_y=USE_LOG_Y_FOR["AmideI/AmideII"], ylim=ylim_ai_aii,
            stats_text=format_stats_for_title(row_ai)
        )

        # NEW: separate overlay violins (WT over KO)
        _overlay_violin(
            WT_ch_dict, KO_ch_dict,
            ylabel="CH$_2$/CH$_3$ ratio (per pixel; top-k in window; shift-only)",
            title="CH$_2$/CH$_3$ Ratio – WT vs KO",
            out_png=os.path.join(save_dir, "ratios_CH2_CH3_overlayviolin.png"),
            log_y=False, ylim=ylim_ch2ch3, stats_text=format_stats_for_title(row_ch)
        )
        _overlay_violin(
            WT_ai_dict, KO_ai_dict,
            ylabel="Amide I / Amide II ratio (per pixel; top-k in window; shift-only)",
            title="Amide I / Amide II – WT vs KO",
            out_png=os.path.join(save_dir, "ratios_AmideI_AmideII_overlayviolin.png"),
            log_y=USE_LOG_Y_FOR["AmideI/AmideII"], ylim=ylim_ai_aii, stats_text=format_stats_for_title(row_ai)
        )

        logger.info("Saved ratio plots and overlay violins.")
    else:
        logger.warning("Skipping base ratios (missing config).")

    # ---------- Extended lipid↔protein ratios ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        ext_stats = _gather_by_group_ext_for_stats(sample_names, groups, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH)
        r_1734_ai_WT_s,  r_1734_ai_KO_s  = ext_stats["1734/AmideI"]
        r_1734_aii_WT_s, r_1734_aii_KO_s = ext_stats["1734/AmideII"]
        ch2_ai_WT_s,     ch2_ai_KO_s     = ext_stats["CH2/AmideI"]
        ch3_ai_WT_s,     ch3_ai_KO_s     = ext_stats["CH3/AmideI"]

        rows_ext = [
            ("1734/AmideI",  r_1734_ai_WT_s,  r_1734_ai_KO_s),
            ("1734/AmideII", r_1734_aii_WT_s, r_1734_aii_KO_s),
            ("CH2/AmideI",   ch2_ai_WT_s,     ch2_ai_KO_s),
            ("CH3/AmideI",   ch3_ai_WT_s,     ch3_ai_KO_s),
        ]
        for name, WT_s, KO_s in rows_ext:
            stats_rows.append(cliffs_only_stats(name, WT_s, KO_s))

        (WT_1734_ai, KO_1734_ai), (WT_1734_aii, KO_1734_aii), (WT_ch2_ai, KO_ch2_ai), (WT_ch3_ai, KO_ch3_ai) = \
            _gather_per_sample_ext(sample_names, groups, center_1734=BAND_1734_CENTER, halfwidth_1734=BAND_1734_HALFWIDTH)

        def _ylim_for(key, WT_s, KO_s):
            return MANUAL_YLIMS[key] if USE_MANUAL_YLIMS else _auto_ylim(WT_s, KO_s, log=False)

        out_1734_ai  = os.path.join(save_dir, "ratios_1734_over_AmideI_by_sample_hues.png")
        out_1734_aii = os.path.join(save_dir, "ratios_1734_over_AmideII_by_sample_hues.png")
        out_ch2_ai   = os.path.join(save_dir, "ratios_CH2_over_AmideI_by_sample_hues.png")
        out_ch3_ai   = os.path.join(save_dir, "ratios_CH3_over_AmideI_by_sample_hues.png")

        _strip_plot_by_sample(
            WT_1734_ai, KO_1734_ai,
            ylabel="1734 / Amide I (per pixel; top-k; shift-only)",
            title=f"1734 / Amide I – WT vs KO @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=out_1734_ai, log_y=False, ylim=_ylim_for("1734/AmideI", r_1734_ai_WT_s, r_1734_ai_KO_s),
            stats_text=format_stats_for_title(stats_rows[-4])
        )
        _strip_plot_by_sample(
            WT_1734_aii, KO_1734_aii,
            ylabel="1734 / Amide II (per pixel; top-k; shift-only)",
            title=f"1734 / Amide II – WT vs KO @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=out_1734_aii, log_y=False, ylim=_ylim_for("1734/AmideII", r_1734_aii_WT_s, r_1734_aii_KO_s),
            stats_text=format_stats_for_title(stats_rows[-3])
        )
        _strip_plot_by_sample(
            WT_ch2_ai, KO_ch2_ai,
            ylabel="CH$_2$ / Amide I (per pixel; top-k; shift-only)",
            title="CH$_2$ / Amide I – WT vs KO",
            out_png=out_ch2_ai, log_y=False, ylim=_ylim_for("CH2/AmideI", ch2_ai_WT_s, ch2_ai_KO_s),
            stats_text=format_stats_for_title(stats_rows[-2])
        )
        _strip_plot_by_sample(
            WT_ch3_ai, KO_ch3_ai,
            ylabel="CH$_3$ / Amide I (per pixel; top-k; shift-only)",
            title="CH$_3$ / Amide I – WT vs KO",
            out_png=out_ch3_ai, log_y=False, ylim=_ylim_for("CH3/AmideI", ch3_ai_WT_s, ch3_ai_KO_s),
            stats_text=format_stats_for_title(stats_rows[-1])
        )

        # Overlay versions (separate figures)
        _overlay_violin(
            WT_1734_ai, KO_1734_ai,
            ylabel="1734 / Amide I (per pixel; top-k; shift-only)",
            title=f"1734 / Amide I @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=os.path.join(save_dir, "ratios_1734_over_AmideI_overlayviolin.png"),
            log_y=False, ylim=_ylim_for("1734/AmideI", r_1734_ai_WT_s, r_1734_ai_KO_s),
            stats_text=format_stats_for_title(stats_rows[-4])
        )
        _overlay_violin(
            WT_1734_aii, KO_1734_aii,
            ylabel="1734 / Amide II (per pixel; top-k; shift-only)",
            title=f"1734 / Amide II @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=os.path.join(save_dir, "ratios_1734_over_AmideII_overlayviolin.png"),
            log_y=False, ylim=_ylim_for("1734/AmideII", r_1734_aii_WT_s, r_1734_aii_KO_s),
            stats_text=format_stats_for_title(stats_rows[-3])
        )
        _overlay_violin(
            WT_ch2_ai, KO_ch2_ai,
            ylabel="CH$_2$ / Amide I (per pixel; top-k; shift-only)",
            title="CH$_2$ / Amide I",
            out_png=os.path.join(save_dir, "ratios_CH2_over_AmideI_overlayviolin.png"),
            log_y=False, ylim=_ylim_for("CH2/AmideI", ch2_ai_WT_s, ch2_ai_KO_s),
            stats_text=format_stats_for_title(stats_rows[-2])
        )
        _overlay_violin(
            WT_ch3_ai, KO_ch3_ai,
            ylabel="CH$_3$ / Amide I (per pixel; top-k; shift-only)",
            title="CH$_3$ / Amide I",
            out_png=os.path.join(save_dir, "ratios_CH3_over_AmideI_overlayviolin.png"),
            log_y=False, ylim=_ylim_for("CH3/AmideI", ch3_ai_WT_s, ch3_ai_KO_s),
            stats_text=format_stats_for_title(stats_rows[-1])
        )

        logger.info("Saved extended ratio plots and overlay violins.")
    else:
        logger.warning("Skipping extended ratios (missing config).")

    # ---------- SINGLE-BAND intensity: lipid 1734 + Amide I + Amide II ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        center_band = BAND_1734_CENTER
        halfwidth   = BAND_1734_HALFWIDTH

        WT_band_s = []; KO_band_s = []
        for n in groups["WT"]:
            v = _collect_band_for_sample(n, center_band, halfwidth=halfwidth)
            if v is not None: WT_band_s.append(v)
        for n in groups["KO"]:
            v = _collect_band_for_sample(n, center_band, halfwidth=halfwidth)
            if v is not None: KO_band_s.append(v)
        WT_band_s = _concat(WT_band_s); KO_band_s = _concat(KO_band_s)
        WT_band_s = _cap_group(WT_band_s, "1734", seed=11)
        KO_band_s = _cap_group(KO_band_s, "1734", seed=12)

        row_1734 = cliffs_only_stats(f"1734@{center_band:.1f}", WT_band_s, KO_band_s); stats_rows.append(row_1734)

        WT_band_dict, KO_band_dict = _gather_per_sample_band(sample_names, groups, center_cm1=center_band, halfwidth=halfwidth)

        ylim_band = MANUAL_YLIMS["1734"] if USE_MANUAL_YLIMS else _auto_ylim(WT_band_s, KO_band_s, log=False)
        out_band = os.path.join(save_dir, f"band_1734cm-1_by_sample_hues.png")
        _strip_plot_by_sample(
            WT_band_dict, KO_band_dict,
            ylabel=f"Absorbance @ {center_band:.1f} cm$^{{-1}}$ (top-k; shift-only)",
            title=f"Lipid ester C=O — WT vs KO @ {center_band:.1f} cm$^{{-1}}$",
            out_png=out_band, log_y=False, ylim=ylim_band,
            stats_text=format_stats_for_title(row_1734)
        )
        # overlay
        _overlay_violin(
            WT_band_dict, KO_band_dict,
            ylabel=f"Absorbance @ {center_band:.1f} cm$^{{-1}}$ (top-k; shift-only)",
            title=f"Lipid ester C=O @ {center_band:.1f} cm$^{{-1}}$",
            out_png=os.path.join(save_dir, "band_1734_overlayviolin.png"),
            log_y=False, ylim=ylim_band, stats_text=format_stats_for_title(row_1734)
        )

        for label, center in [("amideI", BAND_AMIDE_I), ("amideII", BAND_AMIDE_II)]:
            WT_s = []; KO_s = []
            for n in groups["WT"]:
                v = _collect_band_for_sample(n, center, halfwidth=window)
                if v is not None: WT_s.append(v)
            for n in groups["KO"]:
                v = _collect_band_for_sample(n, center, halfwidth=window)
                if v is not None: KO_s.append(v)
            WT_s = _concat(WT_s); KO_s = _concat(KO_s)
            WT_s = _cap_group(WT_s, label, seed=21)
            KO_s = _cap_group(KO_s, label, seed=22)

            row = cliffs_only_stats(f"{label}@{center:.1f}", WT_s, KO_s); stats_rows.append(row)

            WT_d, KO_d = _gather_per_sample_band(sample_names, groups, center_cm1=center, halfwidth=window)
            ylim_sb = MANUAL_YLIMS[label] if USE_MANUAL_YLIMS else _auto_ylim(WT_s, KO_s, log=USE_LOG_Y_FOR[label])
            out_sb = os.path.join(save_dir, f"band_{label}_{int(center)}cm-1_by_sample_hues.png")
            _strip_plot_by_sample(
                WT_d, KO_d,
                ylabel=f"{label.replace('amide','Amide ')} intensity (top-k; shift-only)",
                title=f"{label.replace('amide','Amide ').title()} — WT vs KO @ {center:.1f} cm$^{{-1}}$",
                out_png=out_sb, log_y=USE_LOG_Y_FOR[label], ylim=ylim_sb,
                stats_text=format_stats_for_title(row)
            )
            # overlay
            _overlay_violin(
                WT_d, KO_d,
                ylabel=f"{label.replace('amide','Amide ')} intensity (top-k; shift-only)",
                title=f"{label.replace('amide','Amide ').title()} @ {center:.1f} cm$^{{-1}}$",
                out_png=os.path.join(save_dir, f"band_{label}_overlayviolin.png"),
                log_y=USE_LOG_Y_FOR[label], ylim=ylim_sb, stats_text=format_stats_for_title(row)
            )
        logger.info("Saved single-band plots and overlay violins.")
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
