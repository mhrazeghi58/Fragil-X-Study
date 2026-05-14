# ======================================================================
# WT vs KO spectral analysis + overlays
#
# ANALYSIS:
#   - Uses ONLY up to MAX_POINTS_PER_SAMPLE (=1500) pixels per sample
#     for BOTH bands and ratios (stats are computed on these capped values).
#
# PLOTTING (UPDATED TO MATCH YOUR 2nd SCRIPT SCATTER STYLE):
#   - Uses per-sample subsampling with:
#        k = max(MIN_PLOT_POINTS, round(n * PLOT_SUBSAMPLE_FRAC))
#   - This replaces the old "PLOT_MAX_POINTS_PER_SAMPLE" logic.
#   - Scatter points, edge alpha, median square, black IQR bar match 2nd script.
#   - Overlay violin remains CLEAN (no labels/ticks/grid/δ)
#
# FIXES / UPDATES:
#   - np.trapz keepdims removed (compat fix)
#   - Pixel-aligned ratio computation (same pixels for numerator & denominator)
#   - Optional intensity/thickness normalization (ref area) + optional detrend
#   - Adds PO2 ratios vs carbs and lipids
#   - Keeps white + red/blue ratio overlays vs WT thresholds
# ======================================================================

import os, logging, time
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from scipy import stats
import colorsys, matplotlib.colors as mcolors

# ============================== PATHS / CONFIG ==============================
cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R\UMAP_clustering_test"
cube_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"
save_dir    = os.path.join(cube_dir, "WT_KO_SPEC")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "WT_KO")

sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

# ---------------- Cluster selection ----------------
selected_clusters = {
    "T1":[0,1,2,3,4,5,6,7],
    "T2":[0,1,2,3,4,5,6,7],
    "T3":[0,1,2,3,4,5,6,7],
    "T6":[0,1,2,3,4,5,6,7],
    "T17":[0,1,2,3,4,5,6,7],
    "T19":[0,1,2,3,4.5,6,7],   # 4.5 will be dropped by cleaner
    "T20":[0,1,2,3,4,5,6,7],
    "T21":[0,1,2,3,4,5,6,7],
    "T10":[0,1,2,3,4,5,6,7],
    "T11":[0,1,2,3,4,5,6,7],
    "T12":[0,1,2,3,4,5,6,7],
    "T13":[0,1,2,3,4,5,6,7],
    "T14":[0,1,2,3,4,5,6,7],
    "T15":[0,1,2,3,4,5,6,7],
    "T22":[0,1,2,3,4,5,6,7],
}

# --------- Spectral / band settings ---------
try:
    window
except NameError:
    window = 30

TOPK_PER_BAND = 1  # mean of top-k values inside band window per pixel
EPS = 1e-12

SHIFT_MIN_FLOOR_BANDS  = 0.2
SHIFT_MIN_FLOOR_RATIOS = 0.5

USE_L2_NORM_BANDS  = True
USE_L2_NORM_RATIOS = True

# ===================== THICKNESS / INTENSITY OPTIONS =====================
NORM_MODE_BANDS  = None         # None | "amideI" | "amideII" | "amideI+II"
NORM_MODE_RATIOS = None         # None | "amideI" | "amideII" | "amideI+II"

APPLY_POLY_DETREND_BANDS  = False
APPLY_POLY_DETREND_RATIOS = False
POLY_DEGREE = 2

# ===================== ANALYSIS CAPS =====================
MAX_POINTS_PER_SAMPLE = 1500   # analysis/stats use up to this many pixels per sample
SEED = 1
rng_global = np.random.default_rng(SEED)

# ===================== PLOT-ONLY SUBSAMPLING (UPDATED) =====================
# MATCH 2nd SCRIPT: k = max(MIN_PLOT_POINTS, round(n * PLOT_SUBSAMPLE_FRAC))
PLOT_SUBSAMPLE_FRAC = 0.004
MIN_PLOT_POINTS     = 200
PLOT_MAX_POINTS_PER_GROUP  = 7000   # pooled violin cap (visual only)

# ===================== COMBINED FIGURE LAYOUT =====================
COMBINED_LAYOUT = "side"             # "side" (left-right) or "stack" (top-bottom)

# ===================== MATCH 2nd SCRIPT SCATTER STYLE =====================
BASE_WT = "#2ca02c"
BASE_KO = "#d62728"

DOT_SIZE = 22
DOT_ALPHA = 0.45
DOT_EDGE_ALPHA = 0.55
DOT_EDGE_LW = 0.7
JITTER_SCALE = 0.06

MEDIAN_MARKER = "s"      # square
MEDIAN_SIZE = 130
MEDIAN_FACE = "white"
MEDIAN_EDGE_W = 1.6

IQR_LINE_W = 2.0
IQR_CAP_LW = 2.0
IQR_COLOR  = "black"

SAVE_PDF_ALSO = True

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11

FS = {
    "base":   int(BASE_FONTSIZE),
    "axes":   int(BASE_FONTSIZE + 1),
    "title":  int(BASE_FONTSIZE + 3),
    "tick":   int(BASE_FONTSIZE),
    "annot":  10,
}

VIOLIN_WT_COLOR = (0.3, 0.7, 0.3, 0.22)
VIOLIN_KO_COLOR = (0.9, 0.3, 0.3, 0.22)

TRIM_FOR_PLOTTING_ONLY = True
APPLY_RATIO_HARD_CAP = True
RATIO_HARD_CAP       = 10.0

ANNOTATE_STATS_ON_PLOTS = True

DENOM_MIN_PCT        = 3.0
DENOM_ABS_FLOOR      = 1e-2
NUM_MIN_PCT          = 15.0
NUM_ABS_FLOOR        = 1e-4

# ===================== BAND CENTERS (cm-1) =====================
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

# ===================== YLIMS / LOG =======================
USE_MANUAL_YLIMS = True
MANUAL_YLIMS = {
    "CH2/CH3":           None,
    "AmideI/AmideII":    None,
    "single_band":       None,
}
USE_LOG_Y_FOR = {
    "CH2/CH3":        False,
    "AmideI/AmideII": False,
    "single_band":    False,
}

# ===================== OVERLAY CONFIG =====================
WT_REF_STAT = "median"   # or "percentile"
WT_REF_PCTL = 50.0
OVERLAY_ALPHA = 0.75
OVERLAY_COLOR = (0.85, 0.2, 0.2)        # above WT thr
OVERLAY_COLOR_BELOW = (0.2, 0.2, 0.85)  # below WT thr

# ========================== LOGGING ==========================
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

# ========================== BAND LIBRARY ======================
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

BAND_LABELS = {
    "1734":      "Abs @ 1734 cm$^{-1}$",
    "amideI":    "Amide I intensity",
    "amideII":   "Amide II intensity",
    "ch2":       "CH$_2$ intensity (~1464 cm$^{-1}$)",
    "ch3":       "CH$_3$ intensity (~1375 cm$^{-1}$)",
    "po2_1080":  "Phosphate ~1080 cm$^{-1}$ intensity",
    "po2_1235":  "Phosphate ~1235 cm$^{-1}$ intensity",
    "carb_1030": "Carbohydrate ~1030 cm$^{-1}$ intensity",
    "carb_1155": "Carbohydrate ~1155 cm$^{-1}$ intensity",
}

# ========================== STYLE APPLIER =============================
def _apply_pub_style():
    fam = FONT_FAMILY or "DejaVu Sans"
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": PUB_DPI,
        "font.family": fam,
        "font.size":   FS["base"],
        "axes.labelsize": FS["axes"],
        "axes.titlesize": FS["title"],
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "axes.linewidth": 0.9,
        "grid.color": "#9aa0a6",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "mathtext.default": "regular",
    })

# ========================== IO HELPERS ================================
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

# ===================== PREPROCESS ======================
def _shift_spectra_min_floor(cube, floor):
    C = cube.astype(np.float64, copy=True)
    mins = np.min(C, axis=2, keepdims=True)
    add = np.clip(floor - mins, 0.0, None)
    return C + add

def _l2_normalize_cube(C):
    norms = np.sqrt(np.sum(C*C, axis=2, keepdims=True)) + EPS
    return C / norms

def _poly_detrend_spectra(C, wns, deg=2):
    H, W, Z = C.shape
    x = wns.astype(np.float64)
    x = (x - x.mean()) / (x.std() + EPS)
    V = np.vander(x, deg + 1, increasing=True)
    Y = C.reshape(-1, Z).T
    coef, *_ = np.linalg.lstsq(V, Y, rcond=None)
    baseline = (V @ coef).T.reshape(H, W, Z)
    return C - baseline

def _ref_area(C, wns, center, halfwidth):
    m = (wns >= (center - halfwidth)) & (wns <= (center + halfwidth))
    if not np.any(m):
        return np.ones(C.shape[:2] + (1,), dtype=np.float64)
    a = np.trapz(C[:, :, m], wns[m], axis=2)
    a = np.maximum(a, EPS)
    return a[..., None]

def _normalize_cube(C, wns, norm_mode):
    if norm_mode is None:
        return C
    if norm_mode == "amideI":
        return C / _ref_area(C, wns, BAND_AMIDE_I, window)
    if norm_mode == "amideII":
        return C / _ref_area(C, wns, BAND_AMIDE_II, window)
    if norm_mode == "amideI+II":
        return C / (_ref_area(C, wns, BAND_AMIDE_I, window) + _ref_area(C, wns, BAND_AMIDE_II, window) + EPS)
    logger.warning("Unknown norm_mode=%s; skipping normalization", norm_mode)
    return C

def _preprocess_cube(cube, wns, *, floor: float, l2: bool, detrend: bool, norm_mode):
    C = _shift_spectra_min_floor(cube, floor=floor) if floor is not None else cube.astype(np.float64, copy=True)
    if detrend:
        C = _poly_detrend_spectra(C, wns, deg=POLY_DEGREE)
        if floor is not None:
            C = _shift_spectra_min_floor(C, floor=floor)
    if l2:
        C = _l2_normalize_cube(C)
    C = _normalize_cube(C, wns, norm_mode)
    return C

# ===================== BAND EXTRACTOR ======================
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
    flat = slab.reshape(-1, nwin)
    idx = np.argpartition(flat, -kk, axis=1)[:, -kk:]
    topk = np.take_along_axis(flat, idx, axis=1)
    out_flat = np.nanmean(topk, axis=1)
    return out_flat.reshape(slab.shape[0], slab.shape[1])

_band_value = _band_value_topk_mean

# ========================== UTILS ================================
AUTO_PAD_FRAC = 0.1

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
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=FS["annot"], color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.25")
    )

def _safe_name(s: str) -> str:
    out = str(s).replace("/", "_over_").replace("\\", "_")
    for b in ['"', "'", ":", "*", "?", "<", ">", "|"]:
        out = out.replace(b, "")
    return out.replace(" ", "")

def _concat(lst): return np.concatenate(lst) if lst else np.array([])

# -------- Plot-only subsampling (UPDATED to match 2nd script) --------
def _subsample_for_plot(arr, rng, frac=PLOT_SUBSAMPLE_FRAC, min_pts=MIN_PLOT_POINTS):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return arr
    k = int(max(min_pts, round(n * frac)))
    k = min(k, n)
    if k == n:
        return arr
    idx = rng.choice(n, size=k, replace=False)
    return arr[idx]

# ========================== COLOR MAPS ==========================
def _group_sample_colors(wt_keys, ko_keys):
    def ramp(hex_color, n, v_min=0.45, v_max=0.95, s=0.85):
        r,g,b = mcolors.to_rgb(hex_color)
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        vs = np.linspace(v_min, v_max, max(n,1))
        return [colorsys.hsv_to_rgb(h, s, v) for v in vs]
    wt_cols = ramp(BASE_WT, len(wt_keys))
    ko_cols = ramp(BASE_KO, len(ko_keys))
    return {k: wt_cols[i] for i,k in enumerate(wt_keys)}, {k: ko_cols[i] for i,k in enumerate(ko_keys)}

# ===================== SCATTER SUMMARY DRAWER =====================
def _draw_median_and_iqr(ax, xvals, xpos, edgecolor):
    xvals = np.asarray(xvals)
    xvals = xvals[np.isfinite(xvals)]
    if xvals.size == 0:
        return
    med = float(np.median(xvals))
    q1, q3 = np.quantile(xvals, [0.25, 0.75])

    ax.scatter([xpos], [med],
               s=MEDIAN_SIZE,
               marker=MEDIAN_MARKER,
               c=[MEDIAN_FACE],
               edgecolors=[edgecolor],
               linewidths=MEDIAN_EDGE_W,
               zorder=6)

    ax.vlines(xpos, q1, q3, colors=IQR_COLOR, linewidth=IQR_LINE_W, zorder=5)
    ax.hlines([q1, q3], xpos-0.04, xpos+0.04, colors=IQR_COLOR, linewidth=IQR_CAP_LW, zorder=5)

# ===================== CLEAN OVERLAY AXIS =====================
def _clean_overlay_axis(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# ===================== PLOT PANELS =====================
def _draw_scatter_panel(ax, WT_dict, KO_dict, ylabel, log_y=False, ylim=None, stats_text=None):
    rng = np.random.default_rng(0)

    WT_all_full = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all_full = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    wt_keys = list(WT_dict.keys())
    ko_keys = list(KO_dict.keys())
    wt_col_map, ko_col_map = _group_sample_colors(wt_keys, ko_keys)

    # WT
    for s, arr in (WT_dict or {}).items():
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
            arr = _mask_for_ylim(arr, ylim)
        if arr.size == 0:
            continue
        arrp = _subsample_for_plot(arr, rng=rng)
        xs = 0 + rng.normal(0, JITTER_SCALE, size=arrp.size)
        ax.scatter(xs, arrp,
                   s=DOT_SIZE, alpha=DOT_ALPHA,
                   c=[wt_col_map.get(s)],
                   edgecolors=[(0, 0, 0, DOT_EDGE_ALPHA)],
                   linewidths=DOT_EDGE_LW,
                   rasterized=True, zorder=3)

    # KO
    for s, arr in (KO_dict or {}).items():
        arr = np.asarray(arr)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
            arr = _mask_for_ylim(arr, ylim)
        if arr.size == 0:
            continue
        arrp = _subsample_for_plot(arr, rng=rng)
        xs = 1 + rng.normal(0, JITTER_SCALE, size=arrp.size)
        ax.scatter(xs, arrp,
                   s=DOT_SIZE, alpha=DOT_ALPHA,
                   c=[ko_col_map.get(s)],
                   edgecolors=[(0, 0, 0, DOT_EDGE_ALPHA)],
                   linewidths=DOT_EDGE_LW,
                   rasterized=True, zorder=3)

    _draw_median_and_iqr(ax, WT_all_full, 0, edgecolor=mcolors.to_rgb(BASE_WT))
    _draw_median_and_iqr(ax, KO_all_full, 1, edgecolor=mcolors.to_rgb(BASE_KO))

    ax.set_xticks([0, 1], ["WT", "KO"])
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax.set_ylim(*ylim)

    ax.yaxis.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _annotate(ax, stats_text)

def _draw_overlay_violin_panel(ax, WT_dict, KO_dict, log_y=False, ylim=None):
    # CLEAN overlay violin: no labels/ticks/grid/δ
    WT_all_full = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all_full = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    rng = np.random.default_rng(0)

    WT_plot = WT_all_full[np.isfinite(WT_all_full)]
    KO_plot = KO_all_full[np.isfinite(KO_all_full)]

    # pooled violin cap (visual only)
    if WT_plot.size > PLOT_MAX_POINTS_PER_GROUP:
        WT_plot = rng.choice(WT_plot, size=PLOT_MAX_POINTS_PER_GROUP, replace=False)
    if KO_plot.size > PLOT_MAX_POINTS_PER_GROUP:
        KO_plot = rng.choice(KO_plot, size=PLOT_MAX_POINTS_PER_GROUP, replace=False)

    if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
        WT_plot = _mask_for_ylim(WT_plot, ylim)
        KO_plot = _mask_for_ylim(KO_plot, ylim)

    if KO_plot.size > 0:
        parts_ko = ax.violinplot([KO_plot], positions=[0], widths=[0.95],
                                 showmeans=False, showmedians=False, showextrema=False)
        bko = parts_ko["bodies"][0]
        bko.set_facecolor(VIOLIN_KO_COLOR)
        bko.set_edgecolor("none")
        bko.set_zorder(1)

    if WT_plot.size > 0:
        parts_wt = ax.violinplot([WT_plot], positions=[0], widths=[0.95],
                                 showmeans=False, showmedians=False, showextrema=False)
        bwt = parts_wt["bodies"][0]
        bwt.set_facecolor(VIOLIN_WT_COLOR)
        bwt.set_edgecolor("none")
        bwt.set_zorder(2)

    if log_y:
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_xlim(-1.0, 1.0)
    _clean_overlay_axis(ax)

def _combined_scatter_and_overlay(WT_dict, KO_dict, ylabel, out_png, log_y=False, ylim=None, stats_text=None):
    _apply_pub_style()

    if COMBINED_LAYOUT == "stack":
        fig = plt.figure(figsize=(11.0, 9.0))
        gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.22)
        ax_sc = fig.add_subplot(gs[0, 0])
        ax_ov = fig.add_subplot(gs[1, 0])
    else:
        fig, (ax_sc, ax_ov) = plt.subplots(
            1, 2, figsize=(11.5, 5.2),
            gridspec_kw={"width_ratios": [1.0, 0.55], "wspace": 0.20}
        )

    _draw_scatter_panel(ax_sc, WT_dict, KO_dict, ylabel, log_y=log_y, ylim=ylim, stats_text=stats_text)
    _draw_overlay_violin_panel(ax_ov, WT_dict, KO_dict, log_y=log_y, ylim=ylim)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=PUB_DPI)
    if SAVE_PDF_ALSO:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)

# ========================== STATISTICS =================
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

def bootstrap_ci(stat_fn, x, y, nboot=2000, alpha=0.05, rng=None):
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

def cliffs_only_stats(name, WT, KO):
    WT = np.asarray(WT); KO = np.asarray(KO)
    n_wt, n_ko = len(WT), len(KO)
    if n_wt == 0 or n_ko == 0:
        return {"metric": name, "n_wt": n_wt, "n_ko": n_ko, "cliffs_delta": np.nan,
                "cliffs_delta_lo": np.nan, "cliffs_delta_hi": np.nan,
                "mean_wt": np.nan, "mean_ko": np.nan, "median_wt": np.nan, "median_ko": np.nan}
    delta = cliffs_delta(WT, KO)
    d_lo, d_hi = bootstrap_ci(cliffs_delta, WT, KO, nboot=2000)
    return {"metric": name, "n_wt": n_wt, "n_ko": n_ko,
            "mean_wt": float(np.mean(WT)), "mean_ko": float(np.mean(KO)),
            "median_wt": float(np.median(WT)), "median_ko": float(np.median(KO)),
            "cliffs_delta": float(delta), "cliffs_delta_lo": float(d_lo), "cliffs_delta_hi": float(d_hi)}

def save_stats_csv(rows, path):
    headers = ["metric","n_wt","n_ko","mean_wt","mean_ko","median_wt","median_ko",
               "cliffs_delta","cliffs_delta_lo","cliffs_delta_hi"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    logger.info("Saved stats table: %s", path)

def format_stats_for_title(row):
    return f"δ={row['cliffs_delta']:+.2f}"

# ======================= DATA GATHER (analysis cap per sample) =======================
def _collect_per_pixel_values(sample, bands_to_get, *, floor, use_l2, detrend, norm_mode):
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
    cube = _preprocess_cube(cube, wns, floor=floor, l2=use_l2, detrend=detrend, norm_mode=norm_mode)

    per_band = {}
    rng = np.random.default_rng(abs(hash(sample)) % (2**32))

    for key, (center, halfw) in bands_to_get.items():
        band_img = _band_value(cube, wns, float(center), halfw if halfw is not None else window)
        vals = band_img[mask2d].ravel()
        vals = vals[np.isfinite(vals) & (vals != 0)]

        if vals.size > MAX_POINTS_PER_SAMPLE:
            take = rng.choice(vals.size, size=MAX_POINTS_PER_SAMPLE, replace=False)
            vals = vals[take]

        per_band[key] = vals

    return per_band

def _collect_band_for_sample(sample, band_key):
    center, halfw = BAND_LIBRARY[band_key]
    per = _collect_per_pixel_values(
        sample,
        {"band": (center, halfw)},
        floor=SHIFT_MIN_FLOOR_BANDS,
        use_l2=USE_L2_NORM_BANDS,
        detrend=APPLY_POLY_DETREND_BANDS,
        norm_mode=NORM_MODE_BANDS
    )
    if per is None: return None
    return per["band"]

# ======================= RATIOS =======================
def build_all_band_vs_amide_ratios():
    ratios = {}
    for bkey, (bc, bw) in BAND_LIBRARY.items():
        if bkey in ("amideI", "amideII"):
            continue
        ratios[f"{bkey}/AmideI"]  = {"num": (bc, bw), "den": BAND_LIBRARY["amideI"]}
        ratios[f"{bkey}/AmideII"] = {"num": (bc, bw), "den": BAND_LIBRARY["amideII"]}
    return ratios

def build_po2_vs_carb_and_lipid_ratios():
    ratios = {}
    po2_keys   = ["po2_1080", "po2_1235"]
    carb_keys  = ["carb_1030", "carb_1155"]
    lipid_keys = ["ch2", "ch3", "1734"]
    for p in po2_keys:
        for c in carb_keys:
            ratios[f"{p}/{c}"] = {"num": BAND_LIBRARY[p], "den": BAND_LIBRARY[c]}
        for l in lipid_keys:
            ratios[f"{p}/{l}"] = {"num": BAND_LIBRARY[p], "den": BAND_LIBRARY[l]}
    return ratios

BASE_RATIOS = {
    "CH2/CH3":        {"num": BAND_LIBRARY["ch2"],    "den": BAND_LIBRARY["ch3"]},
    "AmideI/AmideII": {"num": BAND_LIBRARY["amideI"], "den": BAND_LIBRARY["amideII"]},
}

ALL_RATIOS = {}
ALL_RATIOS.update(BASE_RATIOS)
ALL_RATIOS.update(build_all_band_vs_amide_ratios())
ALL_RATIOS.update(build_po2_vs_carb_and_lipid_ratios())

def _collect_ratios_for_sample_generic(sample, ratio_cfg_dict):
    out = _load_cluster_labels_indices(sample)
    cube = _load_padded_cube(sample)
    if out is None or cube is None:
        return None

    labels, indices = out
    H, W, Z = cube.shape

    sel = selected_clusters.get(sample, [])
    if not sel:
        return None

    mask_sel = _selected_mask_from_clusters(H, W, labels, indices, sel)
    if not np.any(mask_sel):
        return None

    wns = np.linspace(950, 1800, Z)
    cube = _preprocess_cube(
        cube, wns,
        floor=SHIFT_MIN_FLOOR_RATIOS,
        l2=USE_L2_NORM_RATIOS,
        detrend=APPLY_POLY_DETREND_RATIOS,
        norm_mode=NORM_MODE_RATIOS
    )

    pix = np.flatnonzero(mask_sel.ravel())
    if pix.size == 0:
        return None

    rng = np.random.default_rng(abs(hash(sample)) % (2**32))
    if pix.size > MAX_POINTS_PER_SAMPLE:
        pix = rng.choice(pix, size=MAX_POINTS_PER_SAMPLE, replace=False)
    pix = np.sort(pix)

    band_cache = {}
    def get_band_vec(center, halfw):
        key = (float(center), float(halfw if halfw is not None else window))
        if key in band_cache:
            return band_cache[key]
        img = _band_value(cube, wns, key[0], key[1])
        vec = img.ravel()[pix]
        band_cache[key] = vec
        return vec

    out_ratios = {}
    for rkey, cfg in ratio_cfg_dict.items():
        num_c, num_w = cfg["num"]
        den_c, den_w = cfg["den"]

        num = get_band_vec(num_c, num_w)
        den = get_band_vec(den_c, den_w)

        valid = np.isfinite(num) & np.isfinite(den)
        ok_num, _ = _num_guard(num)
        ok_den, _ = _denom_guard(den)
        valid = valid & ok_num & ok_den

        if not np.any(valid):
            out_ratios[rkey] = np.array([], dtype=float)
            continue

        rr = num[valid] / (den[valid] + EPS)
        if APPLY_RATIO_HARD_CAP:
            rr = np.clip(rr, 0, RATIO_HARD_CAP)
        rr = rr[np.isfinite(rr) & (rr > 0)]
        out_ratios[rkey] = rr

    return out_ratios

# ======================= OVERLAY MAPS (white + red/blue) =======================
def _ratio_map_for_sample(sample, num_center, num_halfwidth, den_center, den_halfwidth):
    out = _load_cluster_labels_indices(sample); cube = _load_padded_cube(sample)
    if out is None or cube is None:
        return None, None, 0, None

    labels, indices = out
    H, W, Z = cube.shape
    wns = np.linspace(950, 1800, Z)
    cube = _preprocess_cube(
        cube, wns,
        floor=SHIFT_MIN_FLOOR_RATIOS,
        l2=USE_L2_NORM_RATIOS,
        detrend=APPLY_POLY_DETREND_RATIOS,
        norm_mode=NORM_MODE_RATIOS
    )

    sel = selected_clusters.get(sample, [])
    if not sel:
        return None, None, 0, None

    mask_sel = _selected_mask_from_clusters(H, W, labels, indices, sel)

    num_img = _band_value(cube, wns, float(num_center), num_halfwidth if num_halfwidth is not None else window)
    den_img = _band_value(cube, wns, float(den_center), den_halfwidth if den_halfwidth is not None else window)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio_img = num_img / (den_img + EPS)
    if APPLY_RATIO_HARD_CAP:
        ratio_img = np.clip(ratio_img, 0, RATIO_HARD_CAP)

    valid = np.isfinite(ratio_img) & (ratio_img > 0)
    n_sel_valid = int((mask_sel & valid).sum())
    return ratio_img, mask_sel, n_sel_valid, {"num": num_img, "den": den_img}

def compute_wt_ratio_thresholds(groups, cfg_dict):
    thresholds = {}
    for rkey, cfg in cfg_dict.items():
        num_c, num_w = cfg["num"]; den_c, den_w = cfg["den"]
        all_vals = []
        for n in groups.get("WT", []):
            ratio_img, mask_sel, _, _ = _ratio_map_for_sample(n, num_c, num_w, den_c, den_w)
            if ratio_img is None: continue
            valid = np.isfinite(ratio_img) & (ratio_img > 0)
            vals = ratio_img[mask_sel & valid].ravel()
            if vals.size: all_vals.append(vals)
        if not all_vals:
            thresholds[rkey] = np.nan
            logger.warning("No WT pixels found for ratio '%s' -> threshold NaN", rkey)
            continue
        wt_all = np.concatenate(all_vals)
        thr = float(np.nanpercentile(wt_all, WT_REF_PCTL)) if WT_REF_STAT == "percentile" else float(np.nanmedian(wt_all))
        thresholds[rkey] = thr
    return thresholds

def _render_white_with_two_overlays(ax, shape_rc, mask_above, mask_below,
                                   color_above=OVERLAY_COLOR,
                                   color_below=OVERLAY_COLOR_BELOW,
                                   alpha=OVERLAY_ALPHA):
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    H, W = shape_rc

    overlay_blue = np.zeros((H, W, 4), dtype=float)
    overlay_blue[..., :3] = color_below
    overlay_blue[..., 3]  = 0.0
    if mask_below is not None:
        overlay_blue[mask_below, 3] = alpha
    ax.imshow(overlay_blue)

    overlay_red = np.zeros((H, W, 4), dtype=float)
    overlay_red[..., :3] = color_above
    overlay_red[..., 3]  = 0.0
    if mask_above is not None:
        overlay_red[mask_above, 3] = alpha
    ax.imshow(overlay_red)

def make_ratio_overlay_for_sample(sample, rkey, cfg, wt_threshold, out_dir):
    num_c, num_w = cfg["num"]; den_c, den_w = cfg["den"]
    ratio_img, mask_sel, _, bands = _ratio_map_for_sample(sample, num_c, num_w, den_c, den_w)
    if ratio_img is None or bands is None:
        logger.warning("Skipping ratio overlay for %s / %s (missing inputs).", sample, rkey)
        return None

    valid = np.isfinite(ratio_img) & (ratio_img > 0)
    ok_num, _ = _num_guard(bands["num"])
    ok_den, _ = _denom_guard(bands["den"])
    mask_valid_sel = mask_sel & valid & ok_num & ok_den

    if not np.isfinite(wt_threshold) or mask_valid_sel.sum() == 0:
        mask_above = np.zeros_like(mask_sel, dtype=bool)
        mask_below = np.zeros_like(mask_sel, dtype=bool)
    else:
        mask_above = (ratio_img > wt_threshold) & mask_valid_sel
        mask_below = (ratio_img < wt_threshold) & mask_valid_sel

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    _render_white_with_two_overlays(ax, ratio_img.shape, mask_above, mask_below)
    fig.tight_layout()

    safe_ratio = _safe_name(rkey)
    out_png = os.path.join(out_dir, f"{sample}_overlay_ratio_{safe_ratio}_RedsBlues.png")
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    if SAVE_PDF_ALSO:
        fig.savefig(out_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

    return {
        "sample": sample,
        "ratio": rkey,
        "wt_threshold": float(wt_threshold),
        "n_valid_sel": int(mask_valid_sel.sum()),
        "n_above": int(mask_above.sum()),
        "n_below": int(mask_below.sum()),
    }

def save_ratio_overlay_summary(rows, out_csv):
    headers = ["sample","ratio","wt_threshold","n_valid_sel","n_above","n_below"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    logger.info("Saved ratio overlay summary: %s", out_csv)

# =============================== MAIN =================================
if __name__ == "__main__":
    logger.info("Analysis cap: MAX_POINTS_PER_SAMPLE=%d | Plot frac=%.4f (min %d pts/sample)",
                MAX_POINTS_PER_SAMPLE, PLOT_SUBSAMPLE_FRAC, MIN_PLOT_POINTS)
    logger.info("Total ratios: %d", len(ALL_RATIOS))

    stats_rows = []
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    # ---------- RATIOS: collect per-sample arrays ----------
    with LogTimer("Collect per-sample ratios for ALL_RATIOS (pixel-aligned; capped pixels/sample)"):
        WT_ratio_dicts = {r: {} for r in ALL_RATIOS.keys()}
        KO_ratio_dicts = {r: {} for r in ALL_RATIOS.keys()}

        WT_concat = {r: [] for r in ALL_RATIOS.keys()}
        KO_concat = {r: [] for r in ALL_RATIOS.keys()}

        for n in sample_names:
            if sample_to_group.get(n) not in ("WT", "KO"):
                continue
            per_ratios = _collect_ratios_for_sample_generic(n, ALL_RATIOS)
            if per_ratios is None:
                continue
            for rkey, arr in per_ratios.items():
                if arr is None or len(arr) == 0:
                    continue
                if sample_to_group[n] == "WT":
                    WT_ratio_dicts[rkey][n] = arr
                    WT_concat[rkey].append(arr)
                else:
                    KO_ratio_dicts[rkey][n] = arr
                    KO_concat[rkey].append(arr)

    # ---------- RATIOS: stats + combined plots ----------
    with LogTimer("Make COMBINED plots for ALL ratios (scatter style MATCHED)"):
        for rkey in ALL_RATIOS.keys():
            WT_all = _concat(WT_concat[rkey])
            KO_all = _concat(KO_concat[rkey])

            row = cliffs_only_stats(rkey, WT_all, KO_all)
            stats_rows.append(row)

            log_y = USE_LOG_Y_FOR.get(rkey, False)
            ylim = MANUAL_YLIMS.get(rkey, None) if USE_MANUAL_YLIMS else _auto_ylim(WT_all, KO_all, log=log_y)

            ylabel = (rkey.replace("AmideI", "Amide I")
                          .replace("AmideII", "Amide II")
                          .replace("po2_1080", "PO2 1080")
                          .replace("po2_1235", "PO2 1235")
                          .replace("carb_1030", "Carb 1030")
                          .replace("carb_1155", "Carb 1155"))

            _combined_scatter_and_overlay(
                WT_ratio_dicts[rkey], KO_ratio_dicts[rkey],
                ylabel=ylabel,
                out_png=os.path.join(save_dir, f"ratio_{_safe_name(rkey)}_combined.png"),
                log_y=log_y, ylim=ylim,
                stats_text=format_stats_for_title(row)
            )

    # ---------- SINGLE-BAND intensities: stats + combined plots ----------
    with LogTimer("Single-band intensity plots (COMBINED figures)"):
        for bkey in BAND_LIBRARY.keys():
            center, _ = BAND_LIBRARY[bkey]
            WT_s, KO_s = [], []
            for n in groups["WT"]:
                v = _collect_band_for_sample(n, bkey)
                if v is not None: WT_s.append(v)
            for n in groups["KO"]:
                v = _collect_band_for_sample(n, bkey)
                if v is not None: KO_s.append(v)

            WT_all = _concat(WT_s)
            KO_all = _concat(KO_s)

            row = cliffs_only_stats(f"{bkey}@{center:.1f}", WT_all, KO_all)
            stats_rows.append(row)

            WT_d, KO_d = {}, {}
            for n in sample_names:
                g = sample_to_group.get(n)
                if g is None: continue
                v = _collect_band_for_sample(n, bkey)
                if v is None or len(v) == 0: continue
                if g == "WT": WT_d[n] = v
                else:         KO_d[n] = v

            log_y = USE_LOG_Y_FOR.get(bkey, False)
            ylim = MANUAL_YLIMS.get(bkey, None) if USE_MANUAL_YLIMS else _auto_ylim(WT_all, KO_all, log=log_y)

            _combined_scatter_and_overlay(
                WT_d, KO_d,
                ylabel=BAND_LABELS.get(bkey, f"Band {bkey}"),
                out_png=os.path.join(save_dir, f"band_{bkey}_{int(center)}cm-1_combined.png"),
                log_y=log_y, ylim=ylim,
                stats_text=format_stats_for_title(row)
            )

    # ---------- Save stats ----------
    if stats_rows:
        save_stats_csv(stats_rows, out_prefix + "_WT_vs_KO_cliffs_only.csv")

    # ---------- RATIO OVERLAYS (WHITE + red/blue) ----------
    logger.info("Computing WT thresholds per ratio for overlay…")
    wt_ratio_thresh = compute_wt_ratio_thresholds(groups, ALL_RATIOS)

    with LogTimer("Generate ratio overlays for ALL_RATIOS"):
        summary_rows = []
        for rkey, cfg in ALL_RATIOS.items():
            thr = wt_ratio_thresh.get(rkey, np.nan)
            for n in sample_names:
                r = make_ratio_overlay_for_sample(n, rkey, cfg, thr, save_dir)
                if r is not None:
                    summary_rows.append(r)

    save_ratio_overlay_summary(summary_rows, os.path.join(save_dir, "RATIO_overlay_summary.csv"))
    logger.info("Done.")
