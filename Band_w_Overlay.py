# ======================================================================
# WT vs KO spectral analysis + overlays (SHIFT floors + optional L2 norm)
# UPDATED: auto-generate ratios for ALL bands vs Amide I AND vs Amide II
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
save_dir    = os.path.join(cube_dir, "WT_KO_bands_Over_3")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "WT_KO")

# ---------------------------------------------------------------------------
sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}
# ---------------------------------------------------------------------------

# ---------------- Cluster selection (Right) ----------------
selected_clusters = {
    "T1":[0,1,2,3,4,5,6,7],
    "T2":[0,1,2,3,4,5,6,7],
    "T3":[0,1,2,3,4,5,6,7],
    "T6":[0,1,2,3,4,5,6,7],
    "T17":[0,1,2,3,4,5,6,7],
    "T19":[0,1,2,3,4.5,6,7],
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

TOPK_PER_BAND = 1  # mean of the top-k values inside the window (per pixel)

# ---- Dual shift floors ----
SHIFT_MIN_FLOOR_BANDS  = 0.2  # single-band intensity work
SHIFT_MIN_FLOOR_RATIOS = 0.5  # all ratio computations

# ---- L2 normalization toggles (after shift) ----
USE_L2_NORM_BANDS  = False     # apply per-pixel L2 norm for band-intensity work
USE_L2_NORM_RATIOS = False     # apply per-pixel L2 norm for all ratio work
EPS = 1e-12                   # numerical safety

# ===================== BAND CENTERS (cm-1) =====================
# Protein / lipid
BAND_1734_CENTER    = 1734.0
BAND_1734_HALFWIDTH = 13
BAND_AMIDE_I        = 1655.0
BAND_AMIDE_II       = 1545.0
BAND_CH2            = 1464.0
BAND_CH3            = 1375.0

# Phosphate (PO2−) proxies
BAND_PO2_SYM        = 1080.0
BAND_PO2_ASYM       = 1235.0

# Carbohydrate proxies
BAND_CARB_1030      = 1030.0
BAND_CARB_1155      = 1155.0

BAND_CARB_HALFWIDTH = 15
BAND_PO2_HALFWIDTH  = 15

# ===================== MANUAL AXES & PIXEL CAPS =======================
USE_MANUAL_YLIMS = True
MANUAL_YLIMS = {
    # Keep your existing keys if you want to set hard limits.
    # Anything not in this dict will auto-scale (or remain None when USE_MANUAL_YLIMS=True).
    "CH2/CH3":           None,
    "AmideI/AmideII":    None,
    "single_band":       None,
}

USE_LOG_Y_FOR = {
    "CH2/CH3":        False,
    "AmideI/AmideII": False,
    "single_band":    False,
}

PIXEL_CAP_DEFAULT = 30_000   # used if a metric key is not in PIXEL_CAP_PER_GROUP
PIXEL_CAP_PER_GROUP = {
    # optional overrides; otherwise PIXEL_CAP_DEFAULT applies
}

MAX_POINTS_PER_SAMPLE = 2500

# === Figure text policy ===
ANNOTATE_STATS_ON_PLOTS = True  # show only Cliff's delta
SEED = 1
rng_global = np.random.default_rng(SEED)

# ===================== OVERLAY CONFIG =====================
WT_REF_STAT = "median"   # or "percentile"
WT_REF_PCTL = 50.0
OVERLAY_ALPHA = 0.75
OVERLAY_COLOR = (0.85, 0.2, 0.2)        # red-ish for ABOVE
OVERLAY_COLOR_BELOW = (0.2, 0.2, 0.85)  # blue-ish for BELOW

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
SHOW_SAMPLE_LEGEND = False  # force no legends on scatter plots

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11

FS = {
    "base":   int(BASE_FONTSIZE),
    "axes":   int(BASE_FONTSIZE + 1),
    "title":  int(BASE_FONTSIZE + 3),
    "tick":   int(BASE_FONTSIZE),
    "annot":  10,
    "legend": int(BASE_FONTSIZE),
}

VIOLIN_WT_COLOR = (0.3, 0.7, 0.3, 0.18)
VIOLIN_KO_COLOR = (0.9, 0.3, 0.3, 0.22)

TRIM_FOR_PLOTTING_ONLY = True
APPLY_RATIO_HARD_CAP = True
RATIO_HARD_CAP       = 10.0

# Guard thresholds (used for overlays as well)
DENOM_MIN_PCT        = 3.0
DENOM_ABS_FLOOR      = 1e-2
NUM_MIN_PCT          = 15.0
NUM_ABS_FLOOR        = 1e-4

# ========================== BAND LIBRARY (DEFINE ONCE) ======================
# Each entry: key -> (center_cm1, halfwidth)
# Use `window` for broader bands; custom halfwidth for narrower peaks.
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

# Pretty labels for plots
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
        "axes.linewidth": 0.8,
        "grid.color": "#9aa0a6",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
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

# ===================== PREPROCESS (SHIFT + optional L2) ===============
def _shift_spectra_min_floor(cube, floor):
    C = cube.astype(np.float64, copy=True)
    mins = np.min(C, axis=2, keepdims=True)
    add = np.clip(floor - mins, 0.0, None)
    return C + add

def _l2_normalize_cube(C):
    norms = np.sqrt(np.sum(C*C, axis=2, keepdims=True)) + EPS
    return C / norms

def _preprocess_cube(cube, wns, *, floor: float, l2: bool):
    C = _shift_spectra_min_floor(cube, floor=floor)
    if l2:
        C = _l2_normalize_cube(C)
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

# ========================== SMALL UTILS ================================
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

# ========================== PLOTTING HELPERS ==========================
def _group_sample_colors(wt_keys, ko_keys):
    def ramp(hex_color, n, v_min=0.45, v_max=0.95, s=0.85):
        r,g,b = mcolors.to_rgb(hex_color)
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        vs = np.linspace(v_min, v_max, max(n,1))
        return [colorsys.hsv_to_rgb(h, s, v) for v in vs]
    wt_cols = ramp(BASE_WT, len(wt_keys))
    ko_cols = ramp(BASE_KO, len(ko_keys))
    return {k: wt_cols[i] for i,k in enumerate(wt_keys)}, {k: ko_cols[i] for i,k in enumerate(ko_keys)}

def _strip_plot_by_sample(WT_dict, KO_dict, ylabel, out_png, log_y=False, ylim=None, stats_text=None):
    _apply_pub_style()

    WT_all = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    if WT_all.size == 0 and KO_all.size == 0:
        logger.warning("Nothing to plot for %s", out_png)
        return

    if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
        WTv = _mask_for_ylim(WT_all, ylim)
        KOv = _mask_for_ylim(KO_all, ylim)
    else:
        WTv, KOv = WT_all, KO_all

    if WTv.size == 0 and KOv.size == 0:
        logger.warning("All values fell outside ylim for %s; skipping plot.", out_png)
        return

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    rng = np.random.default_rng(0)

    if USE_VIOLIN_BACKGROUND and (WTv.size > 0) and (KOv.size > 0):
        parts_ko = ax.violinplot([KOv], positions=[1], widths=0.6, showmeans=False, showmedians=False, showextrema=False)
        parts_ko["bodies"][0].set_facecolor(VIOLIN_KO_COLOR); parts_ko["bodies"][0].set_edgecolor("none")
        parts_wt = ax.violinplot([WTv], positions=[0], widths=0.6, showmeans=False, showmedians=False, showextrema=False)
        parts_wt["bodies"][0].set_facecolor(VIOLIN_WT_COLOR); parts_wt["bodies"][0].set_edgecolor("none")

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

    ax.yaxis.grid(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _annotate(ax, stats_text)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=PUB_DPI)
    if SAVE_PDF_ALSO:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".pdf", bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)

def _overlay_violin(WT_dict, KO_dict, ylabel, out_png, log_y=False, ylim=None, stats_text=None):
    _apply_pub_style()

    WT_all = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    if WT_all.size == 0 and KO_all.size == 0:
        logger.warning("Nothing to overlay for %s", out_png); return

    if TRIM_FOR_PLOTTING_ONLY and ylim is not None:
        WTv = _mask_for_ylim(WT_all, ylim)
        KOv = _mask_for_ylim(KO_all, ylim)
    else:
        WTv, KOv = WT_all, KO_all

    if WTv.size == 0 and KOv.size == 0:
        logger.warning("All overlay values fell outside ylim for %s; skipping.", out_png)
        return

    fig, ax = plt.subplots(figsize=(6.6, 5.0))

    if KOv.size > 0:
        parts_ko = ax.violinplot([KOv], positions=[0], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
        bko = parts_ko["bodies"][0]; bko.set_facecolor(VIOLIN_KO_COLOR); bko.set_edgecolor("none"); bko.set_zorder(1)

    if WTv.size > 0:
        parts_wt = ax.violinplot([WTv], positions=[0], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
        bwt = parts_wt["bodies"][0]; bwt.set_facecolor(VIOLIN_WT_COLOR); bwt.set_edgecolor("none"); bwt.set_zorder(2)

    def _median_iqr(x, xpos, z=3):
        if x.size == 0: return
        med = np.median(x); q1, q3 = np.quantile(x, [0.25, 0.75])
        ax.scatter([xpos], [med], s=46, c="white", edgecolors="black", linewidths=0.9, zorder=z)
        ax.vlines(xpos, q1, q3, colors="black", linewidth=1.2, zorder=z)

    _median_iqr(KOv, 0, z=2.2); _median_iqr(WTv, 0, z=2.5)

    ax.set_xticks([0], ["WT on KO"])
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylabel(ylabel)

    if log_y:
        ax.set_yscale("log")
        if ylim is not None:
            ax.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax.set_ylim(*ylim)

    ax.yaxis.grid(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _annotate(ax, stats_text)

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
    return f"δ={d:+.2f}"

# ======================= DATA GATHER (DUAL FLOORS + L2) ===============
def _collect_per_pixel_values(sample, bands_to_get, *, floor, use_l2):
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
    cube = _preprocess_cube(cube, wns, floor=floor, l2=use_l2)

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

def _collect_band_for_sample(sample, band_key):
    center, halfw = BAND_LIBRARY[band_key]
    per = _collect_per_pixel_values(
        sample,
        {"band": (center, halfw)},
        floor=SHIFT_MIN_FLOOR_BANDS,
        use_l2=USE_L2_NORM_BANDS
    )
    if per is None: return None
    return per["band"]

# ---- Helpers to gather for STATS (concatenated) ----
def _cap_group(arr, metric_key, seed):
    cap = PIXEL_CAP_PER_GROUP.get(metric_key, PIXEL_CAP_DEFAULT)
    if len(arr) <= cap: return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(len(arr), size=cap, replace=False)]

def _concat(lst): return np.concatenate(lst) if lst else np.array([])

# ======================= RATIO GENERATION (AUTO) =======================
def build_all_band_vs_amide_ratios():
    """
    Build ratio definitions for ALL non-amide bands vs AmideI and vs AmideII.
    Output dict: ratio_name -> {"num": (center, halfw), "den": (center, halfw)}
    """
    ratios = {}
    for bkey, (bc, bw) in BAND_LIBRARY.items():
        if bkey in ("amideI", "amideII"):
            continue
        # vs Amide I
        ratios[f"{bkey}/AmideI"]  = {"num": (bc, bw), "den": BAND_LIBRARY["amideI"]}
        # vs Amide II
        ratios[f"{bkey}/AmideII"] = {"num": (bc, bw), "den": BAND_LIBRARY["amideII"]}
    return ratios

BASE_RATIOS = {
    "CH2/CH3":        {"num": BAND_LIBRARY["ch2"],    "den": BAND_LIBRARY["ch3"]},
    "AmideI/AmideII": {"num": BAND_LIBRARY["amideI"], "den": BAND_LIBRARY["amideII"]},
}

# All ratios we will compute for stats/plots/overlays:
ALL_BAND_VS_AMIDE_RATIOS = build_all_band_vs_amide_ratios()
ALL_RATIOS = {}
ALL_RATIOS.update(BASE_RATIOS)
ALL_RATIOS.update(ALL_BAND_VS_AMIDE_RATIOS)

def _collect_ratios_for_sample_generic(sample, ratio_cfg_dict):
    """
    Collect ALL band images needed for ratio_cfg_dict in ONE pass (same pixels),
    then compute ratio arrays using a shared aligned length per sample.
    Returns dict: ratio_name -> 1D array
    """
    # Determine required unique bands (num and den for each ratio)
    required = {}
    for rkey, cfg in ratio_cfg_dict.items():
        required[f"__num__{rkey}"] = cfg["num"]
        required[f"__den__{rkey}"] = cfg["den"]

    per = _collect_per_pixel_values(sample, required, floor=SHIFT_MIN_FLOOR_RATIOS, use_l2=USE_L2_NORM_RATIOS)
    if per is None:
        return None

    # Align to shared length L (conservative)
    lengths = [len(v) for v in per.values() if v is not None]
    if not lengths:
        return None
    L = min(lengths)
    if L <= 0:
        return None

    out = {}
    for rkey, cfg in ratio_cfg_dict.items():
        num = per[f"__num__{rkey}"][:L]
        den = per[f"__den__{rkey}"][:L]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            rr = num / den
        if APPLY_RATIO_HARD_CAP:
            rr = np.clip(rr, 0, RATIO_HARD_CAP)
        rr = rr[np.isfinite(rr) & (rr > 0)]
        out[rkey] = rr
    return out

# ======================= OVERLAY HELPERS (WHITE + red/blue) ===========
def _ratio_map_for_sample(sample, num_center, num_halfwidth, den_center, den_halfwidth):
    out = _load_cluster_labels_indices(sample); cube = _load_padded_cube(sample)
    if out is None or cube is None:
        return None, None, 0, None

    labels, indices = out
    H, W, Z = cube.shape
    wns = np.linspace(950, 1800, Z)
    cube = _preprocess_cube(cube, wns, floor=SHIFT_MIN_FLOOR_RATIOS, l2=USE_L2_NORM_RATIOS)

    sel = selected_clusters.get(sample, [])
    if not sel:
        return None, None, 0, None

    mask_sel = _selected_mask_from_clusters(H, W, labels, indices, sel)

    num_img = _band_value(cube, wns, float(num_center), num_halfwidth if num_halfwidth is not None else window)
    den_img = _band_value(cube, wns, float(den_center), den_halfwidth if den_halfwidth is not None else window)

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio_img = num_img / den_img
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
        logger.info("WT threshold for ratio %s: %.6g (n=%d WT pixels)", rkey, thr, wt_all.size)
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
    os.makedirs(out_dir, exist_ok=True)

    ratio_img, mask_sel, _, bands = _ratio_map_for_sample(sample, num_c, num_w, den_c, den_w)
    if ratio_img is None or bands is None:
        logger.warning("Skipping ratio overlay for %s / %s (missing inputs).", sample, rkey)
        return [{
            "sample": sample, "ratio": rkey, "n_sel": 0,
            "n_above": 0, "pct_above": np.nan,
            "n_below": 0, "pct_below": np.nan,
            "wt_threshold": wt_threshold, "bg": "white_only"
        }]

    valid = np.isfinite(ratio_img) & (ratio_img > 0)
    ok_num, thr_num = _num_guard(bands["num"])
    ok_den, thr_den = _denom_guard(bands["den"])
    mask_valid_sel = mask_sel & valid & ok_num & ok_den
    n_sel = int(mask_valid_sel.sum())

    if not np.isfinite(wt_threshold) or n_sel == 0:
        mask_above = np.zeros_like(mask_sel, dtype=bool)
        mask_below = np.zeros_like(mask_sel, dtype=bool)
        n_above = n_below = 0
        pct_above = pct_below = np.nan
    else:
        mask_above = (ratio_img > wt_threshold) & mask_valid_sel
        mask_below = (ratio_img < wt_threshold) & mask_valid_sel
        n_above = int(mask_above.sum())
        n_below = int(mask_below.sum())
        pct_above = 100.0 * n_above / n_sel if n_sel > 0 else np.nan
        pct_below = 100.0 * n_below / n_sel if n_sel > 0 else np.nan

    safe_ratio = _safe_name(rkey)
    out_npz = os.path.join(out_dir, f"{sample}_overlay_ratio_{safe_ratio}_RedsBlues.npz")
    np.savez_compressed(
        out_npz,
        mask_above=mask_above.astype(np.uint8),
        mask_below=mask_below.astype(np.uint8),
        mask_selected=mask_sel.astype(np.uint8),
        mask_valid_sel=mask_valid_sel.astype(np.uint8),
        threshold=float(wt_threshold),
        ratio_key=rkey,
        num_center=float(num_c), num_halfwidth=float(num_w),
        den_center=float(den_c), den_halfwidth=float(den_w),
        thr_num=float(thr_num), thr_den=float(thr_den),
        bg_mode="white_only", colors=("red_above", "blue_below")
    )

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    _render_white_with_two_overlays(
        ax, ratio_img.shape, mask_above, mask_below,
        color_above=OVERLAY_COLOR,
        color_below=OVERLAY_COLOR_BELOW,
        alpha=OVERLAY_ALPHA
    )
    fig.tight_layout()
    out_png = os.path.join(out_dir, f"{sample}_overlay_ratio_{safe_ratio}_RedsBlues.png")
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    if SAVE_PDF_ALSO:
        fig.savefig(out_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

    return [{
        "sample": sample, "ratio": rkey, "n_sel": n_sel,
        "n_above": int(mask_above.sum()), "pct_above": pct_above,
        "n_below": int(mask_below.sum()), "pct_below": pct_below,
        "wt_threshold": float(wt_threshold), "bg": "white_only"
    }]

# ======================= SUMMARY CSV HELPERS ==========================
def save_ratio_overlay_summary(rows, out_csv):
    headers = ["sample", "ratio", "n_sel", "n_above", "pct_above", "n_below", "pct_below", "wt_threshold", "bg"]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    logger.info("Saved ratio overlay summary: %s", out_csv)

# =============================== MAIN =================================
if __name__ == "__main__":
    logger.info("Shift floors: bands=%.2f, ratios=%.2f | L2: bands=%s ratios=%s",
                SHIFT_MIN_FLOOR_BANDS, SHIFT_MIN_FLOOR_RATIOS, USE_L2_NORM_BANDS, USE_L2_NORM_RATIOS)
    logger.info("Total ratios (including ALL bands vs AmideI/AmideII): %d", len(ALL_RATIOS))

    stats_rows = []
    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    # ---------- RATIOS: compute ALL ratios per sample (stats + plots) ----------
    with LogTimer("Collect per-sample ratios for ALL_RATIOS (stats + plotting)"):
        WT_ratio_dicts = {r: {} for r in ALL_RATIOS.keys()}  # r -> {sample -> arr}
        KO_ratio_dicts = {r: {} for r in ALL_RATIOS.keys()}

        WT_concat = {r: [] for r in ALL_RATIOS.keys()}       # r -> list of arrays
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

    # For each ratio: stats + plots
    with LogTimer("Make strip+overlay plots for ALL ratios (band/AmideI and band/AmideII included)"):
        for rkey in ALL_RATIOS.keys():
            WT_all = _concat(WT_concat[rkey])
            KO_all = _concat(KO_concat[rkey])

            WT_all = _cap_group(WT_all, rkey, seed=100 + abs(hash(rkey)) % 10_000)
            KO_all = _cap_group(KO_all, rkey, seed=200 + abs(hash(rkey)) % 10_000)

            row = cliffs_only_stats(rkey, WT_all, KO_all)
            stats_rows.append(row)

            # y-lims / log policy
            log_y = USE_LOG_Y_FOR.get(rkey, False)
            ylim = MANUAL_YLIMS.get(rkey, None) if USE_MANUAL_YLIMS else _auto_ylim(WT_all, KO_all, log=log_y)

            # Plot label formatting
            ylabel = rkey.replace("AmideI", "Amide I").replace("AmideII", "Amide II")

            _strip_plot_by_sample(
                WT_ratio_dicts[rkey], KO_ratio_dicts[rkey],
                ylabel=ylabel,
                out_png=os.path.join(save_dir, f"ratio_{_safe_name(rkey)}_strip_by_sample.png"),
                log_y=log_y, ylim=ylim,
                stats_text=format_stats_for_title(row)
            )
            _overlay_violin(
                WT_ratio_dicts[rkey], KO_ratio_dicts[rkey],
                ylabel=ylabel,
                out_png=os.path.join(save_dir, f"ratio_{_safe_name(rkey)}_overlayviolin.png"),
                log_y=log_y, ylim=ylim,
                stats_text=format_stats_for_title(row)
            )

    # ---------- SINGLE-BAND intensities (optional, keep) ----------
    with LogTimer("Single-band intensity plots (optional)"):
        for bkey in BAND_LIBRARY.keys():
            center, halfw = BAND_LIBRARY[bkey]
            WT_s, KO_s = [], []
            for n in groups["WT"]:
                v = _collect_band_for_sample(n, bkey)
                if v is not None: WT_s.append(v)
            for n in groups["KO"]:
                v = _collect_band_for_sample(n, bkey)
                if v is not None: KO_s.append(v)

            WT_s = _cap_group(_concat(WT_s), f"{bkey}_single", seed=1000 + abs(hash(bkey)) % 10_000)
            KO_s = _cap_group(_concat(KO_s), f"{bkey}_single", seed=2000 + abs(hash(bkey)) % 10_000)

            row = cliffs_only_stats(f"{bkey}@{center:.1f}", WT_s, KO_s)
            stats_rows.append(row)

            # per-sample dicts
            WT_d, KO_d = {}, {}
            for n in sample_names:
                g = sample_to_group.get(n)
                if g is None: continue
                v = _collect_band_for_sample(n, bkey)
                if v is None or len(v) == 0: continue
                if g == "WT": WT_d[n] = v
                else:         KO_d[n] = v

            log_y = USE_LOG_Y_FOR.get(bkey, False)
            ylim = MANUAL_YLIMS.get(bkey, None) if USE_MANUAL_YLIMS else _auto_ylim(WT_s, KO_s, log=log_y)

            _strip_plot_by_sample(
                WT_d, KO_d,
                ylabel=BAND_LABELS.get(bkey, f"Band {bkey}"),
                out_png=os.path.join(save_dir, f"band_{bkey}_{int(center)}cm-1_by_sample.png"),
                log_y=log_y, ylim=ylim,
                stats_text=format_stats_for_title(row)
            )
            _overlay_violin(
                WT_d, KO_d,
                ylabel=BAND_LABELS.get(bkey, f"Band {bkey}"),
                out_png=os.path.join(save_dir, f"band_{bkey}_overlayviolin.png"),
                log_y=log_y, ylim=ylim,
                stats_text=format_stats_for_title(row)
            )

    # ---------- Save stats table ----------
    if stats_rows:
        save_stats_csv(stats_rows, out_prefix + "_WT_vs_KO_cliffs_only.csv")

    # ---------- RATIO OVERLAYS (WHITE + red/blue) for ALL ratios ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        logger.info("Computing WT thresholds per RATIO for overlay… (ALL bands/AmideI and bands/AmideII included)")
        wt_ratio_thresh = compute_wt_ratio_thresholds(groups, ALL_RATIOS)

        ratio_rows_all = []
        with LogTimer("Generate ratio overlays for ALL_RATIOS"):
            for rkey, cfg in ALL_RATIOS.items():
                thr = wt_ratio_thresh.get(rkey, np.nan)
                for n in sample_names:
                    rows = make_ratio_overlay_for_sample(
                        sample=n, rkey=rkey, cfg=cfg, wt_threshold=thr, out_dir=save_dir
                    )
                    ratio_rows_all.extend(rows)

        csv_name = f"{_safe_name(os.path.basename(out_prefix))}_RATIO_overlays_vs_WT_ALLbands_to_AmideI_AmideII.csv"
        save_ratio_overlay_summary(ratio_rows_all, os.path.join(save_dir, csv_name))

    logger.info("Done.")
