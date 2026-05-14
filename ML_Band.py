# -*- coding: utf-8 -*-
"""
WT vs KO analysis + DG 10x10 patch Linear SVM (all clusters, leakage-free)
Figures hide sample names; annotations show ONLY p-values (Mann-Whitney U).
Includes ML plots: confusion matrices, ROC/PR curves, per-sample bars (anonymized), feature importance.
"""

import os, logging, time, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from contextlib import contextmanager
from scipy.signal import savgol_filter
from scipy import stats   # mannwhitneyu, rankdata, etc.

# ============================== REGION / PATHS ==============================
REGION = "Area_CA"  # set to "Area_DG" or "Area_CA"

base_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)"
cluster_dir = os.path.join(base_dir, REGION, "UMAP_clustering_8Cluster")
cube_dir    = os.path.join(base_dir, REGION)
save_dir    = os.path.join(cube_dir, "WT_KO_Classification")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "WT_KO")

sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

# ---------------- Cluster selection (kept for masks) ----------------
selected_clusters = {
    "T1":[0,1,2,3,4,5,6,7],
    "T2":[0,1,2,3,4,7],
    "T3":[0,1,2,3,5,6,7],
    "T6":[0,1,3,5,6,7],
    "T17":[0,1,2,3,5,6,7],
    "T19":[0,1,3,4,7],
    "T20":[0,1,2,3,4,5,7],
    "T21":[1,2,4,5,6,7],
    "T10":[1,2,3,6,5,7],
    "T11":[0,1,2,5,6,7],
    "T12":[0,1,2,4,6,7],
    "T13":[0,1,3,4,5,6,7],
    "T14":[0,1,2,4,5,6,7],
    "T15":[0,2,3,4,5,6,7],
    "T22":[0,1,2,3,4,6],
}

# --------- Spectral / band settings ---------
savgol_window = 11
polyorder     = 3
try:
    window
except NameError:
    window = 30

TOPK_PER_BAND = 1  # mean of the top-k values inside the window

# ---- Preprocess policy: shift ONLY (no L2 normalization) ----
NORM_MODE = "none"
SHIFT_MIN_FLOOR = 0.2
NORM_EPS  = 1e-12

# Bands (cm-1)
BAND_1734_CENTER    = 1734.0
BAND_1734_HALFWIDTH = 13
BAND_AMIDE_I        = 1655.0
BAND_AMIDE_II       = 1545.0
BAND_CH2            = 1464.0
BAND_CH3            = 1375.0

# ===================== MANUAL AXES & PIXEL CAPS =======================
USE_MANUAL_YLIMS = True
MANUAL_YLIMS = {k: None for k in [
    "CH2/CH3","AmideI/AmideII","1734/AmideI","1734/AmideII","CH2/AmideI","CH3/AmideI",
    "single_band","amideI","amideII","1734"
]}
USE_LOG_Y_FOR = {k: False for k in ["AmideI/AmideII","1734/AmideI","1734/AmideII","CH2/AmideI","CH3/AmideI","amideI","amideII","1734"]}

PIXEL_CAP_PER_GROUP = {k: 10_000 for k in ["CH2/CH3","AmideI/AmideII","1734/AmideI","1734/AmideII","CH2/AmideI","CH3/AmideI","single_band","amideI","amideII","1734"]}
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

DOT_SIZE     = 5
JITTER_SCALE = 0.06
USE_VIOLIN_BACKGROUND = True
SAVE_PDF_ALSO = True
SHOW_SAMPLE_LEGEND = False   # keep false (no sample names)

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
def _annotate(ax, text):
    if not ANNOTATE_STATS_ON_PLOTS or not text:
        return
    ax.text(0.5, 0.94, text, ha="center", va="top", transform=ax.transAxes, fontsize=10)

# ===== Subsample for plotting only (keeps training/stats intact) =====
PLOT_SUBSAMPLE_FRAC = 0.20
def _subsample_for_plot(arr, frac=0.20, seed=1234):
    if arr is None: return arr
    a = np.asarray(arr); a = a[np.isfinite(a)]
    n = a.size
    if n == 0 or frac >= 1.0: return a
    k = max(1, int(np.floor(n * float(frac))))
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    return a[idx]

# ===== Simple pooled strip/violin (no sample names on figure) =====
def _strip_plot_pooled(WT_all, KO_all, ylabel, title, out_png, log_y=False, ylim=None, stats_text=None):
    _apply_pub_style()

    # prepare pooled (display) arrays
    def _prep(arr, tag):
        arr2 = _mask_for_ylim(arr, ylim) if (ylim is not None and TRIM_FOR_PLOTTING_ONLY) else arr
        seed = abs(hash((title, tag))) % (2**32)
        return _subsample_for_plot(arr2, frac=PLOT_SUBSAMPLE_FRAC, seed=seed)

    WTv = _prep(WT_all, "WT"); KOv = _prep(KO_all, "KO")
    if WTv.size == 0 and KOv.size == 0:
        logging.warning("Nothing to plot for %s", title); return

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    rng = np.random.default_rng(0)

    # Violin background
    if USE_VIOLIN_BACKGROUND and (WTv.size > 0) and (KOv.size > 0):
        parts = ax.violinplot([WTv, KOv], positions=[0, 1], widths=0.6,
                               showmeans=False, showmedians=False, showextrema=False)
        parts["bodies"][0].set_facecolor(VIOLIN_WT_COLOR); parts["bodies"][0].set_edgecolor("none")
        parts["bodies"][1].set_facecolor(VIOLIN_KO_COLOR); parts["bodies"][1].set_edgecolor("none")

    # jittered dots (anonymous)
    x0 = rng.normal(0, JITTER_SCALE, size=WTv.size)
    x1 = rng.normal(1, JITTER_SCALE, size=KOv.size)
    ax.scatter(x0, WTv, s=DOT_SIZE, alpha=0.6, c=[BASE_WT], edgecolors="none", rasterized=True)
    ax.scatter(x1, KOv, s=DOT_SIZE, alpha=0.6, c=[BASE_KO], edgecolors="none", rasterized=True)

    # medians + whiskers
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
        if ylim is not None: ax.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(title)
    ax.yaxis.grid(True)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    _annotate(ax, stats_text)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=PUB_DPI)
    if SAVE_PDF_ALSO:
        root, _ = os.path.splitext(out_png)
        fig.savefig(root + ".pdf", bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)

# ========================== STATISTICS (p-value only) =================
def mwu_pvalue(WT, KO, alternative="two-sided"):
    WT = np.asarray(WT); KO = np.asarray(KO)
    WT = WT[np.isfinite(WT)]; KO = KO[np.isfinite(KO)]
    if WT.size == 0 or KO.size == 0:
        return np.nan, np.nan
    U, p = stats.mannwhitneyu(WT, KO, alternative=alternative, method="auto")
    return float(U), float(p)

def mwu_only_stats(name, WT, KO):
    WT = np.asarray(WT); KO = np.asarray(KO)
    WT = WT[np.isfinite(WT)]; KO = KO[np.isfinite(KO)]
    n_wt, n_ko = len(WT), len(KO)
    if n_wt == 0 or n_ko == 0:
        return {"metric": name,"n_wt": n_wt,"n_ko": n_ko,
                "mean_wt": np.nan,"mean_ko": np.nan,"median_wt": np.nan,"median_ko": np.nan,
                "sd_wt": np.nan,"sd_ko": np.nan,"U": np.nan,"p_value": np.nan,
                "median_diff": np.nan}
    U, p = mwu_pvalue(WT, KO, alternative="two-sided")
    return {"metric": name,"n_wt": n_wt,"n_ko": n_ko,
            "mean_wt": float(np.mean(WT)),"mean_ko": float(np.mean(KO)),
            "median_wt": float(np.median(WT)),"median_ko": float(np.median(KO)),
            "sd_wt": float(np.std(WT, ddof=1)) if n_wt>1 else np.nan,
            "sd_ko": float(np.std(KO, ddof=1)) if n_ko>1 else np.nan,
            "U": float(U),"p_value": float(p),
            "median_diff": float(np.median(WT) - np.median(KO))}

def save_stats_csv(rows, path):
    headers = ["metric","n_wt","n_ko","mean_wt","mean_ko","median_wt","median_ko","sd_wt","sd_ko","U","p_value","median_diff"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")
    logger.info("Saved stats table: %s", path)

def format_stats_for_title_p(row):
    p = row.get("p_value", np.nan)
    if not np.isfinite(p): return "p = n/a"
    if p < 1e-4: return "p < 1e-4"
    return f"p = {p:.2e}"

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
    return per["band"]

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
            per = _collect_per_pixel_values(n, {"ch2": (BAND_CH2, window), "ch3": (BAND_CH3, window), "amideI": (BAND_AMIDE_I, window)})
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

# ===================== SIGNED 2nd-DERIV & CENTROID LEANING (unchanged) ============
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
        # anonymize x labels
        plt.bar(x, margins, color=cols, edgecolor="k", alpha=0.85)
        plt.xticks(x, [f"S{i+1}" for i in range(len(bar))], rotation=90)
        plt.ylabel("Angle margin (mrad): θKO − θWT")
        plt.title("Centroid leaning (2nd-deriv, sign-corrected)")
        plt.tight_layout()
        path = out_prefix + "_centroid_margins.png"
        plt.savefig(path, dpi=600, bbox_inches="tight")
        if SAVE_PDF_ALSO:
            plt.savefig(out_prefix + "_centroid_margins.pdf", bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", path)
        return angles

# ====================== 10x10 PATCH SVM (HIPPO DG, ALL CLUSTERS) ======================
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, auc
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import pandas as pd

PATCH_SIZE = 10
STRIDE     = 10          # non-overlapping tiles
MIN_COVER  = 0.80        # >=80% of pixels in tile must be inside mask
MAX_PATCHES_PER_SAMPLE = 1000
N_SPLITS   = 5
RANDOM_SEED = 13

def _mask_all_clusters(H, W, indices):
    if indices.ndim == 2 and indices.shape[1] == 2:
        r, c = indices[:,0], indices[:,1]
        m = np.zeros((H, W), dtype=bool); m[r, c] = True
        return m
    else:
        flat = indices
        m = np.zeros((H*W,), dtype=bool); m[flat] = True
        return m.reshape(H, W)

def _tile_windows(mask2d, patch=PATCH_SIZE, stride=STRIDE, min_cover=MIN_COVER):
    H, W = mask2d.shape
    for r0 in range(0, H - patch + 1, stride):
        for c0 in range(0, W - patch + 1, stride):
            tile = mask2d[r0:r0+patch, c0:c0+patch]
            if tile.mean() >= min_cover:
                yield (r0, c0)

def _extract_patch_features(cube, wns, r0, c0, patch=PATCH_SIZE):
    tile = cube[r0:r0+patch, c0:c0+patch, :]
    b_ch2   = _band_value(tile, wns, BAND_CH2,     window)
    b_ch3   = _band_value(tile, wns, BAND_CH3,     window)
    b_amI   = _band_value(tile, wns, BAND_AMIDE_I, window)
    b_amII  = _band_value(tile, wns, BAND_AMIDE_II, window)
    b_1734  = _band_value(tile, wns, BAND_1734_CENTER, BAND_1734_HALFWIDTH)

    ch2 = b_ch2.ravel(); ch3 = b_ch3.ravel(); amI = b_amI.ravel(); amII = b_amII.ravel(); b17 = b_1734.ravel()
    def _clean(x):
        x = x[np.isfinite(x) & (x > 0)]
        return x if x.size else np.array([np.nan])
    ch2 = _clean(ch2); ch3 = _clean(ch3); amI = _clean(amI); amII = _clean(amII); b17 = _clean(b17)

    i_ch2 = float(np.nanmean(ch2)); i_ch3 = float(np.nanmean(ch3))
    i_ai  = float(np.nanmean(amI)); i_aii = float(np.nanmean(amII)); i_1734 = float(np.nanmean(b17))

    def _safe_ratio(a, b):
        if not np.isfinite(a) or not np.isfinite(b) or b <= 0: return np.nan
        return float(np.clip(a / b, 0, RATIO_HARD_CAP if APPLY_RATIO_HARD_CAP else np.inf))

    feats = {
        "I_CH2": i_ch2, "I_CH3": i_ch3, "I_AmideI": i_ai, "I_AmideII": i_aii, "I_1734": i_1734,
        "R_CH2_CH3":   _safe_ratio(i_ch2,  i_ch3),
        "R_AmI_AmII":  _safe_ratio(i_ai,   i_aii),
        "R_1734_AmI":  _safe_ratio(i_1734, i_ai),
        "R_1734_AmII": _safe_ratio(i_1734, i_aii),
        "R_CH2_AmI":   _safe_ratio(i_ch2,  i_ai),
        "R_CH3_AmI":   _safe_ratio(i_ch3,  i_ai),
    }
    return feats

def _collect_patches_for_sample(sample, label, max_patches=MAX_PATCHES_PER_SAMPLE):
    out = _load_cluster_labels_indices(sample)
    if out is None: return []
    labels, indices = out
    cube = _load_padded_cube(sample)
    if cube is None: return []
    H, W, Z = cube.shape
    mask2d = _mask_all_clusters(H, W, indices)
    if not np.any(mask2d):
        logger.warning("No DG/all-cluster pixels for %s; skipping.", sample)
        return []
    wns = np.linspace(950, 1800, Z)
    cube_pp = _preprocess_cube(cube, wns)
    coords = list(_tile_windows(mask2d))
    if not coords:
        logger.warning("No eligible %dx%d tiles for %s; skipping.", PATCH_SIZE, sample)
        return []
    rng = np.random.default_rng(abs(hash((sample, PATCH_SIZE, STRIDE))) % (2**32))
    if len(coords) > max_patches:
        idxs = rng.choice(len(coords), size=max_patches, replace=False)
        coords = [coords[i] for i in idxs]
    rows = []
    for (r0, c0) in coords:
        feats = _extract_patch_features(cube_pp, wns, r0, c0, patch=PATCH_SIZE)
        feats.update({"sample": sample, "label": int(label), "r0": r0, "c0": c0})
        rows.append(feats)
    return rows

def _build_patch_table(sample_names, groups):
    sample_to_group = {s:0 for s in groups["WT"]}
    sample_to_group.update({s:1 for s in groups["KO"]})
    table = []
    with LogTimer("Collecting 10x10 DG patch features across samples (balanced per sample)"):
        for s in sample_names:
            if s not in sample_to_group:
                logger.warning("Sample %s not in WT/KO groups; skipping.", s)
                continue
            label = sample_to_group[s]
            rows = _collect_patches_for_sample(s, label)
            if rows:
                table.extend(rows)
                logger.info("%s: kept %d patches", s, len(rows))
    if not table:
        logger.error("No patches collected — check masks/paths.")
        return None
    df = pd.DataFrame(table)
    feat_cols = [c for c in df.columns if c.startswith("I_") or c.startswith("R_")]
    df = df.dropna(subset=feat_cols)
    return df

# --------- Plotting for ML results ---------
def _plot_confusion_matrix(cm, labels, title, outfile, normalize=False):
    _apply_pub_style()
    if normalize:
        with np.errstate(invalid='ignore'):
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, interpolation='nearest', aspect='equal')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color="black")
    ax.set_ylabel("True"); ax.set_xlabel("Predicted")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(outfile, bbox_inches="tight", dpi=600)
    plt.close(fig)

def _plot_roc_pr(y_true, scores, prefix):
    _apply_pub_style()
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    pr_auc  = auc(rec, prec)

    # ROC
    fig1, ax1 = plt.subplots(figsize=(4.6, 4.2))
    ax1.plot(fpr, tpr, lw=2)
    ax1.plot([0,1],[0,1], ls="--", lw=1)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title(f"ROC (AUC={roc_auc:.3f})")
    fig1.tight_layout(); fig1.savefig(prefix + "_ROC.png", dpi=600, bbox_inches="tight")
    plt.close(fig1)

    # PR
    fig2, ax2 = plt.subplots(figsize=(4.6, 4.2))
    ax2.plot(rec, prec, lw=2)
    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision-Recall (AP={pr_auc:.3f})")
    fig2.tight_layout(); fig2.savefig(prefix + "_PR.png", dpi=600, bbox_inches="tight")
    plt.close(fig2)

def _plot_sample_scores_bar(agg_df, prefix):
    """Anonymous per-sample score bars (no sample names on x-axis)."""
    _apply_pub_style()
    df = agg_df.copy()
    df = df.sort_values("score")
    colors = [BASE_WT if y==0 else BASE_KO for y in df["y"]]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(range(len(df)), df["score"].values, color=colors, edgecolor="k", alpha=0.85)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(range(len(df))); ax.set_xticklabels([f"S{i+1}" for i in range(len(df))], rotation=90)
    ax.set_ylabel("Mean OOF score (KO positive)")
    ax.set_title("Per-sample decision scores (anonymous)")
    fig.tight_layout(); fig.savefig(prefix + "_sample_scores.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

def _plot_feature_importance(pipe, feature_names, prefix, top_k=12):
    clf = pipe.named_steps["svm"]
    if not hasattr(clf, "coef_"): return
    coefs = clf.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1]
    idx = order[:min(top_k, len(coefs))]
    feats = np.array(feature_names)[idx]
    vals  = coefs[idx]
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.barh(range(len(vals)), vals, edgecolor="k", alpha=0.9)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(feats)
    ax.invert_yaxis()
    ax.set_xlabel("LinearSVC coefficient (KO direction > 0)")
    ax.set_title("Feature importance (refit on all patches)")
    fig.tight_layout(); fig.savefig(prefix + "_feature_importance.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

# --------- Train / Evaluate (with plotting) ---------
def _fit_eval_linear_svm_patchcv(df, out_prefix, n_splits=N_SPLITS, seed=RANDOM_SEED):
    X_cols = [c for c in df.columns if c.startswith("I_") or c.startswith("R_")]
    X = df[X_cols].values
    y = df["label"].values
    groups_arr = df["sample"].values

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = sgkf.split(X, y, groups=groups_arr)
        used_sgkf = True
    except Exception:
        gkf = GroupKFold(n_splits=n_splits)
        splitter = gkf.split(X, y, groups=groups_arr)
        used_sgkf = False

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))])

    y_true_all, y_score_all, y_pred_all = [], [], []
    df_pred = df[["sample"]].copy()
    df_pred["fold"] = -1; df_pred["score"] = np.nan; df_pred["pred"] = -1

    fold_id = 0
    for train_idx, val_idx in splitter:
        fold_id += 1
        Xtr, Xva = X[train_idx], X[val_idx]
        ytr, yva = y[train_idx], y[val_idx]
        pipe.fit(Xtr, ytr)
        s = pipe.decision_function(Xva)
        p = (s >= 0.0).astype(int)
        y_true_all.append(yva); y_score_all.append(s); y_pred_all.append(p)
        df_pred.loc[df_pred.index[val_idx], "fold"] = fold_id
        df_pred.loc[df_pred.index[val_idx], "score"] = s
        df_pred.loc[df_pred.index[val_idx], "pred"] = p

    y_true = np.concatenate(y_true_all)
    y_score = np.concatenate(y_score_all)
    y_pred = np.concatenate(y_pred_all)

    patch_auc  = roc_auc_score(y_true, y_score)
    patch_ap   = average_precision_score(y_true, y_score)
    patch_acc  = accuracy_score(y_true, y_pred)
    patch_f1   = f1_score(y_true, y_pred)

    agg = (df_pred.assign(y=df["label"].values)
           .groupby("sample", as_index=False)
           .agg(y=("y","first"), score=("score","mean")))
    agg["pred"] = (agg["score"] >= 0.0).astype(int)

    sample_auc = roc_auc_score(agg["y"].values, agg["score"].values) if agg["y"].nunique()==2 else np.nan
    sample_acc = accuracy_score(agg["y"].values, agg["pred"].values)
    sample_f1  = f1_score(agg["y"].values, agg["pred"].values)

    # Save summary & predictions
    summary = {
        "n_patches": int(len(df)),
        "n_samples": int(agg.shape[0]),
        "cv_splits": n_splits,
        "used_stratified_group_kfold": bool(used_sgkf),
        "patch_ROC_AUC": float(patch_auc),
        "patch_PR_AUC":  float(patch_ap),
        "patch_ACC":     float(patch_acc),
        "patch_F1":      float(patch_f1),
        "sample_ROC_AUC": float(sample_auc) if np.isfinite(sample_auc) else None,
        "sample_ACC":     float(sample_acc),
        "sample_F1":      float(sample_f1)
    }
    logger.info("PATCH SVM summary: %s", summary)

    summ_path = out_prefix + "_svm10x10_summary.csv"
    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for k,v in summary.items(): w.writerow([k,v])

    pred_path = out_prefix + "_svm10x10_patch_preds.csv"
    df_out = df.copy()
    df_out["score"] = df_pred["score"].values
    df_out["pred"]  = df_pred["pred"].values
    df_out["fold"]  = df_pred["fold"].values
    df_out.to_csv(pred_path, index=False)

    samp_path = out_prefix + "_svm10x10_sample_preds.csv"
    agg.to_csv(samp_path, index=False)

    # ---- Plots: confusion matrices, ROC/PR, per-sample scores (anonymous) ----
    cm_patch = confusion_matrix(y_true, y_pred, labels=[0,1])
    cm_samp  = confusion_matrix(agg["y"], agg["pred"], labels=[0,1])
    np.savetxt(out_prefix + "_svm10x10_cm_patch.txt", cm_patch, fmt="%d")
    np.savetxt(out_prefix + "_svm10x10_cm_sample.txt", cm_samp, fmt="%d")

    _plot_confusion_matrix(cm_patch, ["WT","KO"], "Patch Confusion Matrix", out_prefix + "_svm10x10_cm_patch.png", normalize=False)
    _plot_confusion_matrix(cm_patch, ["WT","KO"], "Patch Confusion Matrix (Normalized)", out_prefix + "_svm10x10_cm_patch_norm.png", normalize=True)
    _plot_confusion_matrix(cm_samp,  ["WT","KO"], "Sample Confusion Matrix", out_prefix + "_svm10x10_cm_sample.png", normalize=False)
    _plot_confusion_matrix(cm_samp,  ["WT","KO"], "Sample Confusion Matrix (Normalized)", out_prefix + "_svm10x10_cm_sample_norm.png", normalize=True)

    _plot_roc_pr(y_true, y_score, out_prefix + "_svm10x10_oof")
    _plot_sample_scores_bar(agg.rename(columns={"sample":"sample"}), out_prefix + "_svm10x10")

    # ---- Feature importance: refit on ALL data (interpretability only) ----
    pipe_all = Pipeline([("scaler", StandardScaler()),
                         ("svm", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000))])
    pipe_all.fit(X, y)
    _plot_feature_importance(pipe_all, X_cols, out_prefix + "_svm10x10", top_k=12)

    return summary, agg

def run_patch_svm():
    df = _build_patch_table(sample_names, groups)
    if df is None or df.empty:
        logger.error("No patch dataframe to train on.")
        return
    summary, agg = _fit_eval_linear_svm_patchcv(df, out_prefix)
    logger.info("Patch-level: ROC-AUC=%.3f, PR-AUC=%.3f, ACC=%.3f, F1=%.3f",
                summary["patch_ROC_AUC"], summary["patch_PR_AUC"], summary["patch_ACC"], summary["patch_F1"])
    if summary["sample_ROC_AUC"] is not None:
        logger.info("Sample-level: ROC-AUC=%.3f, ACC=%.3f, F1=%.3f",
                    summary["sample_ROC_AUC"], summary["sample_ACC"], summary["sample_F1"])
    else:
        logger.info("Sample-level: ACC=%.3f, F1=%.3f", summary["sample_ACC"], summary["sample_F1"])

# ========================== MAIN RUN ==================================
if __name__ == "__main__":
    logger.info("Region=%s | Preprocess: shift floor=%.3g; normalization=%s",
                REGION, SHIFT_MIN_FLOOR, ("NONE" if NORM_MODE.lower()=="none" else NORM_MODE))
    stats_rows = []

    # ---------- Base ratios (CH2/CH3 & AmideI/II) ----------
    if sample_names and groups.get("WT") and groups.get("KO") and selected_clusters:
        (ch2ch3_WT_s, ch2ch3_KO_s), (ai_aii_WT_s, ai_aii_KO_s) = _gather_by_group_for_stats(sample_names, groups)

        row_ch = mwu_only_stats("CH2/CH3", ch2ch3_WT_s, ch2ch3_KO_s); stats_rows.append(row_ch)
        row_ai = mwu_only_stats("AmideI/AmideII", ai_aii_WT_s, ai_aii_KO_s); stats_rows.append(row_ai)

        # pooled plots (no sample names)
        ylim_ch2ch3 = MANUAL_YLIMS["CH2/CH3"] if USE_MANUAL_YLIMS else _auto_ylim(ch2ch3_WT_s, ch2ch3_KO_s, log=False)
        ylim_ai_aii = MANUAL_YLIMS["AmideI/AmideII"] if USE_MANUAL_YLIMS else _auto_ylim(ai_aii_WT_s, ai_aii_KO_s, log=USE_LOG_Y_FOR["AmideI/AmideII"])

        _strip_plot_pooled(
            ch2ch3_WT_s, ch2ch3_KO_s,
            ylabel="CH$_2$/CH$_3$ ratio (per pixel; top-k; shift-only)",
            title="CH$_2$/CH$_3$ — WT vs KO",
            out_png=os.path.join(save_dir, "ratios_CH2_CH3_pooled.png"),
            log_y=False, ylim=ylim_ch2ch3,
            stats_text=format_stats_for_title_p(row_ch)
        )
        _strip_plot_pooled(
            ai_aii_WT_s, ai_aii_KO_s,
            ylabel="Amide I / Amide II (per pixel; top-k; shift-only)",
            title="Amide I / Amide II — WT vs KO",
            out_png=os.path.join(save_dir, "ratios_AmideI_AmideII_pooled.png"),
            log_y=USE_LOG_Y_FOR["AmideI/AmideII"], ylim=ylim_ai_aii,
            stats_text=format_stats_for_title_p(row_ai)
        )
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
        row_map = {}
        for name, WT_s, KO_s in rows_ext:
            row_map[name] = mwu_only_stats(name, WT_s, KO_s); stats_rows.append(row_map[name])

        def _ylim_for(key, WT_s, KO_s):
            return MANUAL_YLIMS[key] if USE_MANUAL_YLIMS else _auto_ylim(WT_s, KO_s, log=False)

        _strip_plot_pooled(
            r_1734_ai_WT_s, r_1734_ai_KO_s,
            ylabel="1734 / Amide I (per pixel; top-k; shift-only)",
            title=f"1734 / Amide I — WT vs KO @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=os.path.join(save_dir, "ratios_1734_over_AmideI_pooled.png"),
            log_y=False, ylim=_ylim_for("1734/AmideI", r_1734_ai_WT_s, r_1734_ai_KO_s),
            stats_text=format_stats_for_title_p(row_map["1734/AmideI"])
        )
        _strip_plot_pooled(
            r_1734_aii_WT_s, r_1734_aii_KO_s,
            ylabel="1734 / Amide II (per pixel; top-k; shift-only)",
            title=f"1734 / Amide II — WT vs KO @ {BAND_1734_CENTER:.1f}±{BAND_1734_HALFWIDTH} cm$^{{-1}}$",
            out_png=os.path.join(save_dir, "ratios_1734_over_AmideII_pooled.png"),
            log_y=False, ylim=_ylim_for("1734/AmideII", r_1734_aii_WT_s, r_1734_aii_KO_s),
            stats_text=format_stats_for_title_p(row_map["1734/AmideII"])
        )
        _strip_plot_pooled(
            ch2_ai_WT_s, ch2_ai_KO_s,
            ylabel="CH$_2$ / Amide I (per pixel; top-k; shift-only)",
            title="CH$_2$ / Amide I — WT vs KO",
            out_png=os.path.join(save_dir, "ratios_CH2_over_AmideI_pooled.png"),
            log_y=False, ylim=_ylim_for("CH2/AmideI", ch2_ai_WT_s, ch2_ai_KO_s),
            stats_text=format_stats_for_title_p(row_map["CH2/AmideI"])
        )
        _strip_plot_pooled(
            ch3_ai_WT_s, ch3_ai_KO_s,
            ylabel="CH$_3$ / Amide I (per pixel; top-k; shift-only)",
            title="CH$_3$ / Amide I — WT vs KO",
            out_png=os.path.join(save_dir, "ratios_CH3_over_AmideI_pooled.png"),
            log_y=False, ylim=_ylim_for("CH3/AmideI", ch3_ai_WT_s, ch3_ai_KO_s),
            stats_text=format_stats_for_title_p(row_map["CH3/AmideI"])
        )
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

        row_1734 = mwu_only_stats(f"1734@{center_band:.1f}", WT_band_s, KO_band_s); stats_rows.append(row_1734)

        ylim_band = MANUAL_YLIMS["1734"] if USE_MANUAL_YLIMS else _auto_ylim(WT_band_s, KO_band_s, log=False)
        _strip_plot_pooled(
            WT_band_s, KO_band_s,
            ylabel=f"Absorbance @ {center_band:.1f} cm$^{{-1}}$ (top-k; shift-only)",
            title=f"Lipid ester C=O — WT vs KO @ {center_band:.1f} cm$^{{-1}}$",
            out_png=os.path.join(save_dir, f"band_1734cm-1_pooled.png"),
            log_y=False, ylim=ylim_band,
            stats_text=format_stats_for_title_p(row_1734)
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

            row = mwu_only_stats(f"{label}@{center:.1f}", WT_s, KO_s); stats_rows.append(row)

            ylim_sb = MANUAL_YLIMS[label] if USE_MANUAL_YLIMS else _auto_ylim(WT_s, KO_s, log=USE_LOG_Y_FOR[label])
            _strip_plot_pooled(
                WT_s, KO_s,
                ylabel=f"{label.replace('amide','Amide ')} intensity (top-k; shift-only)",
                title=f"{label.replace('amide','Amide ').title()} — WT vs KO @ {center:.1f} cm$^{{-1}}$",
                out_png=os.path.join(save_dir, f"band_{label}_{int(center)}cm-1_pooled.png"),
                log_y=USE_LOG_Y_FOR[label], ylim=ylim_sb,
                stats_text=format_stats_for_title_p(row)
            )
    else:
        logger.warning("Skipping single-band plots (missing config).")

    # ---------- Save table (MWU p-values only) ----------
    if stats_rows:
        for r in stats_rows:
            logger.info("%s: n_wt=%d n_ko=%d  U=%.1f  p=%.3g  medΔ=%.4g",
                        r["metric"], r["n_wt"], r["n_ko"], r["U"], r["p_value"], r["median_diff"])
        save_stats_csv(stats_rows, out_prefix + "_WT_vs_KO_MWU_only.csv")

    # ---------- (Optional) CENTROID leaning (anonymous x-axis) ----------
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

    # ---------- ML: Linear SVM on 10x10 DG patches (ALL clusters) ----------
    try:
        run_patch_svm()
    except Exception as e:
        logger.exception("Patch SVM failed: %s", e)
