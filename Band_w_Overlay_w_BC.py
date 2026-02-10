# ======================================================================
# WT vs KO spectral analysis + overlays  (FULL UPDATED SCRIPT - OCTAVVS RMieSC FIX)
#
# Key updates vs last:
#  - FIX OCTAVVS rmiesc call: rmiesc(wn, app, ref, ...)
#  - auto-build ref spectrum as robust median across pixels (per sample)
#  - AsLS SparseEfficiencyWarning fix: convert matrix to CSC before spsolve
#
# Keeps:
#  - resample all samples to common axis (handles Z=421 etc.)
#  - analysis pixels capped to 2500/sample
#  - plotting uses tiny subsample fraction
#  - baseline (AsLS) + optional OCTAVVS RMieSC + floor shift + normalization
#  - ratios: base + all bands vs Amide I/II + PO2/(carb+lipid)
#  - beautiful plots: bigger points; median square; IQR black
#  - scatter + overlay violin in SAME figure; overlay has no ticks/labels/delta
#
# ======================================================================

from __future__ import annotations

import os, logging, time
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from scipy import stats, sparse
from scipy.sparse.linalg import spsolve
import colorsys, matplotlib.colors as mcolors
import importlib, pkgutil
import inspect, traceback

# ============================== PATHS / CONFIG ==============================
cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R\UMAP_clustering_test"
cube_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"
save_dir    = os.path.join(cube_dir, "WT_KO_bands_Over_3")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "WT_KO")

sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

# ---------------- Cluster selection (Right) ----------------
selected_clusters_raw = {
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

# ===================== Common axis (fixes Z mismatch) =====================
ASSUME_SAMPLE_MIN = 960.0
ASSUME_SAMPLE_MAX = 1800.0

WN_STD_MIN = 960.0
WN_STD_MAX = 1800.0
WN_STD_Z   = 426
WN_STD     = np.linspace(WN_STD_MIN, WN_STD_MAX, WN_STD_Z)

# ===================== Spectral window / extraction =====================
window = 30
TOPK_PER_BAND = 1
EPS = 1e-12

# ===================== Pixels used =====================
MAX_PIXELS_PER_SAMPLE_ANALYSIS = 2500   # analysis cap per sample
PLOT_SUBSAMPLE_FRAC = 0.003             # plotting only
MIN_PLOT_POINTS = 200

# ===================== Preprocessing toggles =====================
APPLY_ASLS_BASELINE_BANDS  = True
APPLY_ASLS_BASELINE_RATIOS = True
BASELINE_LAM = 1e6
BASELINE_P   = 0.01
BASELINE_NIT = 10

# Thickness/intensity normalization:
# "none", "l2", "ref_area_amideI", "ref_area_amideII"
NORM_MODE_BANDS  = "ref_area_amideI"
NORM_MODE_RATIOS = "ref_area_amideI"

SHIFT_MIN_FLOOR_BANDS  = 0.2
SHIFT_MIN_FLOOR_RATIOS = 0.5

# ===================== Mie/RMie scattering correction =====================
SCATTER_BACKEND = "octavvs"   # "none" to disable
OCTAVVS_VERBOSE_SCAN = True

# RMieSC parameters (tunable)
RMIESC_ITERATIONS = 10
RMIESC_VERBOSE = False

# How to pick RMieSC reference spectrum:
# "median" is robust; "mean" is faster but less robust.
RMIESC_REF_MODE = "median"

# ===================== Ratio guards =====================
APPLY_RATIO_HARD_CAP = True
RATIO_HARD_CAP       = 10.0

DENOM_MIN_PCT   = 3.0
DENOM_ABS_FLOOR = 1e-2
NUM_MIN_PCT     = 15.0
NUM_ABS_FLOOR   = 1e-4

# ===================== Plot style =====================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WT_KO")

BASE_WT = "#2ca02c"
BASE_KO = "#d62728"

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11
FS = {"base": BASE_FONTSIZE, "axes": BASE_FONTSIZE+1, "title": BASE_FONTSIZE+3, "tick": BASE_FONTSIZE, "annot": 10}

DOT_SIZE = 22
DOT_ALPHA = 0.45
DOT_EDGE_ALPHA = 0.55
DOT_EDGE_LW = 0.7

MEDIAN_MARKER_SIZE = 130
IQR_LW = 2.0

VIOLIN_WT_COLOR = (0.3, 0.7, 0.3, 0.22)
VIOLIN_KO_COLOR = (0.9, 0.3, 0.3, 0.22)

JITTER_SCALE = 0.06

ANNOTATE_DELTA_ON_SCATTER = True
SAVE_PDF_ALSO = True

# ===================== Manual y-lims =====================
USE_MANUAL_YLIMS = True
MANUAL_YLIMS = {
    "CH2/CH3": None,
    "AmideI/AmideII": None,
}
USE_LOG_Y_FOR = {
    "CH2/CH3": False,
    "AmideI/AmideII": False,
}

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

# ========================== UTILITIES ==========================
@contextmanager
def LogTimer(msg):
    t0 = time.time(); logger.info(msg + " ...")
    try:
        yield
    finally:
        logger.info("%s done in %.2fs", msg, time.time() - t0)

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
        "axes.linewidth": 1.0,
        "grid.color": "#9aa0a6",
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "mathtext.default": "regular",
    })

def _safe_name(s: str) -> str:
    out = str(s).replace("/", "_over_").replace("\\", "_")
    for b in ['"', "'", ":", "*", "?", "<", ">", "|"]:
        out = out.replace(b, "")
    return out.replace(" ", "")

def _auto_ylim(a, b, log=False, pad_frac=0.1):
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

def _mask_for_ylim(arr, ylim):
    if ylim is None:
        return arr
    lo, hi = ylim
    return arr[(arr >= lo) & (arr <= hi) & np.isfinite(arr)]

def _subsample_for_plot(arr, rng, frac=PLOT_SUBSAMPLE_FRAC, min_pts=MIN_PLOT_POINTS):
    arr = np.asarray(arr)
    n = len(arr)
    if n == 0:
        return arr
    k = int(max(min_pts, round(n * frac)))
    k = min(k, n)
    if k == n:
        return arr
    idx = rng.choice(n, size=k, replace=False)
    return arr[idx]

def _group_sample_colors(wt_keys, ko_keys):
    def ramp(hex_color, n, v_min=0.45, v_max=0.95, s=0.85):
        r,g,b = mcolors.to_rgb(hex_color)
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        vs = np.linspace(v_min, v_max, max(n,1))
        return [colorsys.hsv_to_rgb(h, s, v) for v in vs]
    wt_cols = ramp(BASE_WT, len(wt_keys))
    ko_cols = ramp(BASE_KO, len(ko_keys))
    return {k: wt_cols[i] for i,k in enumerate(wt_keys)}, {k: ko_cols[i] for i,k in enumerate(ko_keys)}

# ========================== IO HELPERS ================================
def _load_cluster_labels_indices(sample):
    f = os.path.join(cluster_dir, f"{sample}_umap_kmeans.npz")
    if not os.path.exists(f):
        logger.warning("Missing cluster file for %s", sample); return None
    z = np.load(f)
    return z["cluster_labels"], z["pixel_indices"]

def _load_cube(sample):
    f = os.path.join(cube_dir, f"masked_cube_{sample}.npz")
    if not os.path.exists(f):
        logger.warning("Missing RAW cube for %s", sample); return None
    return np.load(f)["data"]  # (H,W,Z)

def _selected_mask_from_clusters(H, W, labels, indices, sel_lst):
    if indices.ndim == 2 and indices.shape[1] == 2:
        rows = indices[:, 0]; cols = indices[:, 1]
        use = np.isin(labels, sel_lst)
        r = rows[use]; c = cols[use]
        m = np.zeros((H, W), dtype=bool); m[r, c] = True
        return m
    else:
        flat_idx = indices
        use = np.isin(labels, sel_lst)
        flat_use = flat_idx[use]
        m = np.zeros((H*W,), dtype=bool); m[flat_use] = True
        return m.reshape(H, W)

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

selected_clusters = _clean_selected_clusters(selected_clusters_raw, n_clusters=8)

# ========================== RESAMPLING (fix Z mismatch) =====================
def _resample_matrix_to_std(X, Z_src):
    wns_src = np.linspace(ASSUME_SAMPLE_MIN, ASSUME_SAMPLE_MAX, int(Z_src))
    return np.vstack([np.interp(WN_STD, wns_src, X[i]) for i in range(X.shape[0])])

# ========================== BASELINE CORRECTION (AsLS) =================
def baseline_als(y, lam=1e6, p=0.01, niter=10):
    y = np.asarray(y, float)
    L = y.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = (W + lam * (D @ D.T)).tocsc()  # FIX SparseEfficiencyWarning
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def apply_baseline_matrix(X, lam=1e6, p=0.01, niter=10):
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        b = baseline_als(X[i], lam=lam, p=p, niter=niter)
        out[i] = X[i] - b
    return out

# ========================== NORMALIZATION ==============================
def _l2_normalize_matrix(X):
    norms = np.sqrt(np.sum(X*X, axis=1, keepdims=True)) + EPS
    return X / norms

def _ref_area_matrix(X, wns, center, halfwidth):
    m = (wns >= (center - halfwidth)) & (wns <= (center + halfwidth))
    if not np.any(m):
        return np.ones((X.shape[0], 1), float)
    a = np.trapz(X[:, m], wns[m], axis=1)  # keepdims removed (correct)
    a = a.reshape(-1, 1)
    a = np.where(np.isfinite(a) & (a > EPS), a, 1.0)
    return a

def _normalize_matrix(X, wns, mode: str):
    mode = (mode or "none").lower()
    if mode == "none":
        return X
    if mode == "l2":
        return _l2_normalize_matrix(X)
    if mode == "ref_area_amidei":
        a = _ref_area_matrix(X, wns, BAND_AMIDE_I, window)
        return X / (a + EPS)
    if mode == "ref_area_amideii":
        a = _ref_area_matrix(X, wns, BAND_AMIDE_II, window)
        return X / (a + EPS)
    raise ValueError(f"Unknown norm mode: {mode}")

def _shift_min_floor_matrix(X, floor):
    mins = np.min(X, axis=1, keepdims=True)
    add  = np.clip(floor - mins, 0.0, None)
    return X + add

# ========================== OCTAVVS AUTODISCOVERY + FIXED RMieSC CALL =================
_OCTAVVS_RMIESC = None
_OCTAVVS_FAIL_COUNT = 0
_OCTAVVS_MAX_FAILS = 2
_OCTAVVS_LOGGED_FIRST_ERROR = False

def _try_import(module_path):
    try:
        return importlib.import_module(module_path)
    except Exception:
        return None

def _find_octavvs_rmiesc(verbose=True):
    # Your scan shows this exists:
    # octavvs.algorithms.correction.rmiesc(wn, app, ref, ...)
    mod = _try_import("octavvs.algorithms.correction")
    if mod is not None and hasattr(mod, "rmiesc") and callable(getattr(mod, "rmiesc")):
        return getattr(mod, "rmiesc"), "octavvs.algorithms.correction.rmiesc"

    # fallback scan (rarely needed)
    try:
        import octavvs  # noqa
    except Exception as e:
        if verbose:
            logger.warning("OCTAVVS not importable: %s", e)
        return None, None

    found = []
    import octavvs
    for m in pkgutil.walk_packages(octavvs.__path__, prefix="octavvs."):
        mod2 = _try_import(m.name)
        if mod2 is None:
            continue
        if hasattr(mod2, "rmiesc") and callable(getattr(mod2, "rmiesc")):
            found.append((getattr(mod2, "rmiesc"), m.name + ".rmiesc"))

    if not found:
        return None, None
    return found[0][0], found[0][1]

def _build_rmiesc_ref(app, mode="median"):
    # app: (N,Z)
    if app.size == 0:
        return None
    mode = (mode or "median").lower()
    if mode == "mean":
        ref = np.nanmean(app, axis=0)
    else:
        ref = np.nanmedian(app, axis=0)
    ref = np.asarray(ref, dtype=np.float32)
    # ref must be finite
    if not np.all(np.isfinite(ref)):
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    return ref

def _scatter_correct_matrix_rmiesc(X, wns):
    """
    Correct call: rmiesc(wn, app, ref, iterations=..., verbose=...)
    - X: (N,Z)
    - wns: (Z,)
    Returns corrected (N,Z) or original on failure.
    """
    global _OCTAVVS_RMIESC, _OCTAVVS_FAIL_COUNT, _OCTAVVS_LOGGED_FIRST_ERROR

    if (SCATTER_BACKEND or "none").lower() == "none":
        return X
    if (SCATTER_BACKEND or "").lower() != "octavvs":
        logger.warning("Unknown SCATTER_BACKEND=%s -> no scatter correction", SCATTER_BACKEND)
        return X
    if _OCTAVVS_FAIL_COUNT >= _OCTAVVS_MAX_FAILS:
        return X

    if _OCTAVVS_RMIESC is None:
        try:
            import octavvs  # noqa
        except Exception as e:
            logger.warning("OCTAVVS not installed/importable -> skipping Mie correction (%s)", e)
            _OCTAVVS_FAIL_COUNT = _OCTAVVS_MAX_FAILS
            return X

        fn, nm = _find_octavvs_rmiesc(verbose=True)
        if fn is None:
            logger.warning("Could not find OCTAVVS rmiesc -> skipping.")
            _OCTAVVS_FAIL_COUNT = _OCTAVVS_MAX_FAILS
            return X

        _OCTAVVS_RMIESC = fn
        logger.info("Using OCTAVVS rmiesc: %s", nm)
        try:
            logger.info("rmiesc signature: %s", str(inspect.signature(_OCTAVVS_RMIESC)))
        except Exception:
            pass

    app = np.asarray(X, dtype=np.float32, order="C")
    wn  = np.asarray(wns, dtype=np.float32)

    ref = _build_rmiesc_ref(app, mode=RMIESC_REF_MODE)
    if ref is None:
        return X

    try:
        # rmiesc returns corrected apparent absorbance, often same shape as app
        out = _OCTAVVS_RMIESC(wn, app, ref, iterations=RMIESC_ITERATIONS, verbose=RMIESC_VERBOSE)
        out = np.asarray(out)
        if out.shape == app.shape:
            return out.astype(np.float64, copy=False)
        # sometimes returns (app, model) tuple
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            out0 = np.asarray(out[0])
            if out0.shape == app.shape:
                return out0.astype(np.float64, copy=False)
    except Exception as e:
        if not _OCTAVVS_LOGGED_FIRST_ERROR:
            _OCTAVVS_LOGGED_FIRST_ERROR = True
            logger.warning("OCTAVVS rmiesc failed (first error) -> will fallback after a few fails.")
            logger.warning("Exception: %s: %s", type(e).__name__, str(e))
            logger.warning("Traceback:\n%s", traceback.format_exc())

    _OCTAVVS_FAIL_COUNT += 1
    logger.warning("OCTAVVS rmiesc failed at runtime -> skipping for this call. (fail %d/%d)",
                   _OCTAVVS_FAIL_COUNT, _OCTAVVS_MAX_FAILS)
    return X

# ========================== BAND EXTRACTION (matrix) ===================
def _band_value_topk_mean_matrix(X, wns, center, halfwidth, k=TOPK_PER_BAND):
    if halfwidth is None or halfwidth <= 0:
        halfwidth = 10
    m = (wns >= (center - halfwidth)) & (wns <= (center + halfwidth))
    if not np.any(m):
        return np.full((X.shape[0],), np.nan, dtype=float)
    slab = X[:, m]
    nwin = slab.shape[1]
    kk = int(max(1, min(int(k), nwin)))
    idx = np.argpartition(slab, -kk, axis=1)[:, -kk:]
    topk = np.take_along_axis(slab, idx, axis=1)
    return np.nanmean(topk, axis=1)

# ========================== Guards for ratios ==========================
def _denom_guard(den, min_pct=DENOM_MIN_PCT, abs_floor=DENOM_ABS_FLOOR):
    d = den[np.isfinite(den)]
    if d.size == 0:
        return np.full_like(den, False, dtype=bool), abs_floor
    thr = max(np.nanpercentile(d, min_pct), abs_floor)
    return (den > thr), thr

def _num_guard(num, min_pct=NUM_MIN_PCT, abs_floor=NUM_ABS_FLOOR):
    d = num[np.isfinite(num)]
    if d.size == 0:
        return np.full_like(num, False, dtype=bool), abs_floor
    thr = max(np.nanpercentile(d, min_pct), abs_floor)
    return (num > thr), thr

# ========================== Stats: Cliff's delta =======================
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

def format_delta(delta):
    if not np.isfinite(delta):
        return ""
    return f"δ={delta:+.2f}"

# ========================== RATIO DEFINITIONS ==========================
def build_all_band_vs_amide_ratios():
    ratios = {}
    for bkey, (bc, bw) in BAND_LIBRARY.items():
        if bkey in ("amideI", "amideII"):
            continue
        ratios[f"{bkey}/AmideI"]  = {"num": (bc, bw), "den": BAND_LIBRARY["amideI"]}
        ratios[f"{bkey}/AmideII"] = {"num": (bc, bw), "den": BAND_LIBRARY["amideII"]}
    return ratios

def build_po2_extra_ratios():
    out = {}
    po2_keys   = ["po2_1080", "po2_1235"]
    carb_keys  = ["carb_1030", "carb_1155"]
    lipid_keys = ["ch2", "ch3", "1734"]
    for p in po2_keys:
        for c in carb_keys:
            out[f"{p}/{c}"] = {"num": BAND_LIBRARY[p], "den": BAND_LIBRARY[c]}
        for l in lipid_keys:
            out[f"{p}/{l}"] = {"num": BAND_LIBRARY[p], "den": BAND_LIBRARY[l]}
    return out

BASE_RATIOS = {
    "CH2/CH3":        {"num": BAND_LIBRARY["ch2"],    "den": BAND_LIBRARY["ch3"]},
    "AmideI/AmideII": {"num": BAND_LIBRARY["amideI"], "den": BAND_LIBRARY["amideII"]},
}

ALL_RATIOS = {}
ALL_RATIOS.update(BASE_RATIOS)
ALL_RATIOS.update(build_all_band_vs_amide_ratios())
ALL_RATIOS.update(build_po2_extra_ratios())

# ========================== Core: collect spectra ======================
def _collect_selected_pixel_spectra_std(sample):
    out = _load_cluster_labels_indices(sample)
    if out is None:
        return None, None
    labels, indices = out

    cube = _load_cube(sample)
    if cube is None:
        return None, None

    H, W, Z_src = cube.shape
    sel = selected_clusters.get(sample, [])
    if not sel:
        logger.warning("No selected clusters for %s", sample)
        return None, None

    mask2d = _selected_mask_from_clusters(H, W, labels, indices, sel)
    if not np.any(mask2d):
        logger.warning("No pixels in selected clusters for %s", sample)
        return None, None

    rr, cc = np.where(mask2d)
    X = cube[rr, cc, :].astype(np.float64, copy=False)  # (N, Z_src)

    # cap analysis pixels per sample
    if X.shape[0] > MAX_PIXELS_PER_SAMPLE_ANALYSIS:
        rng = np.random.default_rng(abs(hash(sample)) % (2**32))
        take = rng.choice(X.shape[0], size=MAX_PIXELS_PER_SAMPLE_ANALYSIS, replace=False)
        X = X[take]

    # resample to common axis
    if X.shape[1] != WN_STD_Z:
        X = _resample_matrix_to_std(X, Z_src=Z_src)

    return X, WN_STD

def _preprocess_matrix(X, wns, *, do_baseline: bool, floor: float, norm_mode: str, do_scatter: bool):
    Y = np.asarray(X, float)

    # 1) baseline (optional)
    if do_baseline:
        Y = apply_baseline_matrix(Y, lam=BASELINE_LAM, p=BASELINE_P, niter=BASELINE_NIT)

    # 2) RMieSC (optional)
    if do_scatter and (SCATTER_BACKEND.lower() != "none"):
        Y = _scatter_correct_matrix_rmiesc(Y, wns)

    # 3) shift floor (stability)
    if floor is not None:
        Y = _shift_min_floor_matrix(Y, floor)

    # 4) normalize for thickness/intensity
    Y = _normalize_matrix(Y, wns, norm_mode)
    return Y

# ========================== Collect ratios per sample ==================
def _collect_ratios_for_sample(sample, ratio_cfg_dict):
    Xraw, wns = _collect_selected_pixel_spectra_std(sample)
    if Xraw is None:
        return None

    X = _preprocess_matrix(
        Xraw, wns,
        do_baseline=APPLY_ASLS_BASELINE_RATIOS,
        floor=SHIFT_MIN_FLOOR_RATIOS,
        norm_mode=NORM_MODE_RATIOS,
        do_scatter=True
    )

    band_cache = {}
    for rkey, cfg in ratio_cfg_dict.items():
        n_c, n_hw = cfg["num"]
        d_c, d_hw = cfg["den"]
        band_cache[("num", rkey)] = _band_value_topk_mean_matrix(X, wns, float(n_c), float(n_hw), k=TOPK_PER_BAND)
        band_cache[("den", rkey)] = _band_value_topk_mean_matrix(X, wns, float(d_c), float(d_hw), k=TOPK_PER_BAND)

    out = {}
    for rkey in ratio_cfg_dict.keys():
        num = band_cache[("num", rkey)]
        den = band_cache[("den", rkey)]
        ok_num, _ = _num_guard(num)
        ok_den, _ = _denom_guard(den)
        ok = ok_num & ok_den & np.isfinite(num) & np.isfinite(den)
        rr = np.full_like(num, np.nan, dtype=float)
        rr[ok] = num[ok] / (den[ok] + EPS)
        if APPLY_RATIO_HARD_CAP:
            rr = np.clip(rr, 0, RATIO_HARD_CAP)
        rr = rr[np.isfinite(rr) & (rr > 0)]
        out[rkey] = rr

    return out

# ========================== Collect single-band intensities ============
def _collect_band_for_sample(sample, band_key):
    Xraw, wns = _collect_selected_pixel_spectra_std(sample)
    if Xraw is None:
        return None

    X = _preprocess_matrix(
        Xraw, wns,
        do_baseline=APPLY_ASLS_BASELINE_BANDS,
        floor=SHIFT_MIN_FLOOR_BANDS,
        norm_mode=NORM_MODE_BANDS,
        do_scatter=True
    )

    center, halfw = BAND_LIBRARY[band_key]
    v = _band_value_topk_mean_matrix(X, wns, float(center), float(halfw), k=TOPK_PER_BAND)
    v = v[np.isfinite(v) & (v > 0)]
    return v

# ========================== Plotting: scatter + overlay violin =========
def _annotate(ax, text):
    if not text:
        return
    ax.text(
        0.02, 0.98, text,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=FS["annot"], color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, boxstyle="round,pad=0.25")
    )

def _draw_median_iqr(ax, xvals, xpos, edgecolor):
    if xvals.size == 0:
        return
    med = float(np.median(xvals))
    q1, q3 = np.quantile(xvals, [0.25, 0.75])

    ax.scatter([xpos], [med],
               s=MEDIAN_MARKER_SIZE,
               marker="s",
               c=["white"],
               edgecolors=[edgecolor],
               linewidths=1.6,
               zorder=5)

    ax.vlines(xpos, q1, q3, colors="black", linewidth=IQR_LW, zorder=4)
    ax.hlines([q1, q3], xpos-0.04, xpos+0.04, colors="black", linewidth=IQR_LW, zorder=4)

def plot_scatter_plus_overlay(WT_dict, KO_dict, ylabel, out_png, log_y=False, ylim=None, delta_text=""):
    _apply_pub_style()

    WT_all = np.concatenate([v for v in WT_dict.values() if len(v) > 0]) if WT_dict else np.array([])
    KO_all = np.concatenate([v for v in KO_dict.values() if len(v) > 0]) if KO_dict else np.array([])

    if WT_all.size == 0 and KO_all.size == 0:
        logger.warning("Nothing to plot for %s", out_png)
        return

    if ylim is not None:
        WT_all = _mask_for_ylim(WT_all, ylim)
        KO_all = _mask_for_ylim(KO_all, ylim)

    fig, (ax_sc, ax_ov) = plt.subplots(
        1, 2, figsize=(11.5, 5.2),
        gridspec_kw={"width_ratios": [1.0, 0.55], "wspace": 0.20}
    )
    rng = np.random.default_rng(0)

    wt_keys = list(WT_dict.keys()); ko_keys = list(KO_dict.keys())
    wt_col_map, ko_col_map = _group_sample_colors(wt_keys, ko_keys)

    for s, arr in WT_dict.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        if ylim is not None:
            arr = _mask_for_ylim(arr, ylim)
        if arr.size == 0:
            continue
        arrp = _subsample_for_plot(arr, rng=rng)
        xs = 0 + rng.normal(0, JITTER_SCALE, size=arrp.size)
        ax_sc.scatter(xs, arrp, s=DOT_SIZE, alpha=DOT_ALPHA,
                      c=[wt_col_map.get(s)],
                      edgecolors=[(0,0,0, DOT_EDGE_ALPHA)],
                      linewidths=DOT_EDGE_LW,
                      rasterized=True)

    for s, arr in KO_dict.items():
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        if ylim is not None:
            arr = _mask_for_ylim(arr, ylim)
        if arr.size == 0:
            continue
        arrp = _subsample_for_plot(arr, rng=rng)
        xs = 1 + rng.normal(0, JITTER_SCALE, size=arrp.size)
        ax_sc.scatter(xs, arrp, s=DOT_SIZE, alpha=DOT_ALPHA,
                      c=[ko_col_map.get(s)],
                      edgecolors=[(0,0,0, DOT_EDGE_ALPHA)],
                      linewidths=DOT_EDGE_LW,
                      rasterized=True)

    if WT_all.size:
        _draw_median_iqr(ax_sc, WT_all, 0, edgecolor=BASE_WT)
    if KO_all.size:
        _draw_median_iqr(ax_sc, KO_all, 1, edgecolor=BASE_KO)

    ax_sc.set_xticks([0, 1], ["WT", "KO"])
    ax_sc.set_ylabel(ylabel)

    if log_y:
        ax_sc.set_yscale("log")
        if ylim is not None:
            ax_sc.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax_sc.set_ylim(*ylim)

    ax_sc.yaxis.grid(True)
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)

    if ANNOTATE_DELTA_ON_SCATTER:
        _annotate(ax_sc, delta_text)

    # Overlay violin: no labels/ticks/delta/medians/IQR
    ax_ov.set_xticks([])
    ax_ov.set_yticks([])
    for sp in ["top", "right", "left", "bottom"]:
        ax_ov.spines[sp].set_visible(False)

    if KO_all.size:
        parts_ko = ax_ov.violinplot([KO_all], positions=[0], widths=[0.95],
                                    showmeans=False, showmedians=False, showextrema=False)
        bko = parts_ko["bodies"][0]
        bko.set_facecolor(VIOLIN_KO_COLOR)
        bko.set_edgecolor("none")
        bko.set_zorder(1)

    if WT_all.size:
        parts_wt = ax_ov.violinplot([WT_all], positions=[0], widths=[0.95],
                                    showmeans=False, showmedians=False, showextrema=False)
        bwt = parts_wt["bodies"][0]
        bwt.set_facecolor(VIOLIN_WT_COLOR)
        bwt.set_edgecolor("none")
        bwt.set_zorder(2)

    if log_y:
        ax_ov.set_yscale("log")
        if ylim is not None:
            ax_ov.set_ylim(max(ylim[0], 1e-6), ylim[1])
    elif ylim is not None:
        ax_ov.set_ylim(*ylim)
    else:
        ax_ov.set_ylim(*ax_sc.get_ylim())

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", dpi=PUB_DPI)
    if SAVE_PDF_ALSO:
        fig.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight", dpi=PUB_DPI)
    plt.close(fig)

# ========================== Stats CSV ==========================
def save_stats_csv(rows, path):
    headers = ["metric","n_wt","n_ko","median_wt","median_ko","cliffs_delta"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")
    logger.info("Saved stats table: %s", path)

# =============================== MAIN =================================
if __name__ == "__main__":
    logger.info("Axis: assume samples %.1f..%.1f -> resample to STD %.1f..%.1f (Z=%d)",
                ASSUME_SAMPLE_MIN, ASSUME_SAMPLE_MAX, WN_STD_MIN, WN_STD_MAX, WN_STD_Z)
    logger.info("SCATTER_BACKEND=%s | Baseline bands=%s ratios=%s | Norm bands=%s ratios=%s",
                SCATTER_BACKEND, APPLY_ASLS_BASELINE_BANDS, APPLY_ASLS_BASELINE_RATIOS,
                NORM_MODE_BANDS, NORM_MODE_RATIOS)
    logger.info("Analysis pixels/sample=%d | Plot subsample frac=%.4f | Total ratios=%d",
                MAX_PIXELS_PER_SAMPLE_ANALYSIS, PLOT_SUBSAMPLE_FRAC, len(ALL_RATIOS))

    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    stats_rows = []

    # ---------- RATIOS ----------
    with LogTimer("Collect ratios per sample"):
        WT_ratio_dicts = {r: {} for r in ALL_RATIOS.keys()}
        KO_ratio_dicts = {r: {} for r in ALL_RATIOS.keys()}
        WT_concat = {r: [] for r in ALL_RATIOS.keys()}
        KO_concat = {r: [] for r in ALL_RATIOS.keys()}

        for n in sample_names:
            g = sample_to_group.get(n)
            if g not in ("WT", "KO"):
                continue
            per = _collect_ratios_for_sample(n, ALL_RATIOS)
            if per is None:
                continue
            for rkey, arr in per.items():
                if arr is None or len(arr) == 0:
                    continue
                if g == "WT":
                    WT_ratio_dicts[rkey][n] = arr
                    WT_concat[rkey].append(arr)
                else:
                    KO_ratio_dicts[rkey][n] = arr
                    KO_concat[rkey].append(arr)

    with LogTimer("Plot ratios (scatter + overlay violin)"):
        for rkey in ALL_RATIOS.keys():
            WT_all = np.concatenate(WT_concat[rkey]) if WT_concat[rkey] else np.array([])
            KO_all = np.concatenate(KO_concat[rkey]) if KO_concat[rkey] else np.array([])

            delta = cliffs_delta(WT_all, KO_all)
            stats_rows.append({
                "metric": rkey,
                "n_wt": len(WT_all),
                "n_ko": len(KO_all),
                "median_wt": float(np.median(WT_all)) if len(WT_all) else np.nan,
                "median_ko": float(np.median(KO_all)) if len(KO_all) else np.nan,
                "cliffs_delta": float(delta) if np.isfinite(delta) else np.nan
            })

            log_y = USE_LOG_Y_FOR.get(rkey, False)
            ylim = MANUAL_YLIMS.get(rkey, None) if USE_MANUAL_YLIMS else _auto_ylim(WT_all, KO_all, log=log_y)
            ylabel = rkey.replace("AmideI", "Amide I").replace("AmideII", "Amide II")

            plot_scatter_plus_overlay(
                WT_ratio_dicts[rkey], KO_ratio_dicts[rkey],
                ylabel=ylabel,
                out_png=os.path.join(save_dir, f"ratio_{_safe_name(rkey)}_scatter_plus_overlay.png"),
                log_y=log_y, ylim=ylim,
                delta_text=format_delta(delta)
            )

    # ---------- SINGLE-BAND INTENSITIES ----------
    with LogTimer("Single-band intensity plots"):
        for bkey in BAND_LIBRARY.keys():
            WT_d, KO_d = {}, {}
            WT_all_list, KO_all_list = [], []

            for n in groups["WT"]:
                v = _collect_band_for_sample(n, bkey)
                if v is not None and len(v):
                    WT_d[n] = v
                    WT_all_list.append(v)

            for n in groups["KO"]:
                v = _collect_band_for_sample(n, bkey)
                if v is not None and len(v):
                    KO_d[n] = v
                    KO_all_list.append(v)

            WT_all = np.concatenate(WT_all_list) if WT_all_list else np.array([])
            KO_all = np.concatenate(KO_all_list) if KO_all_list else np.array([])

            delta = cliffs_delta(WT_all, KO_all)
            center, _ = BAND_LIBRARY[bkey]
            stats_rows.append({
                "metric": f"{bkey}@{center:.1f}",
                "n_wt": len(WT_all),
                "n_ko": len(KO_all),
                "median_wt": float(np.median(WT_all)) if len(WT_all) else np.nan,
                "median_ko": float(np.median(KO_all)) if len(KO_all) else np.nan,
                "cliffs_delta": float(delta) if np.isfinite(delta) else np.nan
            })

            log_y = USE_LOG_Y_FOR.get(bkey, False)
            ylim = MANUAL_YLIMS.get(bkey, None) if USE_MANUAL_YLIMS else _auto_ylim(WT_all, KO_all, log=log_y)

            plot_scatter_plus_overlay(
                WT_d, KO_d,
                ylabel=BAND_LABELS.get(bkey, f"Band {bkey}"),
                out_png=os.path.join(save_dir, f"band_{bkey}_{int(center)}cm-1_scatter_plus_overlay.png"),
                log_y=log_y, ylim=ylim,
                delta_text=format_delta(delta)
            )

    # ---------- SAVE STATS ----------
    if stats_rows:
        save_stats_csv(stats_rows, out_prefix + "_WT_vs_KO_stats.csv")

    logger.info("Done.")
