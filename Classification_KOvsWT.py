# ======================================================================
# WT vs KO FTIR classification (LEAKAGE-REDUCED, SPLIT-FIRST LOOCV)
# FULL UPDATED SCRIPT WITH:
#   - OPTIONAL background subtraction
#   - OPTIONAL RMieSC scattering correction (OCTAVVS)
#   - NO baseline correction (AsLS)  [disabled by design here]
#   - NO normalization               [disabled by design here]
#   - SPLIT-FIRST LOOCV: feature thresholds/processing learned inside fold
#
# WHY THIS VERSION:
#   - Reduces data leakage, especially for band-ratio guards/thresholds
#   - Keeps your publication-style plots
#   - Fixes ugly probability plot label overlap
#
# ======================================================================

from __future__ import annotations

import os
import logging
import time
import traceback
import importlib
import pkgutil
import inspect
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys

from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# ============================== PATHS / CONFIG ==============================
cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R\UMAP_clustering_test"
cube_dir    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"

save_dir = os.path.join(cube_dir, "WT_KO_ML_SPLITFIRST_LOOCV_BG_SCATTER_OPTIONAL_Ratios")
os.makedirs(save_dir, exist_ok=True)

# >>> BG DIR (EDIT IF NEEDED) <<<
bg_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\Area_Right_B\BG_First300ROI_MeanSpectrum"

# ============================== Samples / Groups ============================
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
    "T19":[0,1,2,3,4.5,6,7],  # cleaned to ints
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

# ===================== Feature extraction settings =======================
# "bands_only", "band_ratios", "median_spectrum"
FEATURE_MODE = "band_ratios"

# Pixel caps
MAX_PIXELS_PER_SAMPLE_ANALYSIS = 2500
MAX_PIXELS_FOR_RAWPLOT = 600

# band peak extraction
TOPK_PER_BAND = 1
window = 30
EPS = 1e-12

# ratio guard (used for FEATURE_MODE="band_ratios")
# IMPORTANT: in this script thresholds are computed from TRAINING FOLDS ONLY
APPLY_RATIO_HARD_CAP = True
RATIO_HARD_CAP       = 10.0
DENOM_MIN_PCT   = 3.0
DENOM_ABS_FLOOR = 1e-2
NUM_MIN_PCT     = 15.0
NUM_ABS_FLOOR   = 1e-4

# ===================== Background subtraction (optional) ===================
APPLY_BG_SUBTRACTION = True
SKIP_BG_IF_MISSING   = True

BG_PATTERNS = [
    "bg_first300_mean_spectrum_{sid}.npz",
    "bg_first30_mean_spectrum_{sid}.npz",
    "bg_mean_spectrum_{sid}.npz",
    "bg_{sid}.npz",
]

# ===================== Scattering handling (optional RMieSC) ==============
# "none" or "octavvs"
SCATTER_BACKEND = "octavvs"   # <-- set to "octavvs" to enable RMieSC if OCTAVVS installed
APPLY_SCATTER_CORRECTION = True  # master switch

RMIESC_ITERATIONS = 10
RMIESC_VERBOSE = False
RMIESC_REF_MODE = "median"  # "median" or "mean"
OCTAVVS_VERBOSE_SCAN = True

# ===================== ML model config ====================================
MODEL_NAME = "logreg"   # "logreg" or "linear_svm"
USE_LOOCV = True

# ===================== Plot style =========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WT_KO_ML")

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 11
FS = {"base": BASE_FONTSIZE, "axes": BASE_FONTSIZE+1, "title": BASE_FONTSIZE+2, "tick": BASE_FONTSIZE, "annot": 10}

BASE_WT = "#2ca02c"
BASE_KO = "#d62728"

# ===================== Bands =============================================
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

BASE_RATIOS = {
    "CH2/CH3":          {"num": BAND_LIBRARY["ch2"],      "den": BAND_LIBRARY["ch3"]},
    "AmideI/AmideII":   {"num": BAND_LIBRARY["amideI"],   "den": BAND_LIBRARY["amideII"]},
    "PO2(1080)/AmideI": {"num": BAND_LIBRARY["po2_1080"], "den": BAND_LIBRARY["amideI"]},
    "PO2(1235)/AmideI": {"num": BAND_LIBRARY["po2_1235"], "den": BAND_LIBRARY["amideI"]},
    "PO2(1080)/CH2":    {"num": BAND_LIBRARY["po2_1080"], "den": BAND_LIBRARY["ch2"]},
    "PO2(1235)/CH2":    {"num": BAND_LIBRARY["po2_1235"], "den": BAND_LIBRARY["ch2"]},
    "Carb1030/AmideI":  {"num": BAND_LIBRARY["carb_1030"],"den": BAND_LIBRARY["amideI"]},
    "Carb1155/AmideI":  {"num": BAND_LIBRARY["carb_1155"],"den": BAND_LIBRARY["amideI"]},
}

# ========================== UTILITIES ======================================
@contextmanager
def LogTimer(msg):
    t0 = time.time()
    logger.info(msg + " ...")
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
        "font.size": FS["base"],
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

def _group_sample_colors(wt_keys, ko_keys):
    def ramp(hex_color, n, v_min=0.45, v_max=0.95, s=0.85):
        r, g, b = mcolors.to_rgb(hex_color)
        h, _, _ = colorsys.rgb_to_hsv(r, g, b)
        vs = np.linspace(v_min, v_max, max(n, 1))
        return [colorsys.hsv_to_rgb(h, s, v) for v in vs]
    wt_cols = ramp(BASE_WT, len(wt_keys))
    ko_cols = ramp(BASE_KO, len(ko_keys))
    return {k: wt_cols[i] for i, k in enumerate(wt_keys)}, {k: ko_cols[i] for i, k in enumerate(ko_keys)}

# ========================== IO HELPERS =====================================
def _load_cluster_labels_indices(sample):
    f = os.path.join(cluster_dir, f"{sample}_umap_kmeans.npz")
    if not os.path.exists(f):
        logger.warning("Missing cluster file for %s", sample)
        return None
    z = np.load(f)
    return z["cluster_labels"], z["pixel_indices"]

def _load_cube(sample):
    f = os.path.join(cube_dir, f"masked_cube_{sample}.npz")
    if not os.path.exists(f):
        logger.warning("Missing RAW cube for %s", sample)
        return None
    return np.load(f)["data"]  # (H,W,Z)

def _selected_mask_from_clusters(H, W, labels, indices, sel_lst):
    if indices.ndim == 2 and indices.shape[1] == 2:
        rows = indices[:, 0]
        cols = indices[:, 1]
        use = np.isin(labels, sel_lst)
        r = rows[use]
        c = cols[use]
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

# ========================== BG SUBTRACTION HELPERS ==========================
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
    raise ValueError(f"{sample}: BG file has no 1D numeric vector. keys={z.files}")

def _align_bg_to_Z(bg, Z_src):
    bg = np.asarray(bg, dtype=np.float64).ravel()
    if bg.size == Z_src:
        return bg

    # special rule from your earlier script
    if bg.size == 426 and Z_src == 421:
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
        out[L:] = out[L - 1]
    return out

def _bg_subtract_cube_spectra(X, Z_src, sample):
    if not APPLY_BG_SUBTRACTION:
        return X, None

    bg, p = _load_bg_vector(sample)
    if bg is None:
        if SKIP_BG_IF_MISSING:
            logger.warning("BG missing for %s -> BG subtraction skipped.", sample)
            return X, None
        raise FileNotFoundError(f"BG missing for {sample} in {bg_dir}")

    bgZ = _align_bg_to_Z(bg, Z_src)
    X2 = X.astype(np.float64, copy=False) - bgZ[None, :]
    logger.info("%s: BG sub used: %s", sample, os.path.basename(p))
    return X2, p

# ========================== RESAMPLING =====================================
def _resample_matrix_to_std(X, Z_src):
    wns_src = np.linspace(ASSUME_SAMPLE_MIN, ASSUME_SAMPLE_MAX, int(Z_src))
    return np.vstack([np.interp(WN_STD, wns_src, X[i]) for i in range(X.shape[0])])

# ========================== OPTIONAL SCATTER (RMieSC via OCTAVVS) ==========
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
    mod = _try_import("octavvs.algorithms.correction")
    if mod is not None and hasattr(mod, "rmiesc") and callable(getattr(mod, "rmiesc")):
        return getattr(mod, "rmiesc"), "octavvs.algorithms.correction.rmiesc"

    try:
        import octavvs  # noqa
    except Exception as e:
        if verbose:
            logger.warning("OCTAVVS not importable: %s", e)
        return None, None

    found = []
    import octavvs  # noqa
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
    if app.size == 0:
        return None
    mode = (mode or "median").lower()
    if mode == "mean":
        ref = np.nanmean(app, axis=0)
    else:
        ref = np.nanmedian(app, axis=0)
    ref = np.asarray(ref, dtype=np.float32)
    if not np.all(np.isfinite(ref)):
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    return ref

def _scatter_correct_matrix_rmiesc(X, wns):
    global _OCTAVVS_RMIESC, _OCTAVVS_FAIL_COUNT, _OCTAVVS_LOGGED_FIRST_ERROR

    if (not APPLY_SCATTER_CORRECTION) or (SCATTER_BACKEND.lower() == "none"):
        return X
    if SCATTER_BACKEND.lower() != "octavvs":
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

        fn, nm = _find_octavvs_rmiesc(verbose=OCTAVVS_VERBOSE_SCAN)
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
    wn = np.asarray(wns, dtype=np.float32)

    ref = _build_rmiesc_ref(app, mode=RMIESC_REF_MODE)
    if ref is None:
        return X

    try:
        out = _OCTAVVS_RMIESC(wn, app, ref, iterations=RMIESC_ITERATIONS, verbose=RMIESC_VERBOSE)
        out = np.asarray(out)
        if out.shape == app.shape:
            return out.astype(np.float64, copy=False)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            out0 = np.asarray(out[0])
            if out0.shape == app.shape:
                return out0.astype(np.float64, copy=False)
    except Exception as e:
        if not _OCTAVVS_LOGGED_FIRST_ERROR:
            _OCTAVVS_LOGGED_FIRST_ERROR = True
            logger.warning("OCTAVVS rmiesc failed (first error) -> fallback after a few fails.")
            logger.warning("Exception: %s: %s", type(e).__name__, str(e))
            logger.warning("Traceback:\n%s", traceback.format_exc())

    _OCTAVVS_FAIL_COUNT += 1
    logger.warning("OCTAVVS rmiesc failed at runtime (fail %d/%d). Using uncorrected spectra.",
                   _OCTAVVS_FAIL_COUNT, _OCTAVVS_MAX_FAILS)
    return X

# ========================== Band extraction ================================
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

# ========================== Data collection ================================
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

    if X.shape[0] > MAX_PIXELS_PER_SAMPLE_ANALYSIS:
        rng = np.random.default_rng(abs(hash(sample)) % (2**32))
        take = rng.choice(X.shape[0], size=MAX_PIXELS_PER_SAMPLE_ANALYSIS, replace=False)
        X = X[take]

    # BG subtraction before resampling
    X, _ = _bg_subtract_cube_spectra(X, Z_src=Z_src, sample=sample)

    # resample to common axis
    if X.shape[1] != WN_STD_Z:
        X = _resample_matrix_to_std(X, Z_src=Z_src)

    # optional scatter correction (RMieSC)
    X = _scatter_correct_matrix_rmiesc(X, WN_STD)

    return X, WN_STD

# ========================== Feature builders ===============================
def _summarize_band_distribution(v, prefix):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            f"{prefix}_med": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
        }
    q1, q3 = np.percentile(v, [25, 75])
    return {
        f"{prefix}_med": float(np.median(v)),
        f"{prefix}_iqr": float(q3 - q1),
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_std": float(np.std(v, ddof=0)),
    }

def _build_features_median_spectrum(X, wns):
    med = np.nanmedian(X, axis=0)
    return {f"wn_{i:03d}": float(med[i]) for i in range(med.size)}

# ---- raw per-sample extraction cache for split-first CV ----
def _extract_per_sample_band_arrays(sample):
    """
    Safe precompute for split-first CV:
      - selected pixels
      - optional BG subtraction
      - resampling
      - optional RMieSC (sample-local)
      - raw per-pixel band arrays
    No train/test thresholds here.
    """
    X, wns = _collect_selected_pixel_spectra_std(sample)
    if X is None:
        return None

    band_dict = {}
    for bkey, (center, halfw) in BAND_LIBRARY.items():
        vals = _band_value_topk_mean_matrix(X, wns, float(center), float(halfw), k=TOPK_PER_BAND)
        band_dict[bkey] = np.asarray(vals, dtype=float)

    return band_dict, X, wns

def _build_features_from_band_arrays_bands_only(band_vals_dict):
    feats = {}
    for bkey in BAND_LIBRARY.keys():
        vals = np.asarray(band_vals_dict[bkey], float)
        vals = vals[np.isfinite(vals)]
        feats.update(_summarize_band_distribution(vals, bkey))
    return feats

def _compute_ratio_thresholds_from_training(train_band_arrays, ratio_defs):
    thresholds = {}

    for rname, cfg in ratio_defs.items():
        num_tuple = cfg["num"]
        den_tuple = cfg["den"]

        num_key = next(k for k, v in BAND_LIBRARY.items() if v == num_tuple)
        den_key = next(k for k, v in BAND_LIBRARY.items() if v == den_tuple)

        all_num = []
        all_den = []

        for band_dict in train_band_arrays:
            if band_dict is None:
                continue
            num = np.asarray(band_dict[num_key], float)
            den = np.asarray(band_dict[den_key], float)

            num = num[np.isfinite(num)]
            den = den[np.isfinite(den)]

            if num.size:
                all_num.append(num)
            if den.size:
                all_den.append(den)

        if all_num:
            all_num = np.concatenate(all_num)
            num_thr = max(np.nanpercentile(all_num, NUM_MIN_PCT), NUM_ABS_FLOOR)
        else:
            num_thr = NUM_ABS_FLOOR

        if all_den:
            all_den = np.concatenate(all_den)
            den_thr = max(np.nanpercentile(all_den, DENOM_MIN_PCT), DENOM_ABS_FLOOR)
        else:
            den_thr = DENOM_ABS_FLOOR

        thresholds[rname] = {"num_thr": float(num_thr), "den_thr": float(den_thr)}

    return thresholds

def _build_features_band_ratios_with_thresholds(band_vals_dict, ratio_defs, thresholds):
    feats = {}

    for rname, cfg in ratio_defs.items():
        num_tuple = cfg["num"]
        den_tuple = cfg["den"]

        num_key = next(k for k, v in BAND_LIBRARY.items() if v == num_tuple)
        den_key = next(k for k, v in BAND_LIBRARY.items() if v == den_tuple)

        num = np.asarray(band_vals_dict[num_key], float)
        den = np.asarray(band_vals_dict[den_key], float)

        thr_num = thresholds[rname]["num_thr"]
        thr_den = thresholds[rname]["den_thr"]

        ok = (
            np.isfinite(num) & np.isfinite(den) &
            (num > thr_num) & (den > thr_den)
        )

        rr = np.full_like(num, np.nan, dtype=float)
        rr[ok] = num[ok] / (den[ok] + EPS)

        if APPLY_RATIO_HARD_CAP:
            rr = np.clip(rr, 0, RATIO_HARD_CAP)

        rr = rr[np.isfinite(rr) & (rr > 0)]
        feats.update(_summarize_band_distribution(rr, _safe_name(rname)))

    return feats

# ========================== ML helpers =====================================
def _build_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "logreg":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=0
        )
    elif model_name == "linear_svm":
        clf = SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=0
        )
    else:
        raise ValueError("MODEL_NAME must be 'logreg' or 'linear_svm'")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])
    return pipe

def _crossval_predict_split_first(sample_names_in, sample_to_group, model_name):
    """
    SPLIT FIRST, then build fold-specific features inside each LOOCV fold.
    This reduces leakage, especially for band-ratio guards/thresholds.
    """
    # Safe precompute only sample-local info
    per_sample_band_cache = {}
    raw_store = {}

    for s in sample_names_in:
        if sample_to_group.get(s) not in ("WT", "KO"):
            continue
        out = _extract_per_sample_band_arrays(s)
        if out is None:
            continue
        band_dict, X, wns = out
        per_sample_band_cache[s] = band_dict
        raw_store[s] = (X, wns)

    valid_samples = sorted(per_sample_band_cache.keys())
    if len(valid_samples) < 4:
        raise RuntimeError("Too few valid samples for LOOCV.")

    samples_arr = np.array(valid_samples, dtype=object)
    y_all = np.array([1 if sample_to_group[s] == "KO" else 0 for s in valid_samples], dtype=int)

    loo = LeaveOneOut() if USE_LOOCV else LeaveOneOut()

    y_true, y_pred, y_prob = [], [], []
    fold_records = []
    feature_cols_master = None

    for train_idx, test_idx in loo.split(samples_arr, y_all):
        train_samples = samples_arr[train_idx]
        test_sample = samples_arr[test_idx][0]

        ytr = y_all[train_idx]
        yte = int(y_all[test_idx][0])

        train_rows = []
        test_row = None

        if FEATURE_MODE == "bands_only":
            for s in train_samples:
                feats = _build_features_from_band_arrays_bands_only(per_sample_band_cache[s])
                row = {"sample": s, "y": 1 if sample_to_group[s] == "KO" else 0}
                row.update(feats)
                train_rows.append(row)

            feats_te = _build_features_from_band_arrays_bands_only(per_sample_band_cache[test_sample])
            test_row = {"sample": test_sample, "y": yte}
            test_row.update(feats_te)

        elif FEATURE_MODE == "band_ratios":
            train_band_arrays = [per_sample_band_cache[s] for s in train_samples]
            ratio_thresholds = _compute_ratio_thresholds_from_training(train_band_arrays, BASE_RATIOS)

            for s in train_samples:
                feats = _build_features_band_ratios_with_thresholds(
                    per_sample_band_cache[s], BASE_RATIOS, ratio_thresholds
                )
                row = {"sample": s, "y": 1 if sample_to_group[s] == "KO" else 0}
                row.update(feats)
                train_rows.append(row)

            feats_te = _build_features_band_ratios_with_thresholds(
                per_sample_band_cache[test_sample], BASE_RATIOS, ratio_thresholds
            )
            test_row = {"sample": test_sample, "y": yte}
            test_row.update(feats_te)

        elif FEATURE_MODE == "median_spectrum":
            # uses raw_store (sample-local spectra), no train thresholds needed
            for s in train_samples:
                Xs, wns = raw_store[s]
                feats = _build_features_median_spectrum(Xs, wns)
                row = {"sample": s, "y": 1 if sample_to_group[s] == "KO" else 0}
                row.update(feats)
                train_rows.append(row)

            Xte, wns_te = raw_store[test_sample]
            feats_te = _build_features_median_spectrum(Xte, wns_te)
            test_row = {"sample": test_sample, "y": yte}
            test_row.update(feats_te)

        else:
            raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")

        if not train_rows:
            raise RuntimeError("No training rows in a fold.")

        # define feature columns from TRAIN only
        feature_cols = sorted([k for k in train_rows[0].keys() if k not in ("sample", "y")])

        for r in train_rows:
            for c in feature_cols:
                r.setdefault(c, np.nan)
        for c in feature_cols:
            test_row.setdefault(c, np.nan)

        Xtr = np.array([[r[c] for c in feature_cols] for r in train_rows], dtype=float)
        Xte = np.array([[test_row[c] for c in feature_cols]], dtype=float)

        if feature_cols_master is None:
            feature_cols_master = feature_cols

        pipe = _build_model(model_name)
        pipe.fit(Xtr, ytr)

        pred = int(pipe.predict(Xte)[0])

        prob_ko = np.nan
        try:
            proba = pipe.predict_proba(Xte)[0]
            classes_ = pipe.named_steps["clf"].classes_
            ko_pos = int(np.where(classes_ == 1)[0][0])
            prob_ko = float(proba[ko_pos])
        except Exception:
            try:
                score = float(pipe.decision_function(Xte)[0])
                prob_ko = 1.0 / (1.0 + np.exp(-score))
            except Exception:
                prob_ko = np.nan

        y_true.append(yte)
        y_pred.append(pred)
        y_prob.append(prob_ko)
        fold_records.append((str(test_sample), yte, pred, prob_ko))

    groups_arr = np.array([sample_to_group[s] for s in samples_arr], dtype=object)

    return (
        np.array(y_true, dtype=int),
        np.array(y_pred, dtype=int),
        np.array(y_prob, dtype=float),
        fold_records,
        samples_arr,
        groups_arr,
        raw_store,
        feature_cols_master
    )

# ========================== Full-data feature table (for coefficient plot only) ====
def _prepare_full_dataset_for_interpretation(sample_names_in, sample_to_group, raw_store):
    """
    Build full dataset feature table AFTER evaluation, for interpretation plots only.
    For band_ratios, thresholds are computed on all samples (NOT for performance reporting).
    """
    rows = []
    feature_cols = None

    # collect sample-local band cache (from raw_store if available)
    band_cache = {}
    for s in sample_names_in:
        if s not in raw_store:
            continue
        X, wns = raw_store[s]
        bdict = {}
        for bkey, (center, halfw) in BAND_LIBRARY.items():
            bdict[bkey] = _band_value_topk_mean_matrix(X, wns, float(center), float(halfw), k=TOPK_PER_BAND)
        band_cache[s] = bdict

    valid_samples = sorted([s for s in sample_names_in if s in raw_store and sample_to_group.get(s) in ("WT","KO")])

    ratio_thresholds_all = None
    if FEATURE_MODE == "band_ratios":
        ratio_thresholds_all = _compute_ratio_thresholds_from_training(
            [band_cache[s] for s in valid_samples], BASE_RATIOS
        )

    for s in valid_samples:
        yv = 1 if sample_to_group[s] == "KO" else 0
        Xs, wns = raw_store[s]

        if FEATURE_MODE == "bands_only":
            feats = _build_features_from_band_arrays_bands_only(band_cache[s])
        elif FEATURE_MODE == "band_ratios":
            feats = _build_features_band_ratios_with_thresholds(band_cache[s], BASE_RATIOS, ratio_thresholds_all)
        elif FEATURE_MODE == "median_spectrum":
            feats = _build_features_median_spectrum(Xs, wns)
        else:
            raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")

        row = {"sample": s, "group": sample_to_group[s], "y": yv}
        row.update(feats)
        rows.append(row)

    if not rows:
        return None, None, None, None, None

    feature_cols = sorted([k for k in rows[0].keys() if k not in ("sample","group","y")])
    for r in rows:
        for c in feature_cols:
            r.setdefault(c, np.nan)

    Xtab = np.array([[r[c] for c in feature_cols] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    samples = np.array([r["sample"] for r in rows], dtype=object)
    groups_arr = np.array([r["group"] for r in rows], dtype=object)
    return Xtab, y, samples, groups_arr, feature_cols

# ========================== Plotting =======================================
def plot_confusion_pub(cm, out_png, labels=("WT","KO"), title="Confusion Matrix"):
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title)
    ax.set_xticks([0,1], labels)
    ax.set_yticks([0,1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=FS["base"]+1)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_roc_pub(y_true, y_prob, out_png, title="ROC Curve"):
    _apply_pub_style()
    mask = np.isfinite(y_prob)
    if mask.sum() < 3 or len(np.unique(y_true[mask])) < 2:
        logger.warning("ROC plot skipped: insufficient valid probabilities.")
        return

    fpr, tpr, _ = roc_curve(y_true[mask], y_prob[mask])
    auc = roc_auc_score(y_true[mask], y_prob[mask])

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0,1], [0,1], "--", lw=1)

    ax.set_xlim(0,1)
    ax.set_ylim(0,1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_cv_probabilities_pub(samples, y_true, y_prob, out_png):
    """
    Improved version to fix overlap / ugly layout:
      - separate top legend and title
      - labels are on x-axis ticks (2-line labels), not huge text clutter on top
      - probability values annotated near points
      - misclassified points marked with X overlay
      - background shading for threshold regions
    """
    _apply_pub_style()

    samples = np.asarray(samples)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    x = np.arange(len(samples))
    pred = np.where(np.isfinite(y_prob), (y_prob >= 0.5).astype(int), -1)
    is_mis = (pred != -1) & (pred != y_true)
    is_cor = (pred != -1) & (pred == y_true)

    point_colors = np.array([BASE_KO if yt == 1 else BASE_WT for yt in y_true], dtype=object)

    fig_w = max(9.2, 0.78 * len(samples))
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))

    # background shading
    ax.axhspan(0.0, 0.5, color="#2ca02c", alpha=0.06, zorder=0)
    ax.axhspan(0.5, 1.0, color="#d62728", alpha=0.06, zorder=0)
    ax.axhline(0.5, linestyle="--", linewidth=1.8, color="#1f77b4", zorder=1)

    valid = np.isfinite(y_prob)

    # base points
    ax.scatter(x[valid], y_prob[valid],
               s=320, c=point_colors[valid],
               edgecolors="black", linewidths=1.2, alpha=0.9, zorder=3)

    # X overlay for misclassified
    if np.any(is_mis & valid):
        ax.scatter(x[is_mis & valid], y_prob[is_mis & valid],
                   s=430, marker="X", c="black",
                   edgecolors="black", linewidths=1.0, zorder=4)

    # annotate probability values
    for i in range(len(samples)):
        if not np.isfinite(y_prob[i]):
            continue
        dy = 0.03 if y_prob[i] < 0.9 else -0.05
        va = "bottom" if dy > 0 else "top"
        ax.text(i, float(np.clip(y_prob[i] + dy, 0.02, 0.98)),
                f"{y_prob[i]:.2f}",
                ha="center", va=va, fontsize=FS["annot"], zorder=5)

    # x tick labels as two lines: sample + true class
    xticklabels = [f"{s}\n{'KO' if yt==1 else 'WT'}" for s, yt in zip(samples, y_true)]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")

    ax.set_xlim(-0.7, len(samples)-0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Predicted KO probability")
    ax.set_xlabel("Sample (true class)")
    ax.set_title("LOOCV predicted KO probabilities by sample", pad=18)
    ax.grid(True, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # legend proxies
    h_wt = plt.Line2D([0], [0], marker='o', color='w', label='True WT',
                      markerfacecolor=BASE_WT, markeredgecolor='black', markersize=10)
    h_ko = plt.Line2D([0], [0], marker='o', color='w', label='True KO',
                      markerfacecolor=BASE_KO, markeredgecolor='black', markersize=10)
    h_cor = plt.Line2D([0], [0], marker='o', color='black', label='Correct',
                       markerfacecolor='none', markeredgecolor='black', markersize=10, linewidth=0)
    h_mis = plt.Line2D([0], [0], marker='X', color='black', label='Misclassified',
                       markerfacecolor='black', markeredgecolor='black', markersize=10, linewidth=0)

    ax.legend(handles=[h_wt, h_ko, h_cor, h_mis],
              loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False)

    # summary text
    valid_mask = np.isfinite(y_prob)
    acc_prob = np.nan
    if np.any(valid_mask):
        acc_prob = np.mean((y_prob[valid_mask] >= 0.5).astype(int) == y_true[valid_mask])
    txt = f"Threshold = 0.50   |   Acc from probs = {acc_prob:.2f}   |   n = {len(samples)}"
    ax.text(0.012, 0.02, txt, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS["annot"]+1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, boxstyle="round,pad=0.2"))

    fig.tight_layout()
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_feature_coefficients_pub(X, y, feature_cols, model_name, out_png, top_n=20):
    """
    Fit full dataset model (for interpretation only; not CV performance) and plot top coefficients.
    """
    if X is None or len(X) == 0:
        logger.warning("Coefficient plot skipped: empty dataset.")
        return

    _apply_pub_style()

    pipe = _build_model(model_name)
    pipe.fit(X, y)

    clf = pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        logger.warning("Coefficient plot skipped: model has no coef_.")
        return

    coefs = np.ravel(clf.coef_)
    order = np.argsort(np.abs(coefs))[::-1][:min(top_n, len(coefs))]
    names = [feature_cols[i] for i in order]
    vals = coefs[order]

    fig_h = max(5, 0.33 * len(order) + 1.5)
    fig, ax = plt.subplots(figsize=(8.8, fig_h))
    y_pos = np.arange(len(order))[::-1]

    bar_colors = [BASE_KO if v > 0 else BASE_WT for v in vals[::-1]]
    ax.barh(y_pos, vals[::-1], edgecolor="black", linewidth=0.4, color=bar_colors)
    ax.set_yticks(y_pos, names[::-1])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Coefficient (interpretation-only; positive -> KO)")
    ax.set_title(f"Top {len(order)} feature coefficients ({model_name})")
    ax.grid(True, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # interpretation-only note
    ax.text(0.99, 0.01, "Fit on full dataset (for interpretation only)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65))

    fig.tight_layout()
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

def plot_sample_median_spectra_pub(raw_store, groups_arr_map, out_png):
    _apply_pub_style()

    wt_keys = [s for s,g in groups_arr_map.items() if g == "WT" and s in raw_store]
    ko_keys = [s for s,g in groups_arr_map.items() if g == "KO" and s in raw_store]
    wt_col_map, ko_col_map = _group_sample_colors(wt_keys, ko_keys)

    fig, ax = plt.subplots(figsize=(8.8, 5.4))

    for s in wt_keys:
        X, wns = raw_store[s]
        med = np.nanmedian(X, axis=0)
        ax.plot(wns, med, lw=1.5, alpha=0.9, color=wt_col_map[s], label=s)

    for s in ko_keys:
        X, wns = raw_store[s]
        med = np.nanmedian(X, axis=0)
        ax.plot(wns, med, lw=1.5, alpha=0.9, color=ko_col_map[s], label=s)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Median absorbance (a.u.)")
    ax.set_title("Sample median spectra (selected clusters)")
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(out_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

# ========================== Reporting ======================================
def _print_summary(y_true, y_pred, y_prob, n_features):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    auc = np.nan
    try:
        m = np.isfinite(y_prob)
        if m.sum() >= 3 and len(np.unique(y_true[m])) == 2:
            auc = roc_auc_score(y_true[m], y_prob[m])
    except Exception:
        auc = np.nan

    print(f"Feature mode:        {FEATURE_MODE}")
    print(f"Model:               {MODEL_NAME}")
    print(f"n_samples:           {len(y_true)}")
    print(f"n_features:          {n_features}")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Balanced accuracy:   {bacc:.4f}")
    print(f"AUC (KO prob):       {auc if np.isfinite(auc) else 'nan'}")
    print("Confusion matrix [WT, KO]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["WT","KO"], digits=4))

    # extra warning if perfect
    if (acc == 1.0) or (np.isfinite(auc) and auc == 1.0):
        print("\n[Warning] Perfect/near-perfect result detected. With n=14, verify robustness carefully.")
        print("- This script reduces leakage by split-first LOOCV.")
        print("- Still consider permutation testing and external validation / new samples.")

    return acc, bacc, auc, cm

def _save_predictions_csv(path, samples, y_true, y_pred, y_prob):
    with open(path, "w", encoding="utf-8") as f:
        f.write("sample,true_label,pred_label,prob_KO\n")
        for s, yt, yp, p in zip(samples, y_true, y_pred, y_prob):
            t = "KO" if yt == 1 else "WT"
            pr = "KO" if yp == 1 else "WT"
            ps = "" if not np.isfinite(p) else f"{p:.8f}"
            f.write(f"{s},{t},{pr},{ps}\n")
    logger.info("Saved predictions: %s", path)

def _save_metrics_txt(path, acc, bacc, auc, n_samples, n_features):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Feature mode:        {FEATURE_MODE}\n")
        f.write(f"Model:               {MODEL_NAME}\n")
        f.write(f"n_samples:           {n_samples}\n")
        f.write(f"n_features:          {n_features}\n")
        f.write(f"Background subtract: {APPLY_BG_SUBTRACTION}\n")
        f.write(f"BG dir:              {bg_dir}\n")
        f.write(f"Scatter correction:  {APPLY_SCATTER_CORRECTION}\n")
        f.write(f"Scatter backend:     {SCATTER_BACKEND}\n")
        f.write(f"Accuracy:            {acc:.6f}\n")
        f.write(f"Balanced accuracy:   {bacc:.6f}\n")
        f.write(f"AUC (KO prob):       {auc if np.isfinite(auc) else 'nan'}\n")
        f.write("CV scheme:           Leave-One-Out (split-first feature building)\n")
    logger.info("Saved metrics: %s", path)

# =============================== MAIN ======================================
if __name__ == "__main__":
    logger.info("FEATURE_MODE=%s | MODEL=%s", FEATURE_MODE, MODEL_NAME)
    logger.info("BG subtraction=%s | bg_dir=%s", APPLY_BG_SUBTRACTION, bg_dir)
    logger.info("Scatter correction=%s | backend=%s", APPLY_SCATTER_CORRECTION, SCATTER_BACKEND)
    logger.info("Axis: %.1f..%.1f -> STD %.1f..%.1f (Z=%d)",
                ASSUME_SAMPLE_MIN, ASSUME_SAMPLE_MAX, WN_STD_MIN, WN_STD_MAX, WN_STD_Z)

    sample_to_group = {s: "WT" for s in groups["WT"]}
    sample_to_group.update({s: "KO" for s in groups["KO"]})

    # ===== Split-first LOOCV (leakage-reduced) =====
    with LogTimer("Cross-validated predictions (split-first LOOCV)"):
        (
            y_true, y_pred, y_prob, fold_records,
            samples_eval, groups_arr_eval, raw_store, feature_cols_cv
        ) = _crossval_predict_split_first(sample_names, sample_to_group, MODEL_NAME)

    n_features_report = len(feature_cols_cv) if feature_cols_cv is not None else 0
    acc, bacc, auc, cm = _print_summary(y_true, y_pred, y_prob, n_features=n_features_report)

    # save outputs
    _save_predictions_csv(os.path.join(save_dir, "cv_predictions_splitfirst.csv"),
                          samples_eval, y_true, y_pred, y_prob)
    _save_metrics_txt(os.path.join(save_dir, "metrics_summary_splitfirst.txt"),
                      acc, bacc, auc, len(y_true), n_features_report)

    # plots (evaluation)
    with LogTimer("Plot confusion matrix"):
        plot_confusion_pub(
            cm,
            os.path.join(save_dir, "confusion_matrix_pub.png"),
            labels=("WT","KO"),
            title=f"Confusion Matrix ({MODEL_NAME}, {FEATURE_MODE})"
        )

    with LogTimer("Plot ROC"):
        plot_roc_pub(
            y_true, y_prob,
            os.path.join(save_dir, "roc_curve_pub.png"),
            title=f"ROC ({MODEL_NAME}, {FEATURE_MODE})"
        )

    with LogTimer("Plot sample probabilities"):
        plot_cv_probabilities_pub(
            samples_eval, y_true, y_prob,
            os.path.join(save_dir, "cv_probabilities_pub.png")
        )

    # ===== Full-data interpretation-only table (for coefficient plot) =====
    with LogTimer("Prepare full dataset for interpretation-only plots"):
        out = _prepare_full_dataset_for_interpretation(sample_names, sample_to_group, raw_store)
        Xtab_full, y_full, samples_full, groups_full, feature_cols_full = out

    with LogTimer("Plot feature coefficients (interpretation-only)"):
        if Xtab_full is not None and FEATURE_MODE in ("bands_only", "band_ratios", "median_spectrum"):
            plot_feature_coefficients_pub(
                Xtab_full, y_full, feature_cols_full, MODEL_NAME,
                os.path.join(save_dir, "feature_coefficients_pub.png"),
                top_n=20
            )
        else:
            logger.warning("Skipping coefficient plot (no full dataset available).")

    with LogTimer("Plot sample median spectra"):
        groups_map = {s: sample_to_group[s] for s in sample_names if s in raw_store}
        plot_sample_median_spectra_pub(
            raw_store, groups_map,
            os.path.join(save_dir, "sample_median_spectra_pub.png")
        )

    logger.info("Done. Outputs saved to: %s", save_dir)