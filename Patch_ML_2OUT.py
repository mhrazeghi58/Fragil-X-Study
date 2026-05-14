# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 15:14:37 2026

@author: hrazeghikondela
"""

# ======================================================================
# WT vs KO FTIR classification (SPLIT-FIRST, LEAKAGE-REDUCED) — PATCH-WISE
#
# PATCH-WISE / ANIMAL-LEVEL EVALUATION:
#   - Patches are the ML samples
#   - Animals are the split groups
#   - In each split: 5 WT + 5 KO animals for training, 2 WT + 2 KO animals for test
#   - Ratio thresholds are computed from TRAIN animals only
#   - Results are reported at both patch level and animal level
#
# RUNS 4 RESULT SETS:
#   (A) Train RIGHT -> Test RIGHT   : repeated balanced 5v2 animal holdout
#   (B) Train RIGHT -> Test LEFT    : repeated balanced 5v2 animal holdout
#   (C) Train LEFT  -> Test LEFT    : repeated balanced 5v2 animal holdout
#   (D) Train LEFT  -> Test RIGHT   : repeated balanced 5v2 animal holdout
#
# Patch extraction:
#   selected DG pixels -> spatial patches -> per-patch bands/ratios -> ML
#
# Notes:
#   - Animal-level separation is enforced before any train/test evaluation
#   - Patch counts are capped per animal to reduce subject imbalance
#   - Animal-level predictions are obtained by averaging patch P(KO)
# ======================================================================

from __future__ import annotations

import os, time, logging, traceback, importlib, pkgutil, inspect
from itertools import combinations
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

# ============================== USER CONFIG ==============================

RIGHT_CLUSTER_DIR = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R\UMAP_clustering_test"
RIGHT_CUBE_DIR    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"

LEFT_CLUSTER_DIR  = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L\UMAP_clustering_8Cluster"
LEFT_CUBE_DIR     = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"

CLUSTER_NPZ_PATTERN = "{sid}_umap_kmeans.npz"
RIGHT_CUBE_PATTERN  = "masked_cube_{sid}.npz"
LEFT_CUBE_PATTERN   = "masked_cube_{sid}.npz"

SAVE_DIR = os.path.join(RIGHT_CUBE_DIR, "WT_KO_ML_PATCHWISE_5v2_ANIMALLEVEL__v1")
os.makedirs(SAVE_DIR, exist_ok=True)

# ---- BG subtraction ----
APPLY_BG_SUBTRACTION = True
SKIP_BG_IF_MISSING   = True

BG_DIR_RIGHT = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\Area_Right_B\BG_First300ROI_MeanSpectrum"
BG_DIR_LEFT  = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\Area_Right_B\BG_First300ROI_MeanSpectrum"

BG_PATTERNS = [
    "bg_first300_mean_spectrum_{sid}.npz",
    "bg_first30_mean_spectrum_{sid}.npz",
    "bg_mean_spectrum_{sid}.npz",
    "bg_{sid}.npz",
]
DROP_BG_FIRST_5_WHEN_426_TO_421 = True

# ---- Common axis ----
ASSUME_SAMPLE_MIN = 960.0
ASSUME_SAMPLE_MAX = 1800.0
WN_STD_MIN = 960.0
WN_STD_MAX = 1800.0
WN_STD_Z   = 426
WN_STD     = np.linspace(WN_STD_MIN, WN_STD_MAX, WN_STD_Z)

# ---- Samples / groups ----
sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

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

# ---- Feature mode / model ----
FEATURE_MODE = "band_ratios"     # "bands_only", "band_ratios"
MODEL_NAME   = "logreg"          # "logreg" or "linear_svm"

# ---- Pixel / patch settings ----
MAX_PIXELS_FOR_SPECTRA_PLOTS = 3000

PATCH_H = 20        
PATCH_W = 20
PATCH_MIN_SELECTED_PIXELS = 200
MAX_PATCHES_PER_ANIMAL = 100
PATCH_SELECTION_MODE = "largest"   # "largest" or "random"

# ---- 5v2 animal-level evaluation ----
TRAIN_ANIMALS_PER_CLASS = 5
TEST_ANIMALS_PER_CLASS  = 2
N_REPEATS_PER_SIDE = None          # None -> use all combinations (21*21=441)
SPLIT_RANDOM_SEED = 123

# ---- Band extraction ----
TOPK_PER_BAND = 1
window = 30
EPS = 1e-12

# ---- Ratio guard thresholds (TRAINING ONLY) ----
APPLY_RATIO_HARD_CAP = True
RATIO_HARD_CAP       = 15.0
DENOM_MIN_PCT   = 3.0
DENOM_ABS_FLOOR = 1e-2
NUM_MIN_PCT     = 15.0
NUM_ABS_FLOOR   = 1e-4

# ===================== RMieSC SETTINGS =====================
APPLY_SCATTER_CORRECTION = False
SCATTER_BACKEND = "octavvs"
RMIESC_ITERATIONS = 10
RMIESC_VERBOSE = False
RMIESC_REF_MODE = "median"
OCTAVVS_VERBOSE_SCAN = True
RMIESC_CLIP = None  # (-2.0, 2.0)

# ===================== SHIFT (DISABLED) =====================
PRE_BG_SHIFT = 0.0

# ===================== BASIC QUALITY FILTER (before RMieSC) =======
FILTER_BAD_SPECTRA_BEFORE_RMIESC = True
MIN_SPECTRUM_L2_NORM = 1e-3
MAX_ABS_VALUE = 1e3
MIN_KEEP_SPECTRA = 100

# ===================== RMieSC RESAMPLING RETRIES ==================
RMIESC_SUBSET_N = 800
RMIESC_MAX_TRIES = 8
RMIESC_RANDOM_SEED_BASE = 1230

# ============================== PUB STYLE ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WT_KO_ML_SAMPLE")

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 12

BASE_WT = "#2ca02c"
BASE_KO = "#d62728"

def _apply_pub_style():
    fam = FONT_FAMILY or "DejaVu Sans"
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": PUB_DPI,
        "font.family": fam,
        "font.size": BASE_FONTSIZE,
        "axes.labelsize": BASE_FONTSIZE + 2,
        "axes.titlesize": BASE_FONTSIZE + 4,
        "xtick.labelsize": BASE_FONTSIZE,
        "ytick.labelsize": BASE_FONTSIZE,
        "axes.linewidth": 1.2,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "mathtext.default": "regular",
    })

def _short_title(s: str, maxlen: int = 32) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= maxlen else (s[:maxlen-1] + "...")

def _format_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"

def _savefig(fig, path_png):
    fig.savefig(path_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(path_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)

@contextmanager
def LogTimer(msg):
    t0 = time.time()
    logger.info(msg + " ...")
    try:
        yield
    finally:
        logger.info("%s done in %.2fs", msg, time.time() - t0)

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

# ============================== BANDS ==============================
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

# ============================== OCTAVVS RMieSC ==============================
_OCTAVVS_RMIESC = None
_OCTAVVS_FAILS = 0
_OCTAVVS_MAX_FAILS = 2
_OCTAVVS_LOGGED_FIRST = False

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
    ref = np.nanmedian(app, axis=0) if mode != "mean" else np.nanmean(app, axis=0)
    ref = np.asarray(ref, dtype=np.float64)
    ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)

    max_abs = float(np.max(np.abs(ref))) if ref.size else 0.0
    energy  = float(ref @ ref) if ref.size else 0.0
    if (not np.isfinite(max_abs)) or (max_abs < 1e-8) or (not np.isfinite(energy)) or (energy < 1e-12):
        norms = np.linalg.norm(app, axis=1)
        norms = np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanmax(norms) < 1e-8:
            return None
        idx = int(np.nanargmax(norms))
        ref = np.asarray(app[idx], dtype=np.float64)
        ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
    return ref

def _quality_mask_basic(X):
    X = np.asarray(X)
    finite = np.all(np.isfinite(X), axis=1)
    norms = np.linalg.norm(X, axis=1)
    norms = np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
    not_flat = norms >= MIN_SPECTRUM_L2_NORM
    not_huge = np.nanmax(np.abs(X), axis=1) <= MAX_ABS_VALUE
    return finite & not_flat & not_huge

def _sample_subset(X, n, rng):
    if X.shape[0] <= n:
        return X
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx]

def apply_rmiesc(X, wn, sid="", side=""):
    global _OCTAVVS_RMIESC, _OCTAVVS_FAILS, _OCTAVVS_LOGGED_FIRST

    if (not APPLY_SCATTER_CORRECTION) or (SCATTER_BACKEND.lower() == "none"):
        return X
    if SCATTER_BACKEND.lower() != "octavvs":
        logger.warning("Unknown SCATTER_BACKEND=%s -> skipping scatter correction", SCATTER_BACKEND)
        return X
    if _OCTAVVS_FAILS >= _OCTAVVS_MAX_FAILS:
        return X

    if _OCTAVVS_RMIESC is None:
        fn, nm = _find_octavvs_rmiesc(verbose=OCTAVVS_VERBOSE_SCAN)
        if fn is None:
            logger.warning("OCTAVVS rmiesc not found -> skipping scatter correction.")
            _OCTAVVS_FAILS = _OCTAVVS_MAX_FAILS
            return X
        _OCTAVVS_RMIESC = fn
        logger.info("Using OCTAVVS rmiesc: %s", nm)
        try:
            logger.info("rmiesc signature: %s", str(inspect.signature(_OCTAVVS_RMIESC)))
        except Exception:
            pass

    wn = np.asarray(wn, dtype=np.float64).ravel()
    app_all = np.asarray(X, dtype=np.float64, order="C")
    app_all = np.nan_to_num(app_all, nan=0.0, posinf=0.0, neginf=0.0)

    if FILTER_BAD_SPECTRA_BEFORE_RMIESC:
        keep = _quality_mask_basic(app_all)
        if np.sum(keep) >= MIN_KEEP_SPECTRA:
            removed = int(np.sum(~keep))
            if removed > 0:
                logger.info("%s %s: removed %d/%d spectra before RMieSC", side, sid, removed, app_all.shape[0])
            app_all = app_all[keep]

    if app_all.shape[0] < MIN_KEEP_SPECTRA:
        logger.warning("%s %s: too few spectra after filtering (%d) -> skip RMieSC", side, sid, app_all.shape[0])
        return X

    if RMIESC_CLIP is not None:
        lo, hi = RMIESC_CLIP
        app_all = np.clip(app_all, lo, hi)

    base_seed = abs(hash((RMIESC_RANDOM_SEED_BASE, side, sid))) % (2**32)

    for t in range(RMIESC_MAX_TRIES):
        rng = np.random.default_rng(base_seed + t)
        app = _sample_subset(app_all, RMIESC_SUBSET_N, rng)

        ref = _build_rmiesc_ref(app, mode=RMIESC_REF_MODE)
        if ref is None:
            continue

        try:
            out = _OCTAVVS_RMIESC(wn, app, ref, iterations=RMIESC_ITERATIONS, verbose=RMIESC_VERBOSE)
            out0 = np.asarray(out[0] if isinstance(out, (tuple, list)) else out)
            if out0.shape != app.shape:
                raise ValueError(f"rmiesc output shape {out0.shape} != input {app.shape}")
            out0 = np.nan_to_num(out0, nan=0.0, posinf=0.0, neginf=0.0)
            logger.info("%s %s: RMieSC success on try %d using n=%d spectra", side, sid, t+1, out0.shape[0])
            return out0.astype(np.float64, copy=False)

        except Exception as e:
            if not _OCTAVVS_LOGGED_FIRST:
                _OCTAVVS_LOGGED_FIRST = True
                logger.warning("RMieSC first failure: %s", e)
                logger.warning("Traceback:\n%s", traceback.format_exc())
            logger.warning("%s %s: RMieSC try %d/%d failed (%s) -> resample",
                           side, sid, t+1, RMIESC_MAX_TRIES, str(e))

    _OCTAVVS_FAILS += 1
    logger.warning("RMieSC failed (%d/%d) -> returning uncorrected spectra",
                   _OCTAVVS_FAILS, _OCTAVVS_MAX_FAILS)
    return X

# ============================== IO + PREP ==============================
def _load_npz(path):
    return np.load(path, allow_pickle=True)

def _load_cluster_labels_indices(cluster_dir, sid):
    f = os.path.join(cluster_dir, CLUSTER_NPZ_PATTERN.format(sid=sid))
    if not os.path.exists(f):
        logger.warning("Missing cluster file for %s: %s", sid, f)
        return None
    z = _load_npz(f)
    if "cluster_labels" not in z or "pixel_indices" not in z:
        raise ValueError(f"{sid}: cluster npz missing required keys. keys={list(z.files)}")
    return np.asarray(z["cluster_labels"]), np.asarray(z["pixel_indices"])

def _load_cube(cube_dir, cube_pattern, sid):
    f = os.path.join(cube_dir, cube_pattern.format(sid=sid))
    if not os.path.exists(f):
        logger.warning("Missing cube file for %s: %s", sid, f)
        return None
    z = _load_npz(f)
    if "data" not in z:
        raise ValueError(f"{sid}: cube npz missing key 'data'. keys={list(z.files)}")
    return np.asarray(z["data"])

def _selected_mask_from_clusters(H, W, labels, indices, sel_lst):
    if indices.ndim == 2 and indices.shape[1] == 2:
        rows = indices[:, 0].astype(int)
        cols = indices[:, 1].astype(int)
        use = np.isin(labels, sel_lst)
        r = np.clip(rows[use], 0, H-1)
        c = np.clip(cols[use], 0, W-1)
        m = np.zeros((H, W), dtype=bool)
        m[r, c] = True
        return m
    flat = indices.astype(int)
    use = np.isin(labels, sel_lst)
    flat_use = flat[use]
    flat_use = flat_use[(flat_use >= 0) & (flat_use < H*W)]
    m = np.zeros((H*W,), dtype=bool)
    m[flat_use] = True
    return m.reshape(H, W)

def _find_bg_file(bg_dir, sid):
    for pat in BG_PATTERNS:
        p = os.path.join(bg_dir, pat.format(sid=sid))
        if os.path.exists(p):
            return p
    return None

def _load_bg_vector(bg_dir, sid):
    p = _find_bg_file(bg_dir, sid)
    if p is None:
        return None, None
    z = _load_npz(p)
    for k in z.files:
        a = np.asarray(z[k])
        if np.issubdtype(a.dtype, np.number) and a.ndim == 1 and a.size > 50:
            return a.astype(np.float64).ravel(), p
    raise ValueError(f"{sid}: BG file has no 1D numeric vector. keys={list(z.files)}")

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

def _bg_subtract(X, Z_src, sid, bg_dir):
    if not APPLY_BG_SUBTRACTION:
        return X
    if not bg_dir:
        return X
    bg, _ = _load_bg_vector(bg_dir, sid)
    if bg is None:
        if SKIP_BG_IF_MISSING:
            logger.warning("BG missing for %s in %s -> skipping.", sid, bg_dir)
            return X
        raise FileNotFoundError(f"Missing BG for {sid} in {bg_dir}")
    bgZ = _align_bg_to_Z(bg, Z_src)
    Y = X.astype(np.float64, copy=False) - bgZ[None, :]
    return np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

def _resample_to_std(X, Z_src):
    wns_src = np.linspace(ASSUME_SAMPLE_MIN, ASSUME_SAMPLE_MAX, int(Z_src))
    Y = np.vstack([np.interp(WN_STD, wns_src, X[i]) for i in range(X.shape[0])])
    return np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)

# ============================== BAND EXTRACT ==============================
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

def _summarize(v, prefix):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"{prefix}_med": np.nan, f"{prefix}_iqr": np.nan, f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan}
    q1, q3 = np.percentile(v, [25, 75])
    return {
        f"{prefix}_med": float(np.median(v)),
        f"{prefix}_iqr": float(q3 - q1),
        f"{prefix}_mean": float(np.mean(v)),
        f"{prefix}_std": float(np.std(v, ddof=0)),
    }

def _build_model(model_name):
    model_name = model_name.lower()
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=0)
    elif model_name == "linear_svm":
        clf = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=0)
    else:
        raise ValueError("MODEL_NAME must be 'logreg' or 'linear_svm'")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

# ============================== SIDE CACHE (PATCH-WISE) ==============================
def _extract_patch_boxes(mask, patch_h, patch_w, min_pixels):
    H, W = mask.shape
    boxes = []
    patch_idx = 0
    for r0 in range(0, H, patch_h):
        r1 = min(H, r0 + patch_h)
        for c0 in range(0, W, patch_w):
            c1 = min(W, c0 + patch_w)
            sub = mask[r0:r1, c0:c1]
            n_sel = int(np.sum(sub))
            if n_sel >= int(min_pixels):
                boxes.append({
                    "patch_idx": patch_idx,
                    "r0": int(r0), "r1": int(r1),
                    "c0": int(c0), "c1": int(c1),
                    "n_selected": n_sel,
                })
                patch_idx += 1
    return boxes

def _select_patch_boxes(boxes, max_patches, sid, side_name):
    if max_patches is None or len(boxes) <= max_patches:
        return boxes

    mode = str(PATCH_SELECTION_MODE).lower()
    if mode == "largest":
        boxes = sorted(boxes, key=lambda b: (-int(b["n_selected"]), int(b["r0"]), int(b["c0"])))
        return boxes[:max_patches]

    rng = np.random.default_rng(abs(hash((side_name, sid, "patchselect"))) % (2**32))
    idx = rng.choice(len(boxes), size=max_patches, replace=False)
    idx = np.sort(idx)
    return [boxes[i] for i in idx]

def build_side_cache(side_name, cluster_dir, cube_dir, cube_pattern, bg_dir):
    """
    Returns:
      patch_cache: dict[sid] -> list[patch_record]
      raw_before: dict[sid] -> (X_before_rmiesc, wn)
      raw_after:  dict[sid] -> (X_after_rmiesc,  wn)
    """
    patch_cache = {}
    raw_before = {}
    raw_after = {}

    for sid in sample_names:
        if sid not in groups["WT"] and sid not in groups["KO"]:
            continue

        out = _load_cluster_labels_indices(cluster_dir, sid)
        if out is None:
            continue
        labels, indices = out

        cube = _load_cube(cube_dir, cube_pattern, sid)
        if cube is None:
            continue

        H, W, Z_src = cube.shape
        sel = selected_clusters.get(sid, [])
        if not sel:
            continue

        mask = _selected_mask_from_clusters(H, W, labels, indices, sel)
        if not np.any(mask):
            logger.warning("%s %s: selected mask empty", side_name, sid)
            continue

        rr, cc = np.where(mask)
        logger.info("%s %s: selected pixels total = %d", side_name, sid, rr.size)

        X_all = cube[rr, cc, :].astype(np.float64, copy=False)

        if PRE_BG_SHIFT is not None and float(PRE_BG_SHIFT) != 0.0:
            X_all = X_all + float(PRE_BG_SHIFT)

        X_all = _bg_subtract(X_all, Z_src, sid, bg_dir)

        if X_all.shape[1] != WN_STD_Z:
            X_all = _resample_to_std(X_all, Z_src)

        X_before = X_all.astype(np.float64, copy=True)
        raw_before[sid] = (X_before, WN_STD)

        X_after = apply_rmiesc(X_all, WN_STD, sid=sid, side=side_name)
        X_after = np.asarray(X_after, dtype=np.float64)

        if X_after.shape[0] != rr.size:
            logger.warning(
                "%s %s: X_after rows (%d) != selected pixels (%d); "
                "falling back to pre-RMieSC-aligned spectra for patch mapping.",
                side_name, sid, X_after.shape[0], rr.size
            )
            X_after = X_before.copy()

        raw_after[sid] = (X_after.copy(), WN_STD)

        boxes = _extract_patch_boxes(mask, PATCH_H, PATCH_W, PATCH_MIN_SELECTED_PIXELS)
        boxes = _select_patch_boxes(boxes, MAX_PATCHES_PER_ANIMAL, sid, side_name)

        patch_records = []
        for box in boxes:
            use = (
                (rr >= box["r0"]) & (rr < box["r1"]) &
                (cc >= box["c0"]) & (cc < box["c1"])
            )
            idx = np.where(use)[0]
            if idx.size < PATCH_MIN_SELECTED_PIXELS:
                continue

            X_patch = X_after[idx]
            if X_patch.size == 0:
                continue

            bdict = {}
            for bkey, (center, halfw) in BAND_LIBRARY.items():
                bdict[bkey] = _band_value_topk_mean_matrix(
                    X_patch, WN_STD, float(center), float(halfw), k=TOPK_PER_BAND
                )

            patch_id = f"{sid}_p{int(box['patch_idx']):03d}_r{int(box['r0']):03d}-{int(box['r1']):03d}_c{int(box['c0']):03d}-{int(box['c1']):03d}"
            patch_records.append({
                "patch_id": patch_id,
                "animal_id": sid,
                "label": 1 if sid in groups["KO"] else 0,
                "side": side_name,
                "n_pixels": int(idx.size),
                "box": box,
                "band_dict": bdict,
            })

        if patch_records:
            patch_cache[sid] = patch_records
            logger.info("%s %s: kept %d patches", side_name, sid, len(patch_records))
        else:
            logger.warning("%s %s: no valid patches kept", side_name, sid)

    logger.info("%s patch cache: %d/%d animals loaded", side_name, len(patch_cache), len(sample_names))
    return patch_cache, raw_before, raw_after

# ============================== SPLIT-FIRST FEATURES ==============================
def _compute_ratio_thresholds_training(train_band_dicts, ratio_defs):
    thr = {}
    for rname, cfg in ratio_defs.items():
        num_tuple = cfg["num"]; den_tuple = cfg["den"]
        num_key = next(k for k, v in BAND_LIBRARY.items() if v == num_tuple)
        den_key = next(k for k, v in BAND_LIBRARY.items() if v == den_tuple)

        all_num, all_den = [], []
        for bdict in train_band_dicts:
            num = np.asarray(bdict[num_key], float); den = np.asarray(bdict[den_key], float)
            num = num[np.isfinite(num)]; den = den[np.isfinite(den)]
            if num.size: all_num.append(num)
            if den.size: all_den.append(den)

        num_thr = max(np.nanpercentile(np.concatenate(all_num), NUM_MIN_PCT), NUM_ABS_FLOOR) if all_num else NUM_ABS_FLOOR
        den_thr = max(np.nanpercentile(np.concatenate(all_den), DENOM_MIN_PCT), DENOM_ABS_FLOOR) if all_den else DENOM_ABS_FLOOR
        thr[rname] = {"num_thr": float(num_thr), "den_thr": float(den_thr)}
    return thr

def build_features_for_sample(band_dict, thresholds=None):
    feats = {}
    if FEATURE_MODE == "bands_only":
        for bkey in BAND_LIBRARY.keys():
            feats.update(_summarize(band_dict[bkey], bkey))
        return feats

    if FEATURE_MODE == "band_ratios":
        if thresholds is None:
            raise ValueError("band_ratios requires training thresholds.")
        for rname, cfg in BASE_RATIOS.items():
            num_tuple = cfg["num"]; den_tuple = cfg["den"]
            num_key = next(k for k, v in BAND_LIBRARY.items() if v == num_tuple)
            den_key = next(k for k, v in BAND_LIBRARY.items() if v == den_tuple)

            num = np.asarray(band_dict[num_key], float)
            den = np.asarray(band_dict[den_key], float)

            tnum = thresholds[rname]["num_thr"]
            tden = thresholds[rname]["den_thr"]

            ok = np.isfinite(num) & np.isfinite(den) & (num > tnum) & (den > tden)

            rr = np.full_like(num, np.nan, dtype=float)
            rr[ok] = num[ok] / (den[ok] + EPS)
            if APPLY_RATIO_HARD_CAP:
                rr = np.clip(rr, 0, RATIO_HARD_CAP)
            rr = rr[np.isfinite(rr) & (rr > 0)]
            feats.update(_summarize(rr, _safe_name(rname)))
        return feats

    raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")

# ============================== CLEAN FIGURES ==============================
def plot_confusion_clean(cm, out_png, title, y_true, y_pred, include_n=True):
    _apply_pub_style()

    cm = np.asarray(cm, dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)

    footer = f"ACC: {_format_pct(acc)}   |   BACC: {_format_pct(bacc)}"
    if include_n:
        n = int(len(y_true))
        n0 = int(np.sum(np.asarray(y_true) == 0))
        n1 = int(np.sum(np.asarray(y_true) == 1))
        footer = f"N={n} (WT={n0}, KO={n1})   |   " + footer

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm_row, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Recall (row-normalized)")

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["WT", "KO"])
    ax.set_yticklabels(["WT", "KO"])
    ax.set_xlabel("Predicted", labelpad=10)
    ax.set_ylabel("True")
    ax.set_title(_short_title(title, 26))

    for i in range(2):
        for j in range(2):
            cnt = int(cm[i, j])
            rec = float(cm_row[i, j])
            txt = f"{cnt}\n({_format_pct(rec)})"
            color = "white" if rec > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=BASE_FONTSIZE + 2, color=color)

    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.text(0.5, 0.02, footer, ha="center", va="bottom", fontsize=BASE_FONTSIZE + 1)
    _savefig(fig, out_png)

def plot_roc_single(y_true, y_prob, out_png, title):
    _apply_pub_style()
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    m = np.isfinite(y_prob)
    if m.sum() < 3 or len(np.unique(y_true[m])) < 2:
        logger.warning("ROC skipped: insufficient probabilities.")
        return
    fpr, tpr, _ = roc_curve(y_true[m], y_prob[m])
    auc = roc_auc_score(y_true[m], y_prob[m])

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.step(fpr, tpr, where="post", lw=3, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(_short_title(title, 32))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out_png)

def plot_roc_mean_sd(fold_truth_probs, out_png, title):
    _apply_pub_style()

    fpr_grid = np.linspace(0.0, 1.0, 201)
    tprs = []
    aucs = []

    for y_t, y_p in fold_truth_probs:
        y_t = np.asarray(y_t, int)
        y_p = np.asarray(y_p, float)
        m = np.isfinite(y_p)
        if m.sum() < 3 or len(np.unique(y_t[m])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_t[m], y_p[m])
        aucs.append(roc_auc_score(y_t[m], y_p[m]))
        tpr_i = np.interp(fpr_grid, fpr, tpr)
        tpr_i[0] = 0.0
        tprs.append(tpr_i)

    if len(tprs) < 2:
        logger.warning("Mean ROC skipped: insufficient valid folds.")
        return

    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    sd_tpr = tprs.std(axis=0, ddof=1)
    mean_tpr[-1] = 1.0

    mean_auc = float(np.mean(aucs)) if len(aucs) else np.nan
    sd_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot(fpr_grid, mean_tpr, lw=3, label=f"Mean AUC = {mean_auc:.3f} ± {sd_auc:.3f}")
    ax.fill_between(fpr_grid,
                    np.clip(mean_tpr - sd_tpr, 0, 1),
                    np.clip(mean_tpr + sd_tpr, 0, 1),
                    alpha=0.18, linewidth=0)
    ax.plot([0, 1], [0, 1], "--", lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(_short_title(title, 32))
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out_png)

def plot_prob_violin(y_true, y_prob, out_png, title):
    _apply_pub_style()
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)

    wt = y_prob[y_true == 0]; wt = wt[np.isfinite(wt)]
    ko = y_prob[y_true == 1]; ko = ko[np.isfinite(ko)]

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    data = [wt, ko]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=False, showmedians=False, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_alpha(0.25)
        pc.set_edgecolor("black")
        pc.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    j_wt = (rng.random(wt.size) - 0.5) * 0.18
    j_ko = (rng.random(ko.size) - 0.5) * 0.18

    ax.scatter(np.zeros_like(wt) + 0 + j_wt, wt, s=28, alpha=0.6,
               edgecolors="black", linewidths=0.25, color=BASE_WT)
    ax.scatter(np.zeros_like(ko) + 1 + j_ko, ko, s=28, alpha=0.6,
               edgecolors="black", linewidths=0.25, color=BASE_KO)

    ax.axhline(0.5, linestyle="--", linewidth=2.0, color="black", alpha=0.55)

    def med_iqr(a):
        if a.size == 0:
            return np.nan, np.nan, np.nan
        q1, q3 = np.percentile(a, [25, 75])
        return float(np.median(a)), float(q1), float(q3)

    wt_med, wt_q1, wt_q3 = med_iqr(wt)
    ko_med, ko_q1, ko_q3 = med_iqr(ko)

    ax.text(0, 1.02, f"WT: med={wt_med:.2f}\nIQR=[{wt_q1:.2f},{wt_q3:.2f}]",
            ha="center", va="bottom", fontsize=BASE_FONTSIZE)
    ax.text(1, 1.02, f"KO: med={ko_med:.2f}\nIQR=[{ko_q1:.2f},{ko_q3:.2f}]",
            ha="center", va="bottom", fontsize=BASE_FONTSIZE)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["WT", "KO"])
    ax.set_ylabel("P(KO)")
    ax.set_ylim(-0.02, 1.08)
    ax.set_title(_short_title(title, 32))
    ax.grid(True, axis="y")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    _savefig(fig, out_png)

# ============================== SPECTRA FIGURES ==============================
def plot_median_spectra(raw_store, out_png, title):
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    for sid, (X, wn) in raw_store.items():
        if X.shape[0] > MAX_PIXELS_FOR_SPECTRA_PLOTS:
            rng = np.random.default_rng(abs(hash(("specplot", sid))) % (2**32))
            take = rng.choice(X.shape[0], size=MAX_PIXELS_FOR_SPECTRA_PLOTS, replace=False)
            Xp = X[take]
        else:
            Xp = X
        med = np.nanmedian(Xp, axis=0)
        color = BASE_KO if sid in groups["KO"] else BASE_WT
        ax.plot(wn, med, lw=1.6, alpha=0.75, color=color)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Median absorbance (a.u.)")
    ax.set_title(title)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_xaxis()
    fig.tight_layout()
    _savefig(fig, out_png)

def plot_group_mean_sd(raw_store, out_png, title):
    _apply_pub_style()

    wt_meds = []
    ko_meds = []
    wn_ref = None
    for sid, (X, wn) in raw_store.items():
        if wn_ref is None:
            wn_ref = wn
        if X.shape[0] > MAX_PIXELS_FOR_SPECTRA_PLOTS:
            rng = np.random.default_rng(abs(hash(("grpplot", sid))) % (2**32))
            take = rng.choice(X.shape[0], size=MAX_PIXELS_FOR_SPECTRA_PLOTS, replace=False)
            Xp = X[take]
        else:
            Xp = X
        med = np.nanmedian(Xp, axis=0)
        if sid in groups["WT"]:
            wt_meds.append(med)
        elif sid in groups["KO"]:
            ko_meds.append(med)

    wt_meds = np.vstack(wt_meds) if len(wt_meds) else np.zeros((0, WN_STD_Z))
    ko_meds = np.vstack(ko_meds) if len(ko_meds) else np.zeros((0, WN_STD_Z))

    def mean_sd(A):
        if A.shape[0] == 0:
            return np.full((WN_STD_Z,), np.nan), np.full((WN_STD_Z,), np.nan)
        m = np.nanmean(A, axis=0)
        sd = np.nanstd(A, axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(m)
        return m, sd

    wt_mean, wt_sd = mean_sd(wt_meds)
    ko_mean, ko_sd = mean_sd(ko_meds)

    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.plot(wn_ref, wt_mean, lw=3.0, color=BASE_WT, label=f"WT (n={wt_meds.shape[0]})")
    ax.plot(wn_ref, ko_mean, lw=3.0, color=BASE_KO, label=f"KO (n={ko_meds.shape[0]})")
    ax.fill_between(wn_ref, wt_mean-wt_sd, wt_mean+wt_sd, color=BASE_WT, alpha=0.20, linewidth=0)
    ax.fill_between(wn_ref, ko_mean-ko_sd, ko_mean+ko_sd, color=BASE_KO, alpha=0.20, linewidth=0)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Median spectrum (mean ± SD across samples)")
    ax.set_title(title)
    ax.grid(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    ax.invert_xaxis()
    fig.tight_layout()
    _savefig(fig, out_png)

# ============================== DATASET → MATRIX HELPERS ==============================
def _collect_patch_rows(cache, sids, thresholds):
    rows = []
    y = []
    patch_ids = []
    animal_ids = []

    for sid in sids:
        if sid not in cache:
            continue
        for rec in cache[sid]:
            feats = build_features_for_sample(rec["band_dict"], thresholds=thresholds)
            rows.append(feats)
            y.append(int(rec["label"]))
            patch_ids.append(rec["patch_id"])
            animal_ids.append(rec["animal_id"])

    cols = sorted(rows[0].keys()) if rows else []
    X = np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float) if rows else np.zeros((0,0))
    return X, np.array(y, dtype=int), cols, np.array(patch_ids, dtype=object), np.array(animal_ids, dtype=object)

def _predict_prob_ko(pipe, X):
    probs = np.full((X.shape[0],), np.nan, dtype=float)
    try:
        proba = pipe.predict_proba(X)
        classes_ = pipe.named_steps["clf"].classes_
        ko_pos = int(np.where(classes_ == 1)[0][0])
        probs = proba[:, ko_pos].astype(float)
        return probs
    except Exception:
        try:
            scores = pipe.decision_function(X).astype(float)
            probs = 1.0 / (1.0 + np.exp(-scores))
            return probs
        except Exception:
            return probs

def _animal_label(sid):
    return 1 if sid in groups["KO"] else 0

def _generate_balanced_holdout_splits(cache, n_train_per_class=5, n_test_per_class=2, n_repeats=None, random_state=123):
    wt = [sid for sid in groups["WT"] if sid in cache]
    ko = [sid for sid in groups["KO"] if sid in cache]

    if len(wt) < n_train_per_class + n_test_per_class:
        raise RuntimeError(f"Not enough WT animals in cache: have {len(wt)}")
    if len(ko) < n_train_per_class + n_test_per_class:
        raise RuntimeError(f"Not enough KO animals in cache: have {len(ko)}")

    wt_tests = list(combinations(wt, n_test_per_class))
    ko_tests = list(combinations(ko, n_test_per_class))
    all_pairs = [(tuple(sorted(wt_te)), tuple(sorted(ko_te))) for wt_te in wt_tests for ko_te in ko_tests]

    if n_repeats is not None and n_repeats < len(all_pairs):
        rng = np.random.default_rng(random_state)
        sel = rng.choice(len(all_pairs), size=n_repeats, replace=False)
        sel = np.sort(sel)
        all_pairs = [all_pairs[i] for i in sel]

    splits = []
    for split_id, (wt_te, ko_te) in enumerate(all_pairs, start=1):
        wt_te = list(wt_te)
        ko_te = list(ko_te)
        wt_tr = [sid for sid in wt if sid not in wt_te]
        ko_tr = [sid for sid in ko if sid not in ko_te]
        splits.append({
            "split_id": split_id,
            "train_sids": wt_tr + ko_tr,
            "test_sids": wt_te + ko_te,
            "train_wt": wt_tr,
            "train_ko": ko_tr,
            "test_wt": wt_te,
            "test_ko": ko_te,
        })
    return splits

def _aggregate_patch_predictions_to_animals(animal_ids, y_true_patch, y_prob_patch):
    animal_ids = np.asarray(animal_ids, dtype=object)
    y_true_patch = np.asarray(y_true_patch, dtype=int)
    y_prob_patch = np.asarray(y_prob_patch, dtype=float)

    uniq = []
    seen = set()
    for a in animal_ids:
        if a not in seen:
            uniq.append(a)
            seen.add(a)

    out_ids, out_y, out_prob, out_n = [], [], [], []
    for sid in uniq:
        m = animal_ids == sid
        probs = y_prob_patch[m]
        probs = probs[np.isfinite(probs)]
        if probs.size == 0:
            prob = np.nan
        else:
            prob = float(np.mean(probs))
        yt_vals = np.unique(y_true_patch[m])
        yt = int(yt_vals[0]) if yt_vals.size else _animal_label(sid)
        out_ids.append(sid)
        out_y.append(yt)
        out_prob.append(prob)
        out_n.append(int(np.sum(m)))

    out_ids = np.array(out_ids, dtype=object)
    out_y = np.array(out_y, dtype=int)
    out_prob = np.array(out_prob, dtype=float)
    out_pred = np.where(np.isfinite(out_prob), (out_prob >= 0.5).astype(int), 0).astype(int)
    out_n = np.array(out_n, dtype=int)
    return out_ids, out_y, out_pred, out_prob, out_n

def eval_repeated_balanced_holdout(train_cache, test_cache, n_train_per_class=5, n_test_per_class=2,
                                   n_repeats=None, random_state=123):
    common = [sid for sid in sample_names if sid in train_cache and sid in test_cache and (sid in groups["WT"] or sid in groups["KO"])]
    common_set = set(common)
    sub_train_cache = {sid: train_cache[sid] for sid in common if sid in train_cache}
    sub_test_cache  = {sid: test_cache[sid]  for sid in common if sid in test_cache}

    splits = _generate_balanced_holdout_splits(
        sub_train_cache, n_train_per_class=n_train_per_class, n_test_per_class=n_test_per_class,
        n_repeats=n_repeats, random_state=random_state
    )

    patch_split_ids = []
    patch_ids_all = []
    animal_ids_all = []
    y_true_patch_all = []
    y_pred_patch_all = []
    y_prob_patch_all = []

    animal_split_ids = []
    animal_ids_all2 = []
    y_true_an_all = []
    y_pred_an_all = []
    y_prob_an_all = []
    animal_npatch_all = []

    fold_truth_probs_patch = []
    fold_truth_probs_animal = []

    for sp in splits:
        split_id = int(sp["split_id"])
        train_sids = [sid for sid in sp["train_sids"] if sid in common_set]
        test_sids = [sid for sid in sp["test_sids"] if sid in common_set]

        thresholds = None
        if FEATURE_MODE == "band_ratios":
            train_band_dicts = [rec["band_dict"] for sid in train_sids for rec in sub_train_cache[sid]]
            thresholds = _compute_ratio_thresholds_training(train_band_dicts, BASE_RATIOS)

        Xtr, ytr, cols, patch_ids_tr, animal_ids_tr = _collect_patch_rows(sub_train_cache, train_sids, thresholds)
        Xte, yte, _, patch_ids_te, animal_ids_te = _collect_patch_rows(sub_test_cache, test_sids, thresholds)

        if Xtr.shape[0] == 0 or Xte.shape[0] == 0:
            logger.warning("Split %d skipped because train/test patch matrix is empty", split_id)
            continue
        if len(np.unique(ytr)) < 2:
            logger.warning("Split %d skipped because training set has only one class", split_id)
            continue

        pipe = _build_model(MODEL_NAME)
        pipe.fit(Xtr, ytr)

        pred_patch = pipe.predict(Xte).astype(int)
        prob_patch = _predict_prob_ko(pipe, Xte).astype(float)

        patch_split_ids.extend([split_id] * len(patch_ids_te))
        patch_ids_all.extend(patch_ids_te.tolist())
        animal_ids_all.extend(animal_ids_te.tolist())
        y_true_patch_all.extend(yte.tolist())
        y_pred_patch_all.extend(pred_patch.tolist())
        y_prob_patch_all.extend(prob_patch.tolist())
        fold_truth_probs_patch.append((yte.astype(int), prob_patch.astype(float)))

        an_ids, an_y, an_pred, an_prob, an_n = _aggregate_patch_predictions_to_animals(animal_ids_te, yte, prob_patch)
        animal_split_ids.extend([split_id] * len(an_ids))
        animal_ids_all2.extend(an_ids.tolist())
        y_true_an_all.extend(an_y.tolist())
        y_pred_an_all.extend(an_pred.tolist())
        y_prob_an_all.extend(an_prob.tolist())
        animal_npatch_all.extend(an_n.tolist())
        fold_truth_probs_animal.append((an_y.astype(int), an_prob.astype(float)))

    patch_results = {
        "ids": np.array(patch_ids_all, dtype=object),
        "animal_ids": np.array(animal_ids_all, dtype=object),
        "split_ids": np.array(patch_split_ids, dtype=int),
        "y_true": np.array(y_true_patch_all, dtype=int),
        "y_pred": np.array(y_pred_patch_all, dtype=int),
        "y_prob": np.array(y_prob_patch_all, dtype=float),
        "fold_truth_probs": fold_truth_probs_patch,
    }

    animal_results = {
        "ids": np.array(animal_ids_all2, dtype=object),
        "n_patches": np.array(animal_npatch_all, dtype=int),
        "split_ids": np.array(animal_split_ids, dtype=int),
        "y_true": np.array(y_true_an_all, dtype=int),
        "y_pred": np.array(y_pred_an_all, dtype=int),
        "y_prob": np.array(y_prob_an_all, dtype=float),
        "fold_truth_probs": fold_truth_probs_animal,
    }
    return patch_results, animal_results

# ============================== SAVE CSV + REPORT ==============================
def save_predictions_csv(path, ids, y_true, y_pred, y_prob, fold_ids=None, animal_ids=None, n_patches=None):
    with open(path, "w", encoding="utf-8") as f:
        cols = []
        if animal_ids is not None:
            cols.append("animal")
        cols.append("id")
        if fold_ids is not None:
            cols.append("split")
        if n_patches is not None:
            cols.append("n_patches")
        cols += ["true_label", "pred_label", "prob_KO"]
        f.write(",".join(cols) + "\n")

        for i in range(len(ids)):
            row = []
            if animal_ids is not None:
                row.append(str(animal_ids[i]))
            row.append(str(ids[i]))
            if fold_ids is not None:
                row.append(str(int(fold_ids[i])))
            if n_patches is not None:
                row.append(str(int(n_patches[i])))
            t = "KO" if int(y_true[i]) == 1 else "WT"
            pr = "KO" if int(y_pred[i]) == 1 else "WT"
            ps = "" if not np.isfinite(y_prob[i]) else f"{float(y_prob[i]):.8f}"
            row += [t, pr, ps]
            f.write(",".join(row) + "\n")

    logger.info("Saved predictions: %s", path)

def report_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])

    auc = np.nan
    m = np.isfinite(y_prob)
    if m.sum() >= 3 and len(np.unique(np.asarray(y_true)[m])) == 2:
        auc = roc_auc_score(np.asarray(y_true)[m], np.asarray(y_prob)[m])

    print("\n====================", name, "====================")
    print("Feature mode:       ", FEATURE_MODE)
    print("Model:              ", MODEL_NAME)
    print("n_samples:          ", len(y_true))
    print("Accuracy:           ", f"{acc:.4f}")
    print("Balanced accuracy:  ", f"{bacc:.4f}")
    print("AUC (KO prob):      ", (f"{auc:.4f}" if np.isfinite(auc) else "nan"))
    print("Confusion matrix [WT, KO]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["WT","KO"], digits=4))
    return float(acc), float(bacc), float(auc) if np.isfinite(auc) else np.nan, cm

# ============================== INTERPRETABILITY (PATCH FULL-FIT) ==============================
def fit_full_train_and_plot_features(train_cache, out_dir, side_label="RIGHT"):
    all_sids = [sid for sid in sample_names if sid in train_cache]
    thresholds = None
    if FEATURE_MODE == "band_ratios":
        all_band_dicts = [rec["band_dict"] for sid in all_sids for rec in train_cache[sid]]
        thresholds = _compute_ratio_thresholds_training(all_band_dicts, BASE_RATIOS)

    X, y, cols, patch_ids, animal_ids = _collect_patch_rows(train_cache, all_sids, thresholds)
    if X.shape[0] == 0:
        logger.warning("Interpretability skipped for %s: empty dataset", side_label)
        return

    pipe = _build_model(MODEL_NAME)
    pipe.fit(X, y)

    clf = pipe.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coefs = np.ravel(clf.coef_)
        order = np.argsort(np.abs(coefs))[::-1]
        top = order[:min(25, len(order))]
        names = [cols[i] for i in top]
        vals = coefs[top]

        _apply_pub_style()
        fig_h = max(5, 0.28 * len(top) + 1.6)
        fig, ax = plt.subplots(figsize=(9.2, fig_h))
        y_pos = np.arange(len(top))[::-1]
        bar_colors = [BASE_KO if v > 0 else BASE_WT for v in vals[::-1]]
        ax.barh(y_pos, vals[::-1], edgecolor="black", linewidth=0.5, color=bar_colors)
        ax.set_yticks(y_pos, names[::-1])
        ax.axvline(0, color="black", linewidth=1.2)
        ax.set_xlabel("Coefficient (positive → KO)")
        ax.set_title(f"Top coefficients (ALL PATCHES {side_label}; {MODEL_NAME}, {FEATURE_MODE})")
        ax.grid(True, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"feature_coefficients_fullfit_{side_label}.png"))

# ============================== RUN HELPERS ==============================
def run_and_save_block(out_dir, tag, ids, y_true, y_pred, y_prob,
                       fold_ids=None, fold_truth_probs=None, animal_ids=None, n_patches=None):
    os.makedirs(out_dir, exist_ok=True)

    save_predictions_csv(
        os.path.join(out_dir, "predictions.csv"),
        ids, y_true, y_pred, y_prob,
        fold_ids=fold_ids, animal_ids=animal_ids, n_patches=n_patches
    )
    acc, bacc, auc, cm = report_metrics(tag, y_true, y_pred, y_prob)

    plot_confusion_clean(cm, os.path.join(out_dir, "confusion_pub.png"),
                         title=tag, y_true=y_true, y_pred=y_pred, include_n=True)

    if fold_truth_probs is None:
        plot_roc_single(y_true, y_prob, os.path.join(out_dir, "roc_pub.png"), title=tag)
    else:
        plot_roc_mean_sd(fold_truth_probs, os.path.join(out_dir, "roc_pub.png"), title=tag)

    plot_prob_violin(y_true, y_prob, os.path.join(out_dir, "prob_violin_pub.png"), title=tag)

    return {"acc": acc, "bacc": bacc, "auc": auc, "cm": cm, "out_dir": out_dir}

# ============================== MAIN ==============================
if __name__ == "__main__":
    logger.info("SAVE_DIR: %s", SAVE_DIR)
    logger.info("Scatter correction: %s | backend=%s", APPLY_SCATTER_CORRECTION, SCATTER_BACKEND)
    logger.info("BG subtraction: %s", APPLY_BG_SUBTRACTION)
    logger.info("RMieSC subset=%d tries=%d clip=%s", RMIESC_SUBSET_N, RMIESC_MAX_TRIES, str(RMIESC_CLIP))
    logger.info("FEATURE_MODE=%s | MODEL=%s", FEATURE_MODE, MODEL_NAME)
    logger.info("Patch size = %dx%d | min selected pixels = %d | max patches/animal = %s",
                PATCH_H, PATCH_W, PATCH_MIN_SELECTED_PIXELS, str(MAX_PATCHES_PER_ANIMAL))
    logger.info("Animal split = %d train/class, %d test/class | repeats=%s",
                TRAIN_ANIMALS_PER_CLASS, TEST_ANIMALS_PER_CLASS, str(N_REPEATS_PER_SIDE))

    with LogTimer("Build RIGHT patch cache (BG + resample + BEFORE/AFTER RMieSC)"):
        right_cache, right_raw_before, right_raw_after = build_side_cache(
            "RIGHT", RIGHT_CLUSTER_DIR, RIGHT_CUBE_DIR, RIGHT_CUBE_PATTERN, BG_DIR_RIGHT
        )

    _OCTAVVS_FAILS = 0
    _OCTAVVS_LOGGED_FIRST = False

    with LogTimer("Build LEFT patch cache (BG + resample + BEFORE/AFTER RMieSC)"):
        left_cache, left_raw_before, left_raw_after = build_side_cache(
            "LEFT", LEFT_CLUSTER_DIR, LEFT_CUBE_DIR, LEFT_CUBE_PATTERN, BG_DIR_LEFT
        )

    fig_dir_specs = os.path.join(SAVE_DIR, "Spectra_Preprocessed")
    fig_dir_before = os.path.join(fig_dir_specs, "Before_RMieSC")
    fig_dir_after  = os.path.join(fig_dir_specs, "After_RMieSC")
    os.makedirs(fig_dir_before, exist_ok=True)
    os.makedirs(fig_dir_after, exist_ok=True)

    plot_median_spectra(right_raw_before, os.path.join(fig_dir_before, "RIGHT_median_spectra_all_samples.png"),
                        "RIGHT: median spectra BEFORE RMieSC (BG + resample)")
    plot_median_spectra(left_raw_before, os.path.join(fig_dir_before, "LEFT_median_spectra_all_samples.png"),
                        "LEFT: median spectra BEFORE RMieSC (BG + resample)")
    plot_group_mean_sd(right_raw_before, os.path.join(fig_dir_before, "RIGHT_group_mean_sd.png"),
                       "RIGHT: WT vs KO mean ± SD BEFORE RMieSC (sample medians)")
    plot_group_mean_sd(left_raw_before, os.path.join(fig_dir_before, "LEFT_group_mean_sd.png"),
                       "LEFT: WT vs KO mean ± SD BEFORE RMieSC (sample medians)")

    plot_median_spectra(right_raw_after, os.path.join(fig_dir_after, "RIGHT_median_spectra_all_samples.png"),
                        "RIGHT: median spectra AFTER RMieSC (BG + resample + RMieSC)")
    plot_median_spectra(left_raw_after, os.path.join(fig_dir_after, "LEFT_median_spectra_all_samples.png"),
                        "LEFT: median spectra AFTER RMieSC (BG + resample + RMieSC)")
    plot_group_mean_sd(right_raw_after, os.path.join(fig_dir_after, "RIGHT_group_mean_sd.png"),
                       "RIGHT: WT vs KO mean ± SD AFTER RMieSC (sample medians)")
    plot_group_mean_sd(left_raw_after, os.path.join(fig_dir_after, "LEFT_group_mean_sd.png"),
                       "LEFT: WT vs KO mean ± SD AFTER RMieSC (sample medians)")

    results = {}

    # ---------------- A) RIGHT→RIGHT ----------------
    baseA = os.path.join(SAVE_DIR, "A_TRAIN_RIGHT__TEST_RIGHT")
    with LogTimer("A RIGHT→RIGHT repeated balanced 5v2 holdout (patch-wise)"):
        patch_res, animal_res = eval_repeated_balanced_holdout(
            right_cache, right_cache,
            n_train_per_class=TRAIN_ANIMALS_PER_CLASS,
            n_test_per_class=TEST_ANIMALS_PER_CLASS,
            n_repeats=N_REPEATS_PER_SIDE,
            random_state=SPLIT_RANDOM_SEED
        )
    results["A_PATCH"] = run_and_save_block(
        out_dir=os.path.join(baseA, "PatchLevel"),
        tag="R→R patch-level (5v2 animal holdout)",
        ids=patch_res["ids"], y_true=patch_res["y_true"], y_pred=patch_res["y_pred"], y_prob=patch_res["y_prob"],
        fold_ids=patch_res["split_ids"], fold_truth_probs=patch_res["fold_truth_probs"], animal_ids=patch_res["animal_ids"]
    )
    results["A_ANIMAL"] = run_and_save_block(
        out_dir=os.path.join(baseA, "AnimalLevel"),
        tag="R→R animal-level (5v2 animal holdout)",
        ids=animal_res["ids"], y_true=animal_res["y_true"], y_pred=animal_res["y_pred"], y_prob=animal_res["y_prob"],
        fold_ids=animal_res["split_ids"], fold_truth_probs=animal_res["fold_truth_probs"], n_patches=animal_res["n_patches"]
    )

    # ---------------- B) RIGHT→LEFT ----------------
    baseB = os.path.join(SAVE_DIR, "B_TRAIN_RIGHT__TEST_LEFT")
    with LogTimer("B RIGHT→LEFT repeated balanced 5v2 holdout (patch-wise)"):
        patch_res, animal_res = eval_repeated_balanced_holdout(
            right_cache, left_cache,
            n_train_per_class=TRAIN_ANIMALS_PER_CLASS,
            n_test_per_class=TEST_ANIMALS_PER_CLASS,
            n_repeats=N_REPEATS_PER_SIDE,
            random_state=SPLIT_RANDOM_SEED
        )
    results["B_PATCH"] = run_and_save_block(
        out_dir=os.path.join(baseB, "PatchLevel"),
        tag="R→L patch-level (5v2 animal holdout)",
        ids=patch_res["ids"], y_true=patch_res["y_true"], y_pred=patch_res["y_pred"], y_prob=patch_res["y_prob"],
        fold_ids=patch_res["split_ids"], fold_truth_probs=patch_res["fold_truth_probs"], animal_ids=patch_res["animal_ids"]
    )
    results["B_ANIMAL"] = run_and_save_block(
        out_dir=os.path.join(baseB, "AnimalLevel"),
        tag="R→L animal-level (5v2 animal holdout)",
        ids=animal_res["ids"], y_true=animal_res["y_true"], y_pred=animal_res["y_pred"], y_prob=animal_res["y_prob"],
        fold_ids=animal_res["split_ids"], fold_truth_probs=animal_res["fold_truth_probs"], n_patches=animal_res["n_patches"]
    )

    # ---------------- C) LEFT→LEFT ----------------
    baseC = os.path.join(SAVE_DIR, "C_TRAIN_LEFT__TEST_LEFT")
    with LogTimer("C LEFT→LEFT repeated balanced 5v2 holdout (patch-wise)"):
        patch_res, animal_res = eval_repeated_balanced_holdout(
            left_cache, left_cache,
            n_train_per_class=TRAIN_ANIMALS_PER_CLASS,
            n_test_per_class=TEST_ANIMALS_PER_CLASS,
            n_repeats=N_REPEATS_PER_SIDE,
            random_state=SPLIT_RANDOM_SEED
        )
    results["C_PATCH"] = run_and_save_block(
        out_dir=os.path.join(baseC, "PatchLevel"),
        tag="L→L patch-level (5v2 animal holdout)",
        ids=patch_res["ids"], y_true=patch_res["y_true"], y_pred=patch_res["y_pred"], y_prob=patch_res["y_prob"],
        fold_ids=patch_res["split_ids"], fold_truth_probs=patch_res["fold_truth_probs"], animal_ids=patch_res["animal_ids"]
    )
    results["C_ANIMAL"] = run_and_save_block(
        out_dir=os.path.join(baseC, "AnimalLevel"),
        tag="L→L animal-level (5v2 animal holdout)",
        ids=animal_res["ids"], y_true=animal_res["y_true"], y_pred=animal_res["y_pred"], y_prob=animal_res["y_prob"],
        fold_ids=animal_res["split_ids"], fold_truth_probs=animal_res["fold_truth_probs"], n_patches=animal_res["n_patches"]
    )

    # ---------------- D) LEFT→RIGHT ----------------
    baseD = os.path.join(SAVE_DIR, "D_TRAIN_LEFT__TEST_RIGHT")
    with LogTimer("D LEFT→RIGHT repeated balanced 5v2 holdout (patch-wise)"):
        patch_res, animal_res = eval_repeated_balanced_holdout(
            left_cache, right_cache,
            n_train_per_class=TRAIN_ANIMALS_PER_CLASS,
            n_test_per_class=TEST_ANIMALS_PER_CLASS,
            n_repeats=N_REPEATS_PER_SIDE,
            random_state=SPLIT_RANDOM_SEED
        )
    results["D_PATCH"] = run_and_save_block(
        out_dir=os.path.join(baseD, "PatchLevel"),
        tag="L→R patch-level (5v2 animal holdout)",
        ids=patch_res["ids"], y_true=patch_res["y_true"], y_pred=patch_res["y_pred"], y_prob=patch_res["y_prob"],
        fold_ids=patch_res["split_ids"], fold_truth_probs=patch_res["fold_truth_probs"], animal_ids=patch_res["animal_ids"]
    )
    results["D_ANIMAL"] = run_and_save_block(
        out_dir=os.path.join(baseD, "AnimalLevel"),
        tag="L→R animal-level (5v2 animal holdout)",
        ids=animal_res["ids"], y_true=animal_res["y_true"], y_pred=animal_res["y_pred"], y_prob=animal_res["y_prob"],
        fold_ids=animal_res["split_ids"], fold_truth_probs=animal_res["fold_truth_probs"], n_patches=animal_res["n_patches"]
    )

    fig_dir_feat_R = os.path.join(SAVE_DIR, "Features_Interpretability_FULLFIT_RIGHT")
    os.makedirs(fig_dir_feat_R, exist_ok=True)
    with LogTimer("Fit full RIGHT patch model for interpretability-only plots"):
        fit_full_train_and_plot_features(right_cache, fig_dir_feat_R, side_label="RIGHT")

    fig_dir_feat_L = os.path.join(SAVE_DIR, "Features_Interpretability_FULLFIT_LEFT")
    os.makedirs(fig_dir_feat_L, exist_ok=True)
    with LogTimer("Fit full LEFT patch model for interpretability-only plots"):
        fit_full_train_and_plot_features(left_cache, fig_dir_feat_L, side_label="LEFT")

    print("\n==================== SUMMARY ====================")

    for k, r in results.items():
        auc_str = "nan" if not np.isfinite(r["auc"]) else f"{r['auc']:.4f}"
        print(f"{k:10s} | acc={r['acc']:.4f}  bacc={r['bacc']:.4f}  auc={auc_str}  | {r['out_dir']}")