# ======================================================================
# WT vs KO FTIR classification (SPLIT-FIRST, LEAKAGE-REDUCED) — SAMPLE-WISE
#
# RUNS 4 RESULT SETS:
#   (A) Train RIGHT -> Test RIGHT   : LOOCV + K-fold CV (auto-capped)
#   (B) Train RIGHT -> Test LEFT    : TrainAll/TestAll
#   (C) Train LEFT  -> Test LEFT    : LOOCV + K-fold CV (auto-capped)
#   (D) Train LEFT  -> Test RIGHT   : TrainAll/TestAll
#
# FULL SCRIPT + RMieSC RANDOM-RETRY + SPECTRA PLOTS (BEFORE + AFTER RMieSC)
# + NumPy compatibility fix (NO np.asarray(copy=...)).
#
# Pipeline per sample:
#   Raw cube -> (pixel cap) -> BG subtract -> resample -> SAVE "BEFORE"
#   -> RMieSC (subset+retry) -> SAVE "AFTER" -> bands/ratios -> ML
#
# Outputs:
#   Spectra_Preprocessed/Before_RMieSC: median + mean±SD
#   Spectra_Preprocessed/After_RMieSC:  median + mean±SD
#   A/B/C/D folders:
#       LOOCV (for RR and LL): confusion/ROC/violin + predictions.csv
#       CV    (for RR and LL): confusion/meanROC/violin + predictions.csv
#       TrainTest (for RL and LR): confusion/ROC/violin + predictions.csv
#   Features_Interpretability_FULLFIT_RIGHT: coef bar + feature heatmap
#   Features_Interpretability_FULLFIT_LEFT:  coef bar + feature heatmap
#
# Notes:
#   - No baseline correction
#   - No normalization
#   - Ratio thresholds computed from TRAIN fold only (split-first)
#   - RMieSC returns corrected SUBSET (RMIESC_SUBSET_N) when enabled/succeeds,
#     so AFTER plots may use fewer pixels than BEFORE plots.
# ======================================================================

from __future__ import annotations

import os, time, logging, traceback, importlib, pkgutil, inspect
from contextlib import contextmanager
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
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

SAVE_DIR = os.path.join(RIGHT_CUBE_DIR, "WT_KO_ML_SAMPLEWISE_FULLFIGS__v1")
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

# ---- Pixel caps ----
MAX_PIXELS_PER_SAMPLE_ANALYSIS = 3000
MAX_PIXELS_FOR_SPECTRA_PLOTS   = 3000

# ---- CV settings (sample-wise) ----
# NOTE: 20-fold is impossible with 14 samples; we auto-cap to valid max.
CV_FOLDS_REQUESTED = 20
CV_SHUFFLE = True
CV_RANDOM_STATE = 123

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
    return s if len(s) <= maxlen else (s[:maxlen-1] + "…")

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

# ============================== SIDE CACHE ==============================
def build_side_cache(side_name, cluster_dir, cube_dir, cube_pattern, bg_dir):
    """
    Returns:
      band_cache: dict[sid] -> per-band arrays used for ML
      raw_before: dict[sid] -> (X_before_rmiesc, wn)
      raw_after:  dict[sid] -> (X_after_rmiesc,  wn)
    """
    band_cache = {}
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
            continue

        rr, cc = np.where(mask)
        logger.info("%s %s: selected pixels before cap = %d", side_name, sid, rr.size)

        X = cube[rr, cc, :].astype(np.float64, copy=False)

        if X.shape[0] > MAX_PIXELS_PER_SAMPLE_ANALYSIS:
            rng = np.random.default_rng(abs(hash((side_name, sid))) % (2**32))
            take = rng.choice(X.shape[0], size=MAX_PIXELS_PER_SAMPLE_ANALYSIS, replace=False)
            X = X[take]

        logger.info("%s %s: pixels used after cap = %d", side_name, sid, X.shape[0])

        if PRE_BG_SHIFT is not None and float(PRE_BG_SHIFT) != 0.0:
            X = X + float(PRE_BG_SHIFT)

        X = _bg_subtract(X, Z_src, sid, bg_dir)

        if X.shape[1] != WN_STD_Z:
            X = _resample_to_std(X, Z_src)

        X_before = X.astype(np.float64, copy=True)
        raw_before[sid] = (X_before, WN_STD)

        X_after = apply_rmiesc(X, WN_STD, sid=sid, side=side_name)
        X_after_copy = np.asarray(X_after, dtype=np.float64).copy()
        raw_after[sid] = (X_after_copy, WN_STD)

        bdict = {}
        for bkey, (center, halfw) in BAND_LIBRARY.items():
            bdict[bkey] = _band_value_topk_mean_matrix(X_after, WN_STD, float(center), float(halfw), k=TOPK_PER_BAND)
        band_cache[sid] = bdict

    logger.info("%s cache: %d/%d samples loaded", side_name, len(band_cache), len(sample_names))
    return band_cache, raw_before, raw_after

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

# ============================== CLEAN FIGURES (batch-style look) ==============================
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

# ============================== SPECTRA FIGURES (unchanged logic) ==============================
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
def _make_dataset_from_sids(cache, sids, thresholds):
    rows = []
    y = []
    for sid in sids:
        feats = build_features_for_sample(cache[sid], thresholds=thresholds)
        rows.append(feats)
        y.append(1 if sid in groups["KO"] else 0)

    cols = sorted(rows[0].keys()) if rows else []
    X = np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float) if rows else np.zeros((0,0))
    return X, np.array(y, dtype=int), cols

def _predict_prob_ko(pipe, X):
    # X shape (n, d)
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

# ============================== EVALS (sample-wise) ==============================
def eval_loocv_samplewise(train_cache, test_cache):
    valid = [sid for sid in sample_names
             if (sid in train_cache) and (sid in test_cache)
             and (sid in groups["WT"] or sid in groups["KO"])]

    y_all = np.array([1 if sid in groups["KO"] else 0 for sid in valid], dtype=int)
    s_all = np.array(valid, dtype=object)

    loo = LeaveOneOut()
    y_true, y_pred, y_prob, sid_out = [], [], [], []

    for tr_idx, te_idx in loo.split(s_all, y_all):
        train_sids = s_all[tr_idx]
        test_sid = s_all[te_idx][0]
        yte = int(y_all[te_idx][0])

        thresholds = None
        if FEATURE_MODE == "band_ratios":
            thresholds = _compute_ratio_thresholds_training([train_cache[s] for s in train_sids], BASE_RATIOS)

        Xtr, ytr, cols = _make_dataset_from_sids(train_cache, train_sids, thresholds)
        Xte, yte_arr, _ = _make_dataset_from_sids(test_cache, [test_sid], thresholds)

        pipe = _build_model(MODEL_NAME)
        pipe.fit(Xtr, ytr)

        pred = int(pipe.predict(Xte)[0])
        prob = float(_predict_prob_ko(pipe, Xte)[0])

        sid_out.append(test_sid)
        y_true.append(yte)
        y_pred.append(pred)
        y_prob.append(prob)

    return np.array(sid_out, object), np.array(y_true, int), np.array(y_pred, int), np.array(y_prob, float)

def _max_valid_folds_for_samplewise(sids):
    y = np.array([1 if s in groups["KO"] else 0 for s in sids], dtype=int)
    c0 = int(np.sum(y == 0))
    c1 = int(np.sum(y == 1))
    return max(2, min(c0, c1))  # must be >=2 to be meaningful

def eval_kfold_samplewise(cache, n_splits_requested, shuffle=True, random_state=123):
    sids = [sid for sid in sample_names if sid in cache and (sid in groups["WT"] or sid in groups["KO"])]
    y = np.array([1 if sid in groups["KO"] else 0 for sid in sids], dtype=int)

    max_folds = _max_valid_folds_for_samplewise(sids)
    n_splits = int(min(n_splits_requested, max_folds))
    if n_splits < 2:
        raise RuntimeError("Not enough samples per class for K-fold CV.")

    if n_splits != n_splits_requested:
        logger.warning("Requested CV_FOLDS=%d but only %d-fold is valid sample-wise (class-limited). Using %d.",
                       n_splits_requested, max_folds, n_splits)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    sid_out, y_true, y_pred, y_prob, fold_ids = [], [], [], [], []
    fold_truth_probs = []

    idx = np.arange(len(sids))
    fold_num = 0
    for tr_i, te_i in skf.split(idx, y):
        fold_num += 1
        tr_sids = [sids[i] for i in tr_i]
        te_sids = [sids[i] for i in te_i]

        thresholds = None
        if FEATURE_MODE == "band_ratios":
            thresholds = _compute_ratio_thresholds_training([cache[s] for s in tr_sids], BASE_RATIOS)

        Xtr, ytr, cols = _make_dataset_from_sids(cache, tr_sids, thresholds)
        Xte, yte, _ = _make_dataset_from_sids(cache, te_sids, thresholds)

        pipe = _build_model(MODEL_NAME)
        pipe.fit(Xtr, ytr)

        pred = pipe.predict(Xte).astype(int)
        prob = _predict_prob_ko(pipe, Xte).astype(float)

        sid_out.extend(te_sids)
        y_true.extend(yte.tolist())
        y_pred.extend(pred.tolist())
        y_prob.extend(prob.tolist())
        fold_ids.extend([fold_num] * len(te_sids))

        fold_truth_probs.append((yte.astype(int), prob.astype(float)))

    return (np.array(sid_out, object),
            np.array(fold_ids, int),
            np.array(y_true, int),
            np.array(y_pred, int),
            np.array(y_prob, float),
            fold_truth_probs,
            n_splits)

def eval_trainall_testall(train_cache, test_cache):
    train_sids = [sid for sid in sample_names if sid in train_cache and (sid in groups["WT"] or sid in groups["KO"])]
    test_sids  = [sid for sid in sample_names if sid in test_cache  and (sid in groups["WT"] or sid in groups["KO"])]

    thresholds = None
    if FEATURE_MODE == "band_ratios":
        thresholds = _compute_ratio_thresholds_training([train_cache[s] for s in train_sids], BASE_RATIOS)

    Xtr, ytr, cols = _make_dataset_from_sids(train_cache, train_sids, thresholds)
    Xte, yte, _ = _make_dataset_from_sids(test_cache, test_sids, thresholds)

    pipe = _build_model(MODEL_NAME)
    pipe.fit(Xtr, ytr)

    pred = pipe.predict(Xte).astype(int)
    prob = _predict_prob_ko(pipe, Xte).astype(float)

    return np.array(test_sids, object), yte.astype(int), pred.astype(int), prob.astype(float)

# ============================== SAVE CSV + REPORT ==============================
def save_predictions_csv(path, ids, y_true, y_pred, y_prob, fold_ids=None):
    with open(path, "w", encoding="utf-8") as f:
        if fold_ids is None:
            f.write("sample,true_label,pred_label,prob_KO\n")
            for s, yt, yp, p in zip(ids, y_true, y_pred, y_prob):
                t = "KO" if int(yt) == 1 else "WT"
                pr = "KO" if int(yp) == 1 else "WT"
                ps = "" if not np.isfinite(p) else f"{float(p):.8f}"
                f.write(f"{s},{t},{pr},{ps}\n")
        else:
            f.write("sample,fold,true_label,pred_label,prob_KO\n")
            for s, fd, yt, yp, p in zip(ids, fold_ids, y_true, y_pred, y_prob):
                t = "KO" if int(yt) == 1 else "WT"
                pr = "KO" if int(yp) == 1 else "WT"
                ps = "" if not np.isfinite(p) else f"{float(p):.8f}"
                f.write(f"{s},{int(fd)},{t},{pr},{ps}\n")
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

# ============================== INTERPRETABILITY (unchanged) ==============================
def fit_full_train_and_plot_features(train_cache, out_dir, side_label="RIGHT"):
    thresholds = None
    if FEATURE_MODE == "band_ratios":
        thresholds = _compute_ratio_thresholds_training([train_cache[sid] for sid in train_cache.keys()], BASE_RATIOS)

    rows, y, samples = [], [], []
    for sid in train_cache.keys():
        feats = build_features_for_sample(train_cache[sid], thresholds=thresholds)
        rows.append(feats)
        y.append(1 if sid in groups["KO"] else 0)
        samples.append(sid)

    cols = sorted(rows[0].keys())
    X = np.array([[r.get(c, np.nan) for c in cols] for r in rows], dtype=float)
    y = np.array(y, dtype=int)
    samples = np.array(samples, dtype=object)

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
        ax.set_title(f"Top coefficients (ALL {side_label}; {MODEL_NAME}, {FEATURE_MODE})")
        ax.grid(True, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        _savefig(fig, os.path.join(out_dir, f"feature_coefficients_fullfit_{side_label}.png"))

    # heatmap
    Xv = X.copy()
    for j in range(Xv.shape[1]):
        col = Xv[:, j]
        med = np.nanmedian(col)
        col[np.isnan(col)] = med
        Xv[:, j] = col
    mu = Xv.mean(axis=0, keepdims=True)
    sd = Xv.std(axis=0, keepdims=True) + 1e-12
    Z = (Xv - mu) / sd

    order_s = np.argsort(np.array([0 if s in groups["WT"] else 1 for s in samples]) * 100 + np.arange(len(samples)))
    samples2 = samples[order_s]
    Z2 = Z[order_s]

    if hasattr(pipe.named_steps["clf"], "coef_"):
        coefs_full = np.ravel(pipe.named_steps["clf"].coef_)
        feat_order = np.argsort(np.abs(coefs_full))[::-1]
    else:
        feat_order = np.argsort(Z2.var(axis=0))[::-1]
    feat_order = feat_order[:min(40, Z2.shape[1])]
    Zp = Z2[:, feat_order]
    colnames = [cols[i] for i in feat_order]

    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    im = ax.imshow(Zp, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="z-score")

    ax.set_yticks(np.arange(len(samples2)))
    ax.set_yticklabels(samples2)
    ax.set_xticks(np.arange(len(colnames)))
    ax.set_xticklabels(colnames, rotation=60, ha="right")
    ax.set_title(f"Feature heatmap (z-scored; ALL {side_label}; interpretability only)")
    ax.set_xlabel("Features")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    _savefig(fig, os.path.join(out_dir, f"feature_heatmap_fullfit_{side_label}.png"))

# ============================== RUN HELPERS ==============================
def run_and_save_block(out_dir, tag, ids, y_true, y_pred, y_prob, fold_ids=None, fold_truth_probs=None):
    os.makedirs(out_dir, exist_ok=True)

    save_predictions_csv(os.path.join(out_dir, "predictions.csv"), ids, y_true, y_pred, y_prob, fold_ids=fold_ids)
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
    logger.info("Requested CV folds (sample-wise) = %d", CV_FOLDS_REQUESTED)

    with LogTimer("Build RIGHT cache (BG + resample + BEFORE/AFTER RMieSC)"):
        right_cache, right_raw_before, right_raw_after = build_side_cache(
            "RIGHT", RIGHT_CLUSTER_DIR, RIGHT_CUBE_DIR, RIGHT_CUBE_PATTERN, BG_DIR_RIGHT
        )

    _OCTAVVS_FAILS = 0
    _OCTAVVS_LOGGED_FIRST = False

    with LogTimer("Build LEFT cache (BG + resample + BEFORE/AFTER RMieSC)"):
        left_cache, left_raw_before, left_raw_after = build_side_cache(
            "LEFT", LEFT_CLUSTER_DIR, LEFT_CUBE_DIR, LEFT_CUBE_PATTERN, BG_DIR_LEFT
        )

    # ---- spectra figures BEFORE/AFTER RMieSC ----
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

    # ---------------- A) RIGHT→RIGHT : LOOCV + K-fold ----------------
    baseA = os.path.join(SAVE_DIR, "A_TRAIN_RIGHT__TEST_RIGHT")

    with LogTimer("A RIGHT→RIGHT LOOCV (sample-wise)"):
        ids, yt, yp, pr = eval_loocv_samplewise(right_cache, right_cache)
    results["A_LOOCV"] = run_and_save_block(
        out_dir=os.path.join(baseA, "LOOCV"),
        tag="R→R LOOCV",
        ids=ids, y_true=yt, y_pred=yp, y_prob=pr
    )

    with LogTimer("A RIGHT→RIGHT K-fold CV (sample-wise)"):
        ids2, folds2, yt2, yp2, pr2, fold_tp, k_used = eval_kfold_samplewise(
            right_cache, CV_FOLDS_REQUESTED, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE
        )
    results["A_CV"] = run_and_save_block(
        out_dir=os.path.join(baseA, f"CV_{k_used}fold"),
        tag=f"R→R {k_used}-fold",
        ids=ids2, y_true=yt2, y_pred=yp2, y_prob=pr2,
        fold_ids=folds2, fold_truth_probs=fold_tp
    )

    # ---------------- B) RIGHT→LEFT : TrainAll/TestAll ----------------
    baseB = os.path.join(SAVE_DIR, "B_TRAIN_RIGHT__TEST_LEFT")
    with LogTimer("B RIGHT→LEFT TrainAll/TestAll (sample-wise)"):
        ids, yt, yp, pr = eval_trainall_testall(right_cache, left_cache)
    results["B_TRAINTEST"] = run_and_save_block(
        out_dir=os.path.join(baseB, "TrainTest"),
        tag="R→L Train/Test",
        ids=ids, y_true=yt, y_pred=yp, y_prob=pr
    )

    # ---------------- C) LEFT→LEFT : LOOCV + K-fold ----------------
    baseC = os.path.join(SAVE_DIR, "C_TRAIN_LEFT__TEST_LEFT")

    with LogTimer("C LEFT→LEFT LOOCV (sample-wise)"):
        ids, yt, yp, pr = eval_loocv_samplewise(left_cache, left_cache)
    results["C_LOOCV"] = run_and_save_block(
        out_dir=os.path.join(baseC, "LOOCV"),
        tag="L→L LOOCV",
        ids=ids, y_true=yt, y_pred=yp, y_prob=pr
    )

    with LogTimer("C LEFT→LEFT K-fold CV (sample-wise)"):
        ids2, folds2, yt2, yp2, pr2, fold_tp, k_used = eval_kfold_samplewise(
            left_cache, CV_FOLDS_REQUESTED, shuffle=CV_SHUFFLE, random_state=CV_RANDOM_STATE
        )
    results["C_CV"] = run_and_save_block(
        out_dir=os.path.join(baseC, f"CV_{k_used}fold"),
        tag=f"L→L {k_used}-fold",
        ids=ids2, y_true=yt2, y_pred=yp2, y_prob=pr2,
        fold_ids=folds2, fold_truth_probs=fold_tp
    )

    # ---------------- D) LEFT→RIGHT : TrainAll/TestAll ----------------
    baseD = os.path.join(SAVE_DIR, "D_TRAIN_LEFT__TEST_RIGHT")
    with LogTimer("D LEFT→RIGHT TrainAll/TestAll (sample-wise)"):
        ids, yt, yp, pr = eval_trainall_testall(left_cache, right_cache)
    results["D_TRAINTEST"] = run_and_save_block(
        out_dir=os.path.join(baseD, "TrainTest"),
        tag="L→R Train/Test",
        ids=ids, y_true=yt, y_pred=yp, y_prob=pr
    )

    # ---- Interpretability-only (both sides) ----
    fig_dir_feat_R = os.path.join(SAVE_DIR, "Features_Interpretability_FULLFIT_RIGHT")
    os.makedirs(fig_dir_feat_R, exist_ok=True)
    with LogTimer("Fit full RIGHT model for interpretability-only plots"):
        fit_full_train_and_plot_features(right_cache, fig_dir_feat_R, side_label="RIGHT")

    fig_dir_feat_L = os.path.join(SAVE_DIR, "Features_Interpretability_FULLFIT_LEFT")
    os.makedirs(fig_dir_feat_L, exist_ok=True)
    with LogTimer("Fit full LEFT model for interpretability-only plots"):
        fit_full_train_and_plot_features(left_cache, fig_dir_feat_L, side_label="LEFT")

    print("\n==================== SUMMARY ====================")
    for k, r in results.items():
        auc_str = "nan" if not np.isfinite(r["auc"]) else f"{r['auc']:.4f}"
        print(f"{k:10s} | acc={r['acc']:.4f}  bacc={r['bacc']:.4f}  auc={auc_str}  | {r['out_dir']}")

    logger.info("Done. Results saved under: %s", SAVE_DIR)