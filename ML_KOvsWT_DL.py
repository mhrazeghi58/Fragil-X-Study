from __future__ import annotations

# ======================================================================
# WT vs KO FTIR classification — DEEP LEARNING PATCH-LEVEL FRAMEWORK
# USING ONLY SELECTED SPECTRAL BANDS
# ----------------------------------------------------------------------
# Adapted from your PATCH-LEVEL evaluation script.
#
# PATCHES:
#   - Randomly sample patches per sample from selected clusters
#   - Extract spectra from selected-cluster pixels inside each patch
#   - Preprocess:
#       PRE_BG_SHIFT -> BG subtract -> resample -> (optional) RMieSC
#   - Keep ONLY user-selected spectral band windows from BAND_LIBRARY
#   - Deep learning uses reduced spectra directly (not full 426 points)
#
# EVALUATIONS:
#   A) RIGHT->RIGHT: patch-level LOOCV + patch-level 20-fold CV
#   B) RIGHT->LEFT : train all RIGHT patches -> test all LEFT patches
#   C) LEFT ->LEFT : patch-level LOOCV + patch-level 20-fold CV
#   D) LEFT ->RIGHT: train all LEFT patches  -> test all RIGHT patches
#   E) MIXED       : patch-level LOOCV + patch-level 20-fold CV
#
# IMPORTANT NOTE:
#   This is patch-level DL. Patches from the same sample may appear in both
#   train and test folds during LOOCV/CV unless the split is done by sample.
#   So this framework can have sample/animal leakage, exactly like your
#   patch-level ML version. Use for exploratory analysis, not as the strongest
#   biological generalization claim.
# ======================================================================

import os
import time
import copy
import random
import logging
import traceback
import importlib
import pkgutil
import inspect
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================== USER CONFIG ==============================

RIGHT_CLUSTER_DIR = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R\UMAP_clustering_test"
RIGHT_CUBE_DIR    = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R"

LEFT_CLUSTER_DIR  = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L\UMAP_clustering_8Cluster"
LEFT_CUBE_DIR     = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"

CLUSTER_NPZ_PATTERN = "{sid}_umap_kmeans.npz"
RIGHT_CUBE_PATTERN  = "masked_cube_{sid}.npz"
LEFT_CUBE_PATTERN   = "masked_cube_{sid}.npz"

SAVE_DIR = os.path.join(RIGHT_CUBE_DIR, "WT_KO_DL_PATCHLEVEL_SELECTEDBANDS_v1")
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

# ===================== PATCH SETTINGS =====================
PATCH_H = 100
PATCH_W = 100
PATCHES_PER_SAMPLE = 20
PATCH_MIN_CLUSTER_PIXELS = 1500
PATCH_MAX_TRIES = 1000
PATCH_RANDOM_SEED_BASE = 2026
MAX_PIXELS_PER_PATCH_ANALYSIS = 2000

# ===================== CV SETTINGS =====================
CV20_FOLDS = 20
CV20_SHUFFLE = True
CV20_RANDOM_STATE = 123

# ===================== RMieSC SETTINGS =====================
APPLY_SCATTER_CORRECTION = False
SCATTER_BACKEND = "octavvs"
RMIESC_ITERATIONS = 10
RMIESC_VERBOSE = False
RMIESC_REF_MODE = "median"
OCTAVVS_VERBOSE_SCAN = True
RMIESC_CLIP = None

# ===================== SHIFT (DISABLED) =====================
PRE_BG_SHIFT = 0.0

# ===================== BASIC QUALITY FILTER =====================
FILTER_BAD_SPECTRA_BEFORE_RMIESC = True
MIN_SPECTRUM_L2_NORM = 1e-3
MAX_ABS_VALUE = 1e3
MIN_KEEP_SPECTRA = 100

# ===================== RMieSC RESAMPLING RETRIES =====================
RMIESC_SUBSET_N = 800
RMIESC_MAX_TRIES = 8
RMIESC_RANDOM_SEED_BASE = 1230

# ===================== BAND DEFINITIONS USED BY DL =====================
window = 30
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

# if True, union of all band windows is kept as a compact reduced spectrum
USE_ONLY_SELECTED_BAND_WINDOWS = True

# ===================== DEEP LEARNING SETTINGS =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GLOBAL_SEED = 42
PATCH_BAG_SIZE_TRAIN = 128
PATCH_BAG_SIZE_EVAL  = 256
TRAIN_REPEATS_PER_PATCH = 1
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.30
ENCODER_CHANNELS = 16
EMBED_DIM = 32
ATTN_DIM = 32

# ============================== PUB STYLE ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("WT_KO_DL_PATCHLEVEL")

PUB_DPI = 600
FONT_FAMILY = "Arial"
BASE_FONTSIZE = 12
BASE_WT = "#2ca02c"
BASE_KO = "#d62728"

# ============================== UTILS ==============================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(GLOBAL_SEED)


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

@contextmanager
def LogTimer(msg):
    t0 = time.time()
    logger.info(msg + " ...")
    try:
        yield
    finally:
        logger.info("%s done in %.2fs", msg, time.time() - t0)


def _short_title(s: str, maxlen: int = 32) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= maxlen else (s[:maxlen-1] + "…")


def _format_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


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


def _make_band_union_mask(wns, band_library):
    mask = np.zeros((wns.size,), dtype=bool)
    for _, (center, halfwidth) in band_library.items():
        mask |= ((wns >= (center - halfwidth)) & (wns <= (center + halfwidth)))
    return mask


SELECTED_BAND_MASK = _make_band_union_mask(WN_STD, BAND_LIBRARY)
SELECTED_BAND_INDICES = np.where(SELECTED_BAND_MASK)[0]
WN_SELECTED = WN_STD[SELECTED_BAND_MASK]
INPUT_SPECTRAL_LEN = int(WN_SELECTED.size) if USE_ONLY_SELECTED_BAND_WINDOWS else int(WN_STD_Z)

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
            app_all = app_all[keep]

    if app_all.shape[0] < MIN_KEEP_SPECTRA:
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
            return out0.astype(np.float64, copy=False)
        except Exception:
            if not _OCTAVVS_LOGGED_FIRST:
                _OCTAVVS_LOGGED_FIRST = True
                logger.warning("RMieSC failure traceback:\n%s", traceback.format_exc())

    _OCTAVVS_FAILS += 1
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


def _select_only_band_windows(X):
    if not USE_ONLY_SELECTED_BAND_WINDOWS:
        return X
    return np.asarray(X[:, SELECTED_BAND_MASK], dtype=np.float32)

# ============================== PATCH SAMPLING ==============================
def _sample_patches_from_mask(mask, patch_h, patch_w, n_patches, min_cluster_pixels, max_tries, rng):
    H, W = mask.shape
    if H < patch_h or W < patch_w:
        return []
    if int(mask.sum()) < min_cluster_pixels:
        return []
    patches = []
    tries = 0
    while len(patches) < n_patches and tries < max_tries:
        tries += 1
        r0 = int(rng.integers(0, H - patch_h + 1))
        c0 = int(rng.integers(0, W - patch_w + 1))
        r1 = r0 + patch_h
        c1 = c0 + patch_w
        sub = mask[r0:r1, c0:c1]
        if int(sub.sum()) < min_cluster_pixels:
            continue
        patches.append((r0, c0, r1, c1))
    return patches

# ============================== BUILD PATCH DATASET ==============================
def build_patch_dataset(side_name, cluster_dir, cube_dir, cube_pattern, bg_dir):
    patches_all = []
    for sid in tqdm(sample_names, desc=f"Build {side_name} patches", leave=True):
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

        rng = np.random.default_rng(abs(hash((PATCH_RANDOM_SEED_BASE, side_name, sid))) % (2**32))
        boxes = _sample_patches_from_mask(mask, PATCH_H, PATCH_W, PATCHES_PER_SAMPLE,
                                          PATCH_MIN_CLUSTER_PIXELS, PATCH_MAX_TRIES, rng)
        if len(boxes) < PATCHES_PER_SAMPLE:
            logger.warning("%s %s: only found %d/%d patches", side_name, sid, len(boxes), PATCHES_PER_SAMPLE)
        if len(boxes) == 0:
            continue

        y = 1 if sid in groups["KO"] else 0

        for pi, (r0, c0, r1, c1) in enumerate(boxes):
            sub_mask = mask[r0:r1, c0:c1]
            rr, cc = np.where(sub_mask)
            rr = rr + r0
            cc = cc + c0
            if rr.size == 0:
                continue

            X = cube[rr, cc, :].astype(np.float64, copy=False)
            if X.shape[0] > MAX_PIXELS_PER_PATCH_ANALYSIS:
                take = rng.choice(X.shape[0], size=MAX_PIXELS_PER_PATCH_ANALYSIS, replace=False)
                X = X[take]

            if PRE_BG_SHIFT is not None and float(PRE_BG_SHIFT) != 0.0:
                X = X + float(PRE_BG_SHIFT)

            X = _bg_subtract(X, Z_src, sid, bg_dir)
            if X.shape[1] != WN_STD_Z:
                X = _resample_to_std(X, Z_src)

            X_after = apply_rmiesc(X, WN_STD, sid=sid, side=side_name)
            X_after = np.asarray(X_after, dtype=np.float32)
            X_after = _select_only_band_windows(X_after)

            patches_all.append({
                "patch_id": f"{side_name}_{sid}_p{pi:02d}",
                "sid": sid,
                "y": y,
                "X": X_after,
            })

    rng = np.random.default_rng(0)
    rng.shuffle(patches_all)
    logger.info("%s: built %d patches total", side_name, len(patches_all))
    return patches_all

# ============================== NORMALIZATION ==============================
def fit_train_normalizer(patches):
    X_all = np.vstack([p["X"] for p in patches]).astype(np.float32)
    mu = np.mean(X_all, axis=0).astype(np.float32)
    sd = np.std(X_all, axis=0).astype(np.float32)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def apply_normalizer_to_patches(patches, mu, sd):
    out = []
    for p in patches:
        q = dict(p)
        q["X"] = ((q["X"].astype(np.float32) - mu[None, :]) / sd[None, :]).astype(np.float32)
        out.append(q)
    return out

# ============================== DATASET ==============================
class PatchBagDataset(Dataset):
    def __init__(self, patches, bag_size, training=True, repeats=1):
        self.patches = list(patches)
        self.bag_size = int(bag_size)
        self.training = bool(training)
        self.repeats = int(max(1, repeats))

    def __len__(self):
        return len(self.patches) * self.repeats if self.training else len(self.patches)

    def _sample_bag(self, X, bag_size, seed):
        rng = np.random.default_rng(seed)
        n = X.shape[0]
        if n <= bag_size:
            idx = rng.choice(n, size=bag_size, replace=True)
        else:
            idx = rng.choice(n, size=bag_size, replace=False)
        return X[idx]

    def __getitem__(self, idx):
        p = self.patches[idx % len(self.patches)] if self.training else self.patches[idx]
        seed = GLOBAL_SEED * 100000 + idx * 97 + abs(hash(p["patch_id"])) % 10000
        bag = self._sample_bag(p["X"], self.bag_size, seed)
        return {
            "bag": torch.from_numpy(bag.astype(np.float32)),
            "label": torch.tensor(int(p["y"]), dtype=torch.long),
            "patch_id": p["patch_id"],
            "sid": p["sid"],
        }


def collate_bags(batch):
    bags = torch.stack([b["bag"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    patch_ids = [b["patch_id"] for b in batch]
    sids = [b["sid"] for b in batch]
    return bags, labels, patch_ids, sids

# ============================== MODEL ==============================
class SpectralEncoder(nn.Module):
    def __init__(self, in_len, channels=16, embed_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(channels, channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 2, channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        x = self.fc(x)
        return x


class AttentionMIL(nn.Module):
    def __init__(self, in_len, channels=16, embed_dim=32, attn_dim=32, dropout=0.3):
        super().__init__()
        self.encoder = SpectralEncoder(in_len=in_len, channels=channels, embed_dim=embed_dim, dropout=dropout)
        self.attn_V = nn.Linear(embed_dim, attn_dim)
        self.attn_U = nn.Linear(embed_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
        )

    def forward(self, bag):
        B, N, Z = bag.shape
        h = self.encoder(bag.reshape(B * N, Z)).reshape(B, N, -1)
        a = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))
        a = self.attn_w(a).squeeze(-1)
        a = torch.softmax(a, dim=1)
        z = torch.sum(h * a.unsqueeze(-1), dim=1)
        logits = self.classifier(z)
        return logits, a


def build_model():
    return AttentionMIL(
        in_len=INPUT_SPECTRAL_LEN,
        channels=ENCODER_CHANNELS,
        embed_dim=EMBED_DIM,
        attn_dim=ATTN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

# ============================== TRAIN / EVAL ==============================
def make_class_weight_tensor(labels):
    y = np.asarray(labels, dtype=int)
    n0 = max(1, int(np.sum(y == 0)))
    n1 = max(1, int(np.sum(y == 1)))
    total = n0 + n1
    return torch.tensor([total/(2*n0), total/(2*n1)], dtype=torch.float32, device=DEVICE)


def evaluate_model(model, loader, criterion=None):
    model.eval()
    losses = []
    patch_ids_all, sid_all, y_true, y_pred, y_prob = [], [], [], [], []
    with torch.no_grad():
        for bags, labels, patch_ids, sids in loader:
            bags = bags.to(DEVICE)
            labels = labels.to(DEVICE)
            logits, _ = model(bags)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            if criterion is not None:
                losses.append(float(criterion(logits, labels).item()))
            patch_ids_all.extend(patch_ids)
            sid_all.extend(sids)
            y_true.extend(labels.cpu().numpy().astype(int).tolist())
            y_pred.extend(preds.cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.cpu().numpy().astype(float).tolist())
    y_true_arr = np.array(y_true, int)
    y_pred_arr = np.array(y_pred, int)
    acc = float(accuracy_score(y_true_arr, y_pred_arr)) if y_true_arr.size else np.nan
    return (
        float(np.mean(losses)) if losses else np.nan,
        np.array(patch_ids_all, object),
        np.array(sid_all, object),
        y_true_arr,
        y_pred_arr,
        np.array(y_prob, float),
        acc,
    )


def train_one_model(train_patches, val_patches=None, tag=""):
    train_ds = PatchBagDataset(train_patches, PATCH_BAG_SIZE_TRAIN, training=True, repeats=TRAIN_REPEATS_PER_PATCH)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_bags)

    if val_patches is not None and len(val_patches) > 0:
        val_ds = PatchBagDataset(val_patches, PATCH_BAG_SIZE_EVAL, training=False, repeats=1)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_bags)
    else:
        val_loader = None

    model = build_model()
    criterion = nn.CrossEntropyLoss(weight=make_class_weight_tensor([p["y"] for p in train_patches]))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    bad_epochs = 0

    epoch_bar = tqdm(range(1, EPOCHS + 1), desc=f"{tag} epochs", leave=False)
    for epoch in epoch_bar:
        model.train()
        train_losses = []
        batch_bar = tqdm(train_loader, desc=f"{tag} train", leave=False)
        for bags, labels, _, _ in batch_bar:
            bags = bags.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(bags)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_value = float(loss.item())
            train_losses.append(loss_value)
            batch_bar.set_postfix(loss=f"{loss_value:.4f}")

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        if val_loader is not None:
            val_loss, _, _, yv, pv, _, val_acc = evaluate_model(model, val_loader, criterion=criterion)
        else:
            val_loss = train_loss
            val_acc = np.nan

        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}", val_acc=("nan" if not np.isfinite(val_acc) else f"{val_acc:.3f}"))
        logger.info("%s epoch %03d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f", tag, epoch, train_loss, val_loss, val_acc if np.isfinite(val_acc) else float('nan'))

        if bad_epochs >= EARLY_STOPPING_PATIENCE:
            logger.info("%s early stopping at epoch %d", tag, epoch)
            break

    model.load_state_dict(best_state)
    return model

# ============================== FIGURES ==============================
def _savefig(fig, path_png):
    fig.savefig(path_png, dpi=PUB_DPI, bbox_inches="tight")
    fig.savefig(path_png.replace(".png", ".pdf"), dpi=PUB_DPI, bbox_inches="tight")
    plt.close(fig)


def _metrics_footer_short(y_true, y_pred, include_n=True):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    if include_n:
        n = int(y_true.size)
        n0 = int(np.sum(y_true == 0))
        n1 = int(np.sum(y_true == 1))
        footer = f"N={n} (WT={n0}, KO={n1})   |   ACC: {_format_pct(acc)}   |   BACC: {_format_pct(bacc)}"
    else:
        footer = f"ACC: {_format_pct(acc)}   |   BACC: {_format_pct(bacc)}"
    return footer, float(acc), float(bacc)


def plot_confusion(cm, out_png, title, y_true, y_pred, include_n=True):
    _apply_pub_style()
    cm = np.asarray(cm, dtype=int)
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_row = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)
    footer, _, _ = _metrics_footer_short(y_true, y_pred, include_n=include_n)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm_row, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Recall (row-normalized)")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["WT", "KO"]); ax.set_yticklabels(["WT", "KO"])
    ax.set_xlabel("Predicted", labelpad=10); ax.set_ylabel("True")
    ax.set_title(_short_title(title, 26))
    for i in range(2):
        for j in range(2):
            cnt = int(cm[i, j]); rec = float(cm_row[i, j])
            txt = f"{cnt}\n({_format_pct(rec)})"
            color = "white" if rec > 0.55 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=BASE_FONTSIZE + 2, color=color)
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
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out_png)


def plot_roc_mean_sd(fold_truth_probs, out_png, title):
    _apply_pub_style()
    fpr_grid = np.linspace(0.0, 1.0, 201)
    tprs = []; aucs = []
    for y_t, y_p in fold_truth_probs:
        y_t = np.asarray(y_t, int); y_p = np.asarray(y_p, float)
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
    ax.fill_between(fpr_grid, np.clip(mean_tpr - sd_tpr, 0, 1), np.clip(mean_tpr + sd_tpr, 0, 1), alpha=0.18, linewidth=0)
    ax.plot([0, 1], [0, 1], "--", lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title(_short_title(title, 32))
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out_png)


def plot_prob_violin(y_true, y_prob, out_png, title):
    _apply_pub_style()
    y_true = np.asarray(y_true, int); y_prob = np.asarray(y_prob, float)
    wt = y_prob[y_true == 0]; ko = y_prob[y_true == 1]
    wt = wt[np.isfinite(wt)]; ko = ko[np.isfinite(ko)]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    data = [wt, ko]
    parts = ax.violinplot(data, positions=[0, 1], showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.25); pc.set_edgecolor("black"); pc.set_linewidth(0.8)
    rng = np.random.default_rng(0)
    j_wt = (rng.random(wt.size) - 0.5) * 0.18
    j_ko = (rng.random(ko.size) - 0.5) * 0.18
    ax.scatter(np.zeros_like(wt) + j_wt, wt, s=18, alpha=0.55, edgecolors="black", linewidths=0.25, color=BASE_WT)
    ax.scatter(np.ones_like(ko) + j_ko, ko, s=18, alpha=0.55, edgecolors="black", linewidths=0.25, color=BASE_KO)
    ax.axhline(0.5, linestyle="--", linewidth=2.0, color="black", alpha=0.55)

    def med_iqr(a):
        if a.size == 0:
            return np.nan, np.nan, np.nan
        q1, q3 = np.percentile(a, [25, 75])
        return float(np.median(a)), float(q1), float(q3)

    wt_med, wt_q1, wt_q3 = med_iqr(wt)
    ko_med, ko_q1, ko_q3 = med_iqr(ko)
    ax.text(0, 1.02, f"WT: med={wt_med:.2f}\nIQR=[{wt_q1:.2f},{wt_q3:.2f}]", ha="center", va="bottom", fontsize=BASE_FONTSIZE)
    ax.text(1, 1.02, f"KO: med={ko_med:.2f}\nIQR=[{ko_q1:.2f},{ko_q3:.2f}]", ha="center", va="bottom", fontsize=BASE_FONTSIZE)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["WT", "KO"])
    ax.set_ylabel("P(KO)"); ax.set_ylim(-0.02, 1.08)
    ax.set_title(_short_title(title, 32))
    ax.grid(True, axis="y")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    _savefig(fig, out_png)

# ============================== CSV + METRICS ==============================
def save_patch_predictions_csv(path, patch_ids, sample_ids, y_true, y_pred, y_prob, fold_ids=None):
    with open(path, "w", encoding="utf-8") as f:
        if fold_ids is None:
            f.write("patch_id,sample,true_label,pred_label,prob_KO\n")
            for pid, sid, yt, yp, p in zip(patch_ids, sample_ids, y_true, y_pred, y_prob):
                t = "KO" if int(yt) == 1 else "WT"
                pr = "KO" if int(yp) == 1 else "WT"
                ps = "" if not np.isfinite(p) else f"{float(p):.8f}"
                f.write(f"{pid},{sid},{t},{pr},{ps}\n")
        else:
            f.write("patch_id,sample,fold,true_label,pred_label,prob_KO\n")
            for pid, sid, fd, yt, yp, p in zip(patch_ids, sample_ids, fold_ids, y_true, y_pred, y_prob):
                t = "KO" if int(yt) == 1 else "WT"
                pr = "KO" if int(yp) == 1 else "WT"
                ps = "" if not np.isfinite(p) else f"{float(p):.8f}"
                f.write(f"{pid},{sid},{fd},{t},{pr},{ps}\n")
    logger.info("Saved: %s", path)


def report_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    auc = np.nan
    m = np.isfinite(y_prob)
    if m.sum() >= 3 and len(np.unique(y_true[m])) == 2:
        auc = roc_auc_score(y_true[m], y_prob[m])
    print("\n====================", name, "(patch-level DL, selected bands)", "====================")
    print("Device:             ", DEVICE)
    print("Input spectral len: ", INPUT_SPECTRAL_LEN)
    print("n_patches:          ", len(y_true))
    print("Accuracy:           ", f"{acc:.4f}")
    print("Balanced accuracy:  ", f"{bacc:.4f}")
    print("AUC (KO prob):      ", (f"{auc:.4f}" if np.isfinite(auc) else "nan"))
    print("Confusion matrix [WT, KO]:")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["WT","KO"], digits=4))
    return float(acc), float(bacc), float(auc) if np.isfinite(auc) else np.nan, cm

# ============================== EVALS ==============================
def eval_patch_loocv(patches):
    idx = np.arange(len(patches))
    loo = LeaveOneOut()
    pids, sids, y_true, y_pred, y_prob = [], [], [], [], []
    for fold_i, (tr_idx, te_idx) in tqdm(enumerate(loo.split(idx), start=1), total=len(idx), desc="LOOCV folds", leave=True):
        train_p = [patches[i] for i in tr_idx]
        test_p  = [patches[int(te_idx[0])]]
        mu, sd = fit_train_normalizer(train_p)
        train_p = apply_normalizer_to_patches(train_p, mu, sd)
        test_p  = apply_normalizer_to_patches(test_p,  mu, sd)
        model = train_one_model(train_p, val_patches=test_p, tag=f"LOOCV fold {fold_i}")
        test_loader = DataLoader(PatchBagDataset(test_p, PATCH_BAG_SIZE_EVAL, training=False), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_bags)
        _, pid, sid, yt, yp, pr, test_acc = evaluate_model(model, test_loader, criterion=None)
        pids.extend(pid.tolist()); sids.extend(sid.tolist())
        y_true.extend(yt.tolist()); y_pred.extend(yp.tolist()); y_prob.extend(pr.tolist())
        running_acc = accuracy_score(np.array(y_true, int), np.array(y_pred, int)) if len(y_true) else np.nan
        logger.info("Fold result | current fold test_acc=%.4f | running_acc=%.4f", test_acc if np.isfinite(test_acc) else float('nan'), running_acc if np.isfinite(running_acc) else float('nan'))
    return np.array(pids, object), np.array(sids, object), np.array(y_true, int), np.array(y_pred, int), np.array(y_prob, float)


def eval_patch_stratified_kfold(patches, n_splits=20, shuffle=True, random_state=123):
    y_all = np.array([int(p["y"]) for p in patches], dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    idx = np.arange(len(patches))
    pids, sids, fold_ids = [], [], []
    y_true, y_pred, y_prob = [], [], []
    fold_truth_probs = []
    fold_num = 0
    for tr_idx, te_idx in tqdm(skf.split(idx, y_all), total=n_splits, desc="CV folds", leave=True):
        fold_num += 1
        train_p = [patches[i] for i in tr_idx]
        test_p  = [patches[i] for i in te_idx]
        mu, sd = fit_train_normalizer(train_p)
        train_p = apply_normalizer_to_patches(train_p, mu, sd)
        test_p  = apply_normalizer_to_patches(test_p,  mu, sd)
        model = train_one_model(train_p, val_patches=test_p, tag=f"CV fold {fold_num}")
        test_loader = DataLoader(PatchBagDataset(test_p, PATCH_BAG_SIZE_EVAL, training=False), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_bags)
        _, pid, sid, yt, yp, pr, test_acc = evaluate_model(model, test_loader, criterion=None)
        fold_truth_probs.append((yt.astype(int), pr.astype(float)))
        pids.extend(pid.tolist()); sids.extend(sid.tolist()); fold_ids.extend([fold_num]*len(pid))
        y_true.extend(yt.tolist()); y_pred.extend(yp.tolist()); y_prob.extend(pr.tolist())
    return np.array(pids, object), np.array(sids, object), np.array(fold_ids, int), np.array(y_true, int), np.array(y_pred, int), np.array(y_prob, float), fold_truth_probs


def eval_train_all_test_all(train_patches, test_patches, tag=""):
    mu, sd = fit_train_normalizer(train_patches)
    train_p = apply_normalizer_to_patches(train_patches, mu, sd)
    test_p  = apply_normalizer_to_patches(test_patches,  mu, sd)
    model = train_one_model(train_p, val_patches=train_p, tag=tag)
    test_loader = DataLoader(PatchBagDataset(test_p, PATCH_BAG_SIZE_EVAL, training=False), batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_bags)
    _, pid, sid, yt, yp, pr, test_acc = evaluate_model(model, test_loader, criterion=None)
    logger.info("Final test accuracy for %s = %.4f", tag, test_acc if np.isfinite(test_acc) else float('nan'))
    return pid, sid, yt, yp, pr

# ============================== RUN HELPERS ==============================
def run_and_save_eval(out_dir, tag, patch_ids, sample_ids, y_true, y_pred, y_prob, fold_ids=None, fold_truth_probs=None):
    os.makedirs(out_dir, exist_ok=True)
    save_patch_predictions_csv(os.path.join(out_dir, "patch_predictions.csv"), patch_ids, sample_ids, y_true, y_pred, y_prob, fold_ids=fold_ids)
    acc, bacc, auc, cm = report_metrics(tag, y_true, y_pred, y_prob)
    plot_confusion(cm, os.path.join(out_dir, "confusion_pub.png"), title=tag, y_true=y_true, y_pred=y_pred, include_n=True)
    if fold_truth_probs is None:
        plot_roc_single(y_true, y_prob, os.path.join(out_dir, "roc_pub.png"), title=tag)
    else:
        plot_roc_mean_sd(fold_truth_probs, os.path.join(out_dir, "roc_pub.png"), title=tag)
    plot_prob_violin(y_true, y_prob, os.path.join(out_dir, "prob_violin_pub.png"), title=tag)
    return {"acc": acc, "bacc": bacc, "auc": auc, "out_dir": out_dir}

# ============================== MAIN ==============================
if __name__ == "__main__":
    print("========== LIVE RUN START ==========")
    logger.info("SAVE_DIR: %s", SAVE_DIR)
    logger.info("DEVICE: %s", DEVICE)
    logger.info("PATCH %dx%d | patches/sample=%d | min cluster pixels=%d", PATCH_H, PATCH_W, PATCHES_PER_SAMPLE, PATCH_MIN_CLUSTER_PIXELS)
    logger.info("BG subtraction=%s | RMieSC=%s", APPLY_BG_SUBTRACTION, APPLY_SCATTER_CORRECTION)
    logger.info("Using selected band windows only: %s", USE_ONLY_SELECTED_BAND_WINDOWS)
    logger.info("Input spectral length after band selection = %d", INPUT_SPECTRAL_LEN)

    with LogTimer("Build RIGHT patches"):
        right_patches = build_patch_dataset("RIGHT", RIGHT_CLUSTER_DIR, RIGHT_CUBE_DIR, RIGHT_CUBE_PATTERN, BG_DIR_RIGHT)

    _OCTAVVS_FAILS = 0
    _OCTAVVS_LOGGED_FIRST = False

    with LogTimer("Build LEFT patches"):
        left_patches = build_patch_dataset("LEFT", LEFT_CLUSTER_DIR, LEFT_CUBE_DIR, LEFT_CUBE_PATTERN, BG_DIR_LEFT)

    mixed_patches = list(right_patches) + list(left_patches)
    rng_mix = np.random.default_rng(999)
    rng_mix.shuffle(mixed_patches)
    logger.info("MIXED: built %d patches total (RIGHT+LEFT)", len(mixed_patches))

    results = {}

    baseA = os.path.join(SAVE_DIR, "A_RIGHT_RIGHT")
    with LogTimer("A RIGHT->RIGHT LOOCV"):
        pids, sids, yt, yp, pr = eval_patch_loocv(right_patches)
    results["A_LOOCV"] = run_and_save_eval(os.path.join(baseA, "LOOCV"), "R->R LOOCV (DL, selected bands)", pids, sids, yt, yp, pr)

    with LogTimer("A RIGHT->RIGHT 20-fold"):
        pids2, sids2, folds2, yt2, yp2, pr2, fold_tp = eval_patch_stratified_kfold(right_patches, n_splits=CV20_FOLDS, shuffle=CV20_SHUFFLE, random_state=CV20_RANDOM_STATE)
    results["A_CV20"] = run_and_save_eval(os.path.join(baseA, "CV20"), "R->R 20-fold (DL, selected bands)", pids2, sids2, yt2, yp2, pr2, fold_ids=folds2, fold_truth_probs=fold_tp)

    baseB = os.path.join(SAVE_DIR, "B_RIGHT_LEFT")
    with LogTimer("B RIGHT->LEFT Train/Test"):
        pids, sids, yt, yp, pr = eval_train_all_test_all(right_patches, left_patches, tag="R->L Train/Test")
    results["B_TRAINTEST"] = run_and_save_eval(os.path.join(baseB, "TrainTest"), "R->L Train/Test (DL, selected bands)", pids, sids, yt, yp, pr)

    baseC = os.path.join(SAVE_DIR, "C_LEFT_LEFT")
    with LogTimer("C LEFT->LEFT LOOCV"):
        pids, sids, yt, yp, pr = eval_patch_loocv(left_patches)
    results["C_LOOCV"] = run_and_save_eval(os.path.join(baseC, "LOOCV"), "L->L LOOCV (DL, selected bands)", pids, sids, yt, yp, pr)

    with LogTimer("C LEFT->LEFT 20-fold"):
        pids2, sids2, folds2, yt2, yp2, pr2, fold_tp = eval_patch_stratified_kfold(left_patches, n_splits=CV20_FOLDS, shuffle=CV20_SHUFFLE, random_state=CV20_RANDOM_STATE)
    results["C_CV20"] = run_and_save_eval(os.path.join(baseC, "CV20"), "L->L 20-fold (DL, selected bands)", pids2, sids2, yt2, yp2, pr2, fold_ids=folds2, fold_truth_probs=fold_tp)

    baseD = os.path.join(SAVE_DIR, "D_LEFT_RIGHT")
    with LogTimer("D LEFT->RIGHT Train/Test"):
        pids, sids, yt, yp, pr = eval_train_all_test_all(left_patches, right_patches, tag="L->R Train/Test")
    results["D_TRAINTEST"] = run_and_save_eval(os.path.join(baseD, "TrainTest"), "L->R Train/Test (DL, selected bands)", pids, sids, yt, yp, pr)

    baseE = os.path.join(SAVE_DIR, "E_MIXED_BOTH_SIDES")
    with LogTimer("E MIXED LOOCV"):
        pids, sids, yt, yp, pr = eval_patch_loocv(mixed_patches)
    results["E_LOOCV"] = run_and_save_eval(os.path.join(baseE, "LOOCV"), "Mixed LOOCV (DL, selected bands)", pids, sids, yt, yp, pr)

    with LogTimer("E MIXED 20-fold"):
        pids2, sids2, folds2, yt2, yp2, pr2, fold_tp = eval_patch_stratified_kfold(mixed_patches, n_splits=CV20_FOLDS, shuffle=CV20_SHUFFLE, random_state=CV20_RANDOM_STATE)
    results["E_CV20"] = run_and_save_eval(os.path.join(baseE, "CV20"), "Mixed 20-fold (DL, selected bands)", pids2, sids2, yt2, yp2, pr2, fold_ids=folds2, fold_truth_probs=fold_tp)

    print("\n==================== SUMMARY ====================")
    for k, r in results.items():
        auc_str = "nan" if not np.isfinite(r["auc"]) else f"{r['auc']:.4f}"
        print(f"{k:14s} | acc={r['acc']:.4f}  bacc={r['bacc']:.4f}  auc={auc_str}  | {r['out_dir']}")

    logger.info("Done. Results saved under: %s", SAVE_DIR)
    print("========== LIVE RUN FINISHED ==========")
