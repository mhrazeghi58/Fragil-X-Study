# -*- coding: utf-8 -*-
"""
FTIR WT vs KO — RAW + 2nd-derivative (SG) using:
- pixel_indices + cluster_labels from {sid}_umap_kmeans.npz  (ROI mask)
- spectra cube from maske_cube_{sid}.npz (key 'data' -> H,W,Z)
- per-sample BG subtraction from bg_first*_mean_spectrum_{sid}.npz (NO wn inside; align by index)

NO resampling. Crop to 960–1800 (step=2).

Update requested:
- Move legend OUTSIDE the axes (right side), stacked vertically.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =========================
# PATHS
# =========================
cluster_dir = r"D:\Filiz Lab\Data_2025\Mid-IR Data\11_18_2025\Area_EX\UMAP_clustering_8Cluster"
cube_dir    = r"D:\Filiz Lab\Data_2025\Mid-IR Data\11_18_2025\Area_EX"
bg_dir      = r"D:\Filiz Lab\Data_2025\Mid-IR Data\11_18_2025\Area_B\BG_First300ROI_MeanSpectrum"

save_dir    = os.path.join(cube_dir, "WT_KO_PIXELINDICES")
os.makedirs(save_dir, exist_ok=True)
out_prefix  = os.path.join(save_dir, "WT_KO")

# =========================
# SAMPLES / GROUPS
# =========================
sample_names = ["T4E","T3E",#"T6","T17","T19","T20","T21",
                "T1E","T2E",]#"T12","T13","T14","T15","T22"]

groups = {
    "WT": ["T4E","T3E"],#,"T6","T17","T19","T20","T21"],
    "KO": ["T1E","T2E"]#,"T12","T13","T14","T15","T22"]
}

selected_clusters = {
    "T1E":[0,1,2,3,4,5,6,7],
    "T3E":[0,1,2,3,4,5,6,7],
    "T2E":[0,1,2,3,4,5,6,7],
    "T4E":[0,1,2,3,4,5,6,7],
  #  "T19":[0,1,2,3,4.5,6,7],  # 4.5 dropped
  #  "T20":[0,1,2,3,4,5,6,7],
  #  "T21":[0,1,2,3,4,5,6,7],
  #  "T10":[0,1,2,3,4,5,6,7],
  #  "T11":[0,1,2,3,4,5,6,7],
  #  "T12":[0,1,2,3,4,5,6,7],
  #  "T13":[0,1,2,3,4,5,6,7],
  #  "T14":[0,1,2,3,4,5,6,7],
  #  "T15":[0,1,2,3,4,5,6,7],
  #  "T22":[0,1,2,3,4,5,6,7],
}

# =========================
# SETTINGS
# =========================
WN_MIN, WN_MAX = 960, 1800
WN_STEP = 2
SYNTH_START_FOR_426 = 950   # 426 pts -> 950..1800 step2
SYNTH_START_FOR_421 = 960   # 421 pts -> 960..1800 step2

SG_WINDOW = 35
SG_POLY = 3
DERIV_ORDER = 2

DPI = 300
LINEWIDTH = 3.6
ALPHA_SEM = 0.35
FONT_SIZE = 16

WT_COLOR = "#2ca02c"  # green
KO_COLOR = "#d62728"  # red

# square-ish plot area; we will reserve extra right margin for legend
FIGSIZE = (10.5, 7.2)

CLUSTER_NPZ_PATTERN = "{sid}_umap_kmeans.npz"
CUBE_NPZ_PATTERN    = "masked_cube_{sid}.npz"

BG_PATTERNS = [
    "bg_first30_mean_spectrum_{sid}.npz",
    "bg_first300_mean_spectrum_{sid}.npz",
]
SKIP_BG_IF_MISSING = True

# =========================
# BAND WINDOWS (edit if needed)
# =========================
BANDS = [
    (1735, 1755, "Ester C=O",         "#d67c7c"),
    (1690, 1715, "Carbonyl (alt)",    "#c39bd3"),
    (1640, 1680, "Amide I",           "#6aa6d8"),
    (1600, 1625, "Aromatic / ring",   "#7fb3d5"),
    (1510, 1580, "Amide II",          "#9bc1e6"),
    (1450, 1475, "CH2",               "#f0b27a"),
    (1410, 1445, "COO- / CH bend",    "#f7dc6f"),
    (1370, 1390, "CH3",               "#f5cba7"),
    (1310, 1340, "Amide III",         "#82e0aa"),
    (1230, 1260, "PO2- (asym)",       "#7fbf7f"),
    (1190, 1225, "C–O / P–O",         "#a9dfbf"),
    (1120, 1160, "C–O–C / glycogen",  "#a3e4d7"),
    (1060, 1100, "PO2- (sym)",        "#aed6f1"),
    (1020, 1055, "Carbohydrate",      "#d6eaf8"),
]

# =========================
# Helpers
# =========================
def clean_cluster_list(lst):
    out = []
    for v in lst:
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
        elif isinstance(v, float) and float(v).is_integer():
            out.append(int(v))
    return sorted(list(set(out)))

def load_cube_data_3d(sid):
    p = os.path.join(cube_dir, CUBE_NPZ_PATTERN.format(sid=sid))
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing cube file: {p}")
    npz = np.load(p, allow_pickle=True)
    if "data" not in npz.files:
        raise ValueError(f"{sid}: cube npz has no 'data' key. keys={npz.files}")
    data = np.asarray(npz["data"])
    if data.ndim != 3:
        raise ValueError(f"{sid}: cube 'data' is not 3D. shape={data.shape}")
    H, W, Z = data.shape
    return data, H, W, Z

def synthesize_wn(Z):
    if Z == 426:
        return np.arange(SYNTH_START_FOR_426, SYNTH_START_FOR_426 + WN_STEP*Z, WN_STEP, dtype=float)
    if Z == 421:
        return np.arange(SYNTH_START_FOR_421, SYNTH_START_FOR_421 + WN_STEP*Z, WN_STEP, dtype=float)
    return np.arange(Z, dtype=float)

def load_bg_mean(sid):
    bg_path = None
    for pat in BG_PATTERNS:
        p = os.path.join(bg_dir, pat.format(sid=sid))
        if os.path.exists(p):
            bg_path = p
            break
    if bg_path is None:
        return None, None

    npz = np.load(bg_path, allow_pickle=True)
    for k in npz.files:
        a = np.asarray(npz[k])
        if np.issubdtype(a.dtype, np.number) and a.ndim == 1 and a.size > 50:
            return a.astype(float).ravel(), bg_path
    raise ValueError(f"{sid}: BG npz had no 1D numeric vector. keys={npz.files}")

def load_cluster_indices_and_labels(sid, H, W):
    p = os.path.join(cluster_dir, CLUSTER_NPZ_PATTERN.format(sid=sid))
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing cluster file: {p}")
    npz = np.load(p, allow_pickle=True)

    if "cluster_labels" not in npz.files:
        raise ValueError(f"{sid}: cluster npz missing 'cluster_labels'. keys={npz.files}")
    if "pixel_indices" not in npz.files:
        raise ValueError(f"{sid}: cluster npz missing 'pixel_indices'. keys={npz.files}")

    labels = np.asarray(npz["cluster_labels"]).astype(np.int32, copy=False)
    pix = np.asarray(npz["pixel_indices"])

    if pix.ndim == 2 and pix.shape[1] == 2:
        rc = pix.astype(np.int64, copy=False)
        rows, cols = rc[:, 0], rc[:, 1]
        flat_idx = rows * W + cols
    elif pix.ndim == 2 and pix.shape[0] == 2:
        rc = pix.astype(np.int64, copy=False)
        rows, cols = rc[0, :], rc[1, :]
        flat_idx = rows * W + cols
    elif pix.ndim == 1:
        flat_idx = pix.astype(np.int64, copy=False)
    else:
        raise ValueError(f"{sid}: unsupported pixel_indices shape={pix.shape}")

    if flat_idx.size != labels.size:
        raise ValueError(
            f"{sid}: pixel_indices and cluster_labels mismatch: "
            f"{flat_idx.size} vs {labels.size}. pixel_indices shape={pix.shape}"
        )

    n_pix = H * W
    if flat_idx.min() < 0 or flat_idx.max() >= n_pix:
        raise ValueError(
            f"{sid}: pixel_indices out of bounds. min={flat_idx.min()} max={flat_idx.max()} n_pix={n_pix}"
        )
    return flat_idx, labels

def sample_mean_and_sem(sample_mat):
    mean = np.nanmean(sample_mat, axis=0)
    n = np.sum(~np.isnan(sample_mat), axis=0)
    std = np.nanstd(sample_mat, axis=0, ddof=1)
    sem = std / np.sqrt(np.maximum(n, 1))
    sem[n < 2] = np.nan
    return mean, sem, n

def second_derivative_1d(y):
    y2 = y.copy()
    if np.any(np.isnan(y2)):
        x = np.arange(len(y2))
        good = ~np.isnan(y2)
        if np.sum(good) < 5:
            return np.full_like(y2, np.nan)
        y2[~good] = np.interp(x[~good], x[good], y2[good])

    w = min(SG_WINDOW, len(y2) - (len(y2)+1) % 2)
    if w < 5:
        return np.full_like(y2, np.nan)
    if w % 2 == 0:
        w -= 1
    return savgol_filter(y2, window_length=w, polyorder=SG_POLY, deriv=DERIV_ORDER)

# =========================
# Plot helpers
# =========================
def pub_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_SIZE)
    ax.set_facecolor("white")

def add_bands(ax, bands):
    for (x0, x1, label, color) in bands:
        ax.axvspan(x0, x1, color=color, alpha=0.20, lw=0, zorder=0)
        ax.text(
            (x0 + x1) / 2,
            0.98,
            label,
            rotation=90,
            va="top",
            ha="center",
            fontsize=FONT_SIZE - 2,
            color=color,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            zorder=5
        )

def legend_outside_right(ax):
    # Put legend outside on the right, stacked vertically
    ax.legend(
        frameon=False,
        fontsize=FONT_SIZE,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),  # outside right
        borderaxespad=0.0,
        handlelength=2.8,
        labelspacing=0.6,
        handletextpad=0.8,
    )

def ylim_robust(mean_a, sem_a, mean_b, sem_b, pad_frac=0.15, p_lo=2, p_hi=98):
    y = np.r_[mean_a - sem_a, mean_a + sem_a, mean_b - sem_b, mean_b + sem_b]
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1, 1)
    lo = np.percentile(y, p_lo)
    hi = np.percentile(y, p_hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = np.nanmin(y)
        hi = np.nanmax(y)
    rng = hi - lo
    if rng == 0:
        rng = max(abs(hi), 1e-6)
    return (lo - pad_frac*rng, hi + pad_frac*rng)

# =========================
# Per-sample mean spectrum (union clusters)
# =========================
def compute_sample_mean_spectrum(sid, ref_grid=None):
    cube, H, W, Z = load_cube_data_3d(sid)
    cube2d = cube.reshape(H * W, Z)

    wn = synthesize_wn(Z)
    crop = (wn >= WN_MIN) & (wn <= WN_MAX)
    wn2 = wn[crop]

    if ref_grid is not None:
        if len(wn2) != len(ref_grid) or not np.allclose(wn2, ref_grid, atol=1e-6, rtol=0):
            raise ValueError(
                f"Wavenumber grid mismatch for {sid}.\n"
                f"{sid}: {wn2[0]}..{wn2[-1]} N={len(wn2)} step={wn2[1]-wn2[0]}\n"
                f"REF: {ref_grid[0]}..{ref_grid[-1]} N={len(ref_grid)} step={ref_grid[1]-ref_grid[0]}"
            )

    pix_idx, labels = load_cluster_indices_and_labels(sid, H, W)

    allowed = clean_cluster_list(selected_clusters.get(sid, []))
    if not allowed:
        print(f"WARNING: {sid} has no allowed clusters.")
        return None, None, 0

    keep = np.isin(labels, allowed)
    sel_idx = pix_idx[keep]
    n_sel = int(sel_idx.size)
    if n_sel == 0:
        print(f"WARNING: {sid} has 0 pixels in selected clusters.")
        return None, None, 0

    pix_spec = cube2d[sel_idx, :][:, crop].astype(np.float32)

    bg, _ = load_bg_mean(sid)
    if bg is None:
        if SKIP_BG_IF_MISSING:
            print(f"WARNING: BG missing for {sid}. BG subtraction skipped.")
        bg2 = np.zeros(pix_spec.shape[1], dtype=np.float32)
    else:
        bg_full = np.zeros(Z, dtype=np.float32)
        L = min(len(bg), Z)
        bg_full[:L] = bg[:L].astype(np.float32)
        bg2 = bg_full[crop]

    pix_spec = pix_spec - bg2[None, :]
    sm = np.nanmean(pix_spec, axis=0)
    return sm, wn2, n_sel

# =========================
# Reference grid from T1
# =========================
sm_ref, grid, _ = compute_sample_mean_spectrum("T1E", ref_grid=None)
print("REF GRID:", grid[0], "to", grid[-1], "N=", len(grid), "step=", (grid[1] - grid[0]))

# =========================
# Collect groups
# =========================
def collect_group(group_sids):
    rows, used = [], []
    for sid in group_sids:
        sm, wn2, nsel = compute_sample_mean_spectrum(sid, ref_grid=grid)
        if sm is None:
            continue
        rows.append(sm)
        used.append(sid)
    if not rows:
        return np.zeros((0, len(grid))), used
    return np.vstack(rows), used

WT_mat, WT_used = collect_group(groups["WT"])
KO_mat, KO_used = collect_group(groups["KO"])
print("WT used:", WT_used)
print("KO used:", KO_used)

# =========================
# Stats
# =========================
wt_mean, wt_sem, _ = sample_mean_and_sem(WT_mat)
ko_mean, ko_sem, _ = sample_mean_and_sem(KO_mat)

WT_d2 = np.vstack([second_derivative_1d(WT_mat[i]) for i in range(WT_mat.shape[0])])
KO_d2 = np.vstack([second_derivative_1d(KO_mat[i]) for i in range(KO_mat.shape[0])])

wt2_mean, wt2_sem, _ = sample_mean_and_sem(WT_d2)
ko2_mean, ko2_sem, _ = sample_mean_and_sem(KO_d2)

# =========================
# Plot
# =========================
plt.rcParams.update({"font.size": FONT_SIZE, "axes.linewidth": 2.0})

# RAW
fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=False)
pub_axes(ax)

ax.plot(grid, wt_mean, color=WT_COLOR, linewidth=LINEWIDTH, label=f"WT (n={WT_mat.shape[0]})")
ax.plot(grid, ko_mean, color=KO_COLOR, linewidth=LINEWIDTH, label=f"KO (n={KO_mat.shape[0]})")

ax.fill_between(grid, wt_mean-wt_sem, wt_mean+wt_sem, color=WT_COLOR, alpha=ALPHA_SEM, lw=0)
ax.fill_between(grid, ko_mean-ko_sem, ko_mean+ko_sem, color=KO_COLOR, alpha=ALPHA_SEM, lw=0)

ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
ax.set_ylabel("RAW mean (union clusters; BG-sub per sample when available)")
ax.invert_xaxis()
ax.set_xlim(WN_MAX, WN_MIN)

ymin = np.nanmin(np.r_[wt_mean - wt_sem, ko_mean - ko_sem])
ymax = np.nanmax(np.r_[wt_mean + wt_sem, ko_mean + ko_sem])
ax.set_ylim(ymin - 0.05, ymax + 0.18)

add_bands(ax, BANDS)
legend_outside_right(ax)

# Reserve space on the right for legend
fig.subplots_adjust(right=0.78)

raw_out = f"{out_prefix}_RAW_960_1800_step2_BANDS_square_LEGOUT.png"
fig.savefig(raw_out, dpi=DPI, bbox_inches="tight")
plt.close(fig)

# D2
fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=False)
pub_axes(ax)

ax.plot(grid, wt2_mean, color=WT_COLOR, linewidth=LINEWIDTH, label=f"WT (n={WT_d2.shape[0]})")
ax.plot(grid, ko2_mean, color=KO_COLOR, linewidth=LINEWIDTH, label=f"KO (n={KO_d2.shape[0]})")

ax.fill_between(grid, wt2_mean-wt2_sem, wt2_mean+wt2_sem, color=WT_COLOR, alpha=ALPHA_SEM, lw=0)
ax.fill_between(grid, ko2_mean-ko2_sem, ko2_mean+ko2_sem, color=KO_COLOR, alpha=ALPHA_SEM, lw=0)

ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
ax.set_ylabel("2nd-derivative mean (SG on sample means)")
ax.invert_xaxis()
ax.set_xlim(WN_MAX, WN_MIN)

lo, hi = ylim_robust(wt2_mean, wt2_sem, ko2_mean, ko2_sem, pad_frac=0.20, p_lo=2, p_hi=98)
ax.set_ylim(lo, hi)

add_bands(ax, BANDS)
legend_outside_right(ax)

fig.subplots_adjust(right=0.78)

d2_out = f"{out_prefix}_D2_960_1800_step2_BANDS_square_LEGOUT.png"
fig.savefig(d2_out, dpi=DPI, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(" -", raw_out)
print(" -", d2_out)
