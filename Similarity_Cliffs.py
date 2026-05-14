# -*- coding: utf-8 -*-
"""
Similarity-only analysis (FAST) — UPDATED (Group bars on heatmaps)
- RAW + 2nd-deriv cosine similarity (optional sign correction for derivatives)
- Exact mean pairwise cosine via sample "mean of unit rows"
- Distributions via random pair sampling (max_pairs_dist)
- Pub-style plots:
    (A) cosine similarity meta-hue scatter
    (B) log10(1 - cosine + eps) meta-hue scatter
    (C) heatmaps of median sampled similarities
- Cliff’s delta + Mann–Whitney on:
    (1) cosine similarity distributions
    (2) log10(1 - cosine + eps) distributions
- Excludes SELF-PAIRS from distribution sampling

UPDATES:
- Heatmaps fixed to 0.97–1.00
- Heatmap colormap is sequential (Reds by default)
- WT/KO shown explicitly via top + left group color bars (WT=green, KO=red)
- Legend matches those bars
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter
from scipy.stats import mannwhitneyu


# ------------------------------ #
# Paths & basic params (edit as needed)
# ------------------------------ #
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
save_dir = os.path.join(input_dir, "UMAP_clustering_8Cluster")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "similarity_scatter_pub2")

# Savitzky–Golay (2nd-deriv) params
savgol_window = 15
polyorder = 3

# Performance knobs
subsample_size = 300
max_pairs_dist = 20000
rng_master_seed = 10

# For log-distance plots
EPS_DIST = 1e-6


# ------------------------------ #
# Heatmap display choices
# ------------------------------ #
HEATMAP_VMIN = 0.97
HEATMAP_VMAX = 1.00
HEATMAP_TICKS = [0.97, 0.98, 0.99, 1.00]
HEATMAP_CMAP = "Reds"   # change to "Greens" if you want


# ------------------------------ #
# Group colors (WT/KO)
# ------------------------------ #
GROUP_LUT = {"WT": "#2ca02c", "KO": "#d62728"}  # WT green, KO red


# ------------------------------ #
# Pub-ish style
# ------------------------------ #
def apply_pub_style(font="Arial", base=11):
    plt.rcParams.update({
        "font.family": font,
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base + 1,
        "xtick.labelsize": base - 1,
        "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ------------------------------ #
# Utility helpers
# ------------------------------ #
def blend_colors(color1, color2):
    rgb1 = np.array(mcolors.to_rgb(color1))
    rgb2 = np.array(mcolors.to_rgb(color2))
    return tuple((rgb1 + rgb2) / 2)

def _clean_selected_clusters(raw_map, n_clusters=8):
    out = {}
    for name, vals in raw_map.items():
        ints = []
        for v in vals:
            try:
                vi = int(v)
            except Exception:
                continue
            if 0 <= vi < n_clusters:
                ints.append(vi)
        out[name] = sorted(set(ints))
    return out

def _orient_sign(spectra, ref_vec):
    """Flip any spectrum whose cosine to ref is negative (sign-consistent derivatives)."""
    if spectra.size == 0:
        return spectra
    ref_norm = np.linalg.norm(ref_vec)
    if not np.isfinite(ref_norm) or ref_norm == 0:
        return spectra
    ref = ref_vec / ref_norm
    dots = spectra @ ref
    flip = dots < 0
    if np.any(flip):
        spectra[flip] *= -1.0
    return spectra

def balance_sims(sims_dict, random_state=50):
    """Equalize list lengths across dict keys by downsampling to min length."""
    rng = np.random.default_rng(random_state)
    nonempty = [len(v) for v in sims_dict.values() if len(v) > 0]
    if len(nonempty) == 0:
        return {k: np.asarray(v) for k, v in sims_dict.items()}
    min_len = min(nonempty)
    balanced = {}
    for k, v in sims_dict.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            balanced[k] = v
        elif v.size > min_len:
            balanced[k] = rng.choice(v, size=min_len, replace=False)
        else:
            balanced[k] = v
    return balanced


# ---------- Cliff's delta helpers ----------
def cliffs_delta_1d(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    a_sort = np.sort(a)
    b_sort = np.sort(b)
    less = np.searchsorted(b_sort, a_sort, side='left')
    greater = (b_sort.size - np.searchsorted(b_sort, a_sort, side='right'))
    d = (less.sum() - greater.sum()) / (a_sort.size * b_sort.size)
    return float(d)

def bootstrap_cliffs_ci(a, b, n_boot=2000, seed=0, ci=0.95):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan, (np.nan, np.nan)
    obs = cliffs_delta_1d(a, b)
    if np.isnan(obs):
        return np.nan, (np.nan, np.nan)
    boots = np.empty(n_boot, float)
    na, nb = a.size, b.size
    for i in range(n_boot):
        aa = a[rng.integers(0, na, na)]
        bb = b[rng.integers(0, nb, nb)]
        boots[i] = cliffs_delta_1d(aa, bb)
    lo = np.quantile(boots, (1-ci)/2)
    hi = np.quantile(boots, 1-(1-ci)/2)
    return obs, (lo, hi)

def effect_bin(di):
    if not np.isfinite(di): return "nan"
    ad = abs(di)
    if ad < 0.147: lab = "negligible"
    elif ad < 0.33: lab = "small"
    elif ad < 0.474: lab = "medium"
    else: lab = "large"
    arrow = "A>B" if di > 0 else ("A<B" if di < 0 else "A≈B")
    return f"{lab} ({arrow})"


# ---------- Fast math helpers ----------
def _unit_row_normalize(X):
    if X.size == 0:
        return X.astype(np.float32)
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def _sample_repr(X):
    if X.size == 0:
        return None
    return _unit_row_normalize(X).mean(axis=0).astype(np.float32, copy=False)

def sample_pairwise_cosines(A, B, max_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    if A.size == 0 or B.size == 0:
        return np.array([], dtype=np.float32)
    na, nb = A.shape[0], B.shape[0]
    P = min(max_pairs, na * nb)
    ia = rng.integers(0, na, P, dtype=np.int32)
    ib = rng.integers(0, nb, P, dtype=np.int32)
    return np.einsum('ij,ij->i', A[ia], B[ib], dtype=np.float32)


# ------------------------------ #
# Core: build similarities
# ------------------------------ #
def build_similarity_matrices_custom(
    sample_paths,
    sample_names,
    groups,
    selected_clusters,
    second_derivative=False,
    subsample_size=300,
    random_state=50,
    normalize_spectra=True,
    fix_derivative_sign=True,
    abs_cosine=False,
    max_pairs_dist=20000,
    exclude_self_from_distributions=True,
):
    rng = np.random.default_rng(random_state)
    staged = {}
    spectra_all = {}

    for path, name in zip(sample_paths, sample_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        data = np.load(path, mmap_mode='r')
        spec = data["spectra"]
        labels = data["cluster_labels"]

        clusters = selected_clusters.get(name, [])
        if not clusters:
            staged[name] = np.empty((0, spec.shape[1]), dtype=np.float32)
            continue

        mask = np.isin(labels, clusters)
        spec_sel = spec[mask]
        if spec_sel.size == 0:
            staged[name] = np.empty((0, spec.shape[1]), dtype=np.float32)
            continue

        spec_sel = spec_sel[np.all(np.isfinite(spec_sel), axis=1)]
        if len(spec_sel) > subsample_size:
            idx = rng.choice(len(spec_sel), size=subsample_size, replace=False)
            spec_sel = spec_sel[idx]

        if second_derivative and spec_sel.size:
            spec_sel = savgol_filter(
                spec_sel, window_length=savgol_window,
                polyorder=polyorder, deriv=2, axis=1
            )

        staged[name] = np.asarray(spec_sel, dtype=np.float32, order='C')

    if fix_derivative_sign and second_derivative:
        nonempty = [v for v in staged.values() if v.size > 0]
        if len(nonempty) > 0:
            all_for_ref = np.vstack(nonempty)
            ref_vec = np.median(all_for_ref, axis=0) if all_for_ref.shape[0] >= 5 else np.mean(all_for_ref, axis=0)
            if np.all(np.isfinite(ref_vec)):
                for name, spec_sel in staged.items():
                    if spec_sel.size:
                        staged[name] = _orient_sign(spec_sel.copy(), ref_vec)

    for name, spec_sel in staged.items():
        if normalize_spectra and spec_sel.size:
            spec_sel = _unit_row_normalize(spec_sel)
        spectra_all[name] = spec_sel

    repr_vecs = {name: _sample_repr(spectra_all[name]) for name in sample_names}

    n = len(sample_names)
    matrix = np.full((n, n), np.nan, dtype=np.float32)
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}
    sim_pairs = {"WT-WT": [], "KO-KO": [], "WT-KO": []}

    for i, name1 in enumerate(sample_names):
        A = spectra_all[name1]
        v1 = repr_vecs[name1]
        if A.size == 0 or v1 is None:
            continue

        for j in range(i, n):
            name2 = sample_names[j]
            B = spectra_all[name2]
            v2 = repr_vecs[name2]
            if B.size == 0 or v2 is None:
                continue

            mean_ij = float(np.dot(v1, v2))
            matrix[i, j] = matrix[j, i] = mean_ij

            if exclude_self_from_distributions and (i == j):
                continue

            sims_pair = sample_pairwise_cosines(
                A, B, max_pairs=max_pairs_dist,
                seed=(hash((name1, name2, random_state)) & 0xffffffff)
            )
            if abs_cosine:
                sims_pair = np.abs(sims_pair)

            g1 = "WT" if name1 in groups["WT"] else "KO"
            g2 = "WT" if name2 in groups["WT"] else "KO"
            key = f"{g1}-{g2}" if g1 == g2 else "WT-KO"
            sims[key].extend(sims_pair.tolist())
            sim_pairs[key].append((name1, name2, sims_pair.tolist()))

    return matrix, sample_names, sims, sim_pairs


def build_signed_secderiv_matrices(sample_paths, sample_names, selected_clusters,
                                   window=savgol_window, poly=polyorder,
                                   subsample_size=subsample_size, random_state=40,
                                   do_normalize=True):
    rng = np.random.default_rng(random_state)
    staged = {}
    for path, name in zip(sample_paths, sample_names):
        if not os.path.exists(path):
            staged[name] = np.empty((0, 0), dtype=np.float32)
            continue
        data = np.load(path, mmap_mode='r')
        spec = data["spectra"]
        labels = data["cluster_labels"]

        sel = selected_clusters.get(name, [])
        if not sel:
            staged[name] = np.empty((0, 0), dtype=np.float32)
            continue

        mask = np.isin(labels, sel)
        S = spec[mask]
        S = S[np.all(np.isfinite(S), axis=1)]
        if len(S) > subsample_size:
            S = S[rng.choice(len(S), size=subsample_size, replace=False)]
        if S.size == 0:
            staged[name] = np.empty((0, 0), dtype=np.float32)
            continue

        S = savgol_filter(S, window_length=window, polyorder=poly, deriv=2, axis=1)
        staged[name] = np.asarray(S, dtype=np.float32)

    nonempty = [v for v in staged.values() if v.size > 0]
    ref = None
    if len(nonempty) > 0:
        all_for_ref = np.vstack(nonempty)
        ref_vec = np.median(all_for_ref, axis=0) if all_for_ref.shape[0] >= 5 else np.mean(all_for_ref, axis=0)
        ref = ref_vec / (np.linalg.norm(ref_vec) + 1e-12)

    X = {}
    for name, S in staged.items():
        if S.size == 0:
            X[name] = S
            continue
        if ref is not None:
            dots = S @ ref
            S[dots < 0] *= -1.0
        if do_normalize:
            S = normalize(S, axis=1).astype(np.float32, copy=False)
        X[name] = S
    return X


# ------------------------------ #
# Stats summaries
# ------------------------------ #
def summarize_delta(title, sims_dict, transform=None, balance=True, seed=123, csv_path=None):
    transform = transform or (lambda x: x)
    dct = balance_sims(sims_dict, random_state=seed) if balance else sims_dict

    comps = [("WT-WT", "KO-KO"), ("WT-WT", "WT-KO"), ("KO-KO", "WT-KO")]
    rows = []
    print(f"\n=== Cliff's delta + Mann–Whitney: {title} ===")

    for A, B in comps:
        a = np.asarray(dct.get(A, []), float); a = a[np.isfinite(a)]
        b = np.asarray(dct.get(B, []), float); b = b[np.isfinite(b)]
        a = transform(a); b = transform(b)
        a = np.asarray(a, float); a = a[np.isfinite(a)]
        b = np.asarray(b, float); b = b[np.isfinite(b)]

        if a.size < 2 or b.size < 2:
            msg = f"{A} vs {B}: insufficient data (|A|={a.size}, |B|={b.size})"
            print(msg)
            rows.append([A, B, a.size, b.size, np.nan, np.nan, np.nan, np.nan, msg])
            continue

        d, (lo, hi) = bootstrap_cliffs_ci(a, b, n_boot=2000, seed=seed, ci=0.95)
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided", method="asymptotic")
        except TypeError:
            _, p = mannwhitneyu(a, b, alternative="two-sided")

        note = effect_bin(d)
        print(f"{A:>5s} vs {B:<5s} | nA={a.size:6d}, nB={b.size:6d} | "
              f"Cliff's d = {d:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}] -> {note} | "
              f"MWU p = {p:.3e}")

        rows.append([A, B, a.size, b.size, d, lo, hi, p, ""])

    if csv_path is not None:
        hdr = "groupA,groupB,nA,nB,cliffs_d,ci_lo,ci_hi,mannwhitney_p,note"
        np.savetxt(csv_path, np.array(rows, dtype=object), fmt="%s",
                   delimiter=",", header=hdr, comments="")
        print("Saved:", csv_path)


# ------------------------------ #
# Scatter plot (meta-hue)
# ------------------------------ #
def plot_meta_color_scatter(
    sim_pairs,
    groups,
    y_mode="cosine",
    fraction=0.3,
    random_state=50,
    figsize=(7.5, 7.0),
    out_path=None,
    title=None
):
    apply_pub_style(base=11)
    rng = np.random.default_rng(random_state)

    wt_colors = sns.color_palette("Greens", n_colors=len(groups["WT"]))
    ko_colors = sns.color_palette("Reds",   n_colors=len(groups["KO"]))
    sample_color = {name: col for name, col in zip(groups["WT"], wt_colors)}
    sample_color.update({name: col for name, col in zip(groups["KO"], ko_colors)})

    positions = {"WT-WT": 1, "KO-KO": 2, "WT-KO": 3}
    jitter_scale = 0.06
    marker_s = 7
    alpha = 0.35

    def transform_y(y):
        y = np.asarray(y, float)
        if y_mode == "cosine":
            return y
        d = 1.0 - y
        d = np.clip(d, EPS_DIST, None)
        return np.log10(d)

    fig, ax = plt.subplots(figsize=figsize)

    for key, pairs in sim_pairs.items():
        pos = positions[key]
        for n1, n2, sims_list in pairs:
            sims = np.array(sims_list, dtype=float)
            sims = sims[np.isfinite(sims)]
            if sims.size == 0:
                continue

            subsample_n = max(20, int(len(sims) * fraction))
            subsample_n = min(subsample_n, sims.size)
            subsampled = rng.choice(sims, size=subsample_n, replace=False)
            yy = transform_y(subsampled)

            x = rng.normal(pos, jitter_scale, size=subsample_n)
            cpair = blend_colors(sample_color.get(n1, (0.4,0.4,0.4)),
                                 sample_color.get(n2, (0.4,0.4,0.4)))
            ax.scatter(x, yy, color=cpair, alpha=alpha, s=marker_s,
                       edgecolors="none", rasterized=True)

    ax.set_xlim(0.4, 3.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)

    if y_mode == "cosine":
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(0.5, 1.0)
    else:
        ax.set_ylabel(r"log10(1 − cosine + eps)")

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path + ".png", bbox_inches="tight", dpi=600)
        fig.savefig(out_path + ".pdf", bbox_inches="tight")
        print("Saved:", out_path + ".[png|pdf]")
    plt.show()
    plt.close(fig)


# ------------------------------ #
# Heatmap helpers (UPDATED: group bars)
# ------------------------------ #
def pair_median_matrix(sim_pairs, sample_names):
    name_to_idx = {n:i for i,n in enumerate(sample_names)}
    n = len(sample_names)
    M = np.full((n, n), np.nan, dtype=float)
    for _, pairs in sim_pairs.items():
        for n1, n2, sims in pairs:
            v = np.asarray(sims, float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            m = float(np.median(v))
            i, j = name_to_idx[n1], name_to_idx[n2]
            M[i, j] = M[j, i] = m
    np.fill_diagonal(M, 1.0)
    return M

def plot_similarity_heatmap_with_groupbars(M, sample_names, groups, title, out_path_prefix,
                                          cluster=False, cmap=HEATMAP_CMAP):
    """
    cluster=False:
      - custom layout with top+left WT/KO bars + heatmap + colorbar + legend
    cluster=True:
      - seaborn clustermap with fixed spacing:
          * title as suptitle (no overlap)
          * colorbar moved to reserved right margin
          * legend in reserved right margin
    """
    apply_pub_style(base=11)

    group_label = ["WT" if n in groups["WT"] else "KO" for n in sample_names]
    handles = [
        mpatches.Patch(color=GROUP_LUT["WT"], label="WT"),
        mpatches.Patch(color=GROUP_LUT["KO"], label="KO"),
    ]

    if cluster:
        # -------------------- #
        # CLUSTERED CLUSTERMAP (FIXED LAYOUT)
        # -------------------- #
        row_colors = [GROUP_LUT[g] for g in group_label]
        col_colors = [GROUP_LUT[g] for g in group_label]

        # Bigger figure + controlled dendrogram and color-bar strip widths
        cg = sns.clustermap(
            M,
            method="average",
            metric="euclidean",
            cmap=cmap,
            vmin=HEATMAP_VMIN,
            vmax=HEATMAP_VMAX,
            xticklabels=sample_names,
            yticklabels=sample_names,
            row_colors=row_colors,
            col_colors=col_colors,
            linewidths=0.2,
            linecolor="white",
            figsize=(12.8, 8.0),
            dendrogram_ratio=(0.14, 0.14),
            colors_ratio=(0.03, 0.03),
            # Put colorbar in a reserved right margin; will fine-tune after adjust
            cbar_pos=(0.86, 0.22, 0.02, 0.55),
            cbar_kws={"ticks": HEATMAP_TICKS},
        )

        # Reserve top for title, right for colorbar+legend
        cg.fig.subplots_adjust(top=0.90, right=0.82)

        # Clear axis-title (prevents overlap with dendrograms)
        cg.ax_heatmap.set_title("")

        # Use figure-level title (safe)
        cg.fig.suptitle(title, y=0.96, fontsize=20)

        # Heatmap tick styling
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90, ha="center")
        plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0)
        cg.ax_heatmap.tick_params(axis="both", labelsize=14)

        # Colorbar styling (clean label + ticks)
        cg.cax.set_ylabel("Median cosine (sampled pairs)", fontsize=18, rotation=90, labelpad=18)
        cg.cax.tick_params(labelsize=14, width=1.2, length=6)

        # Move colorbar slightly right (so it never touches y tick labels)
        # (Because clustermap sometimes reflows positions after subplots_adjust)
        cg.cax.set_position([0.84, 0.22, 0.02, 0.55])

        # Legend goes in the reserved right margin (not on top of anything)
        cg.fig.legend(
            handles=handles,
            title="Group",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.88, 0.92),
            fontsize=18,
            title_fontsize=20,
        )

        cg.savefig(out_path_prefix + "_clustered_heatmap.png", dpi=300, bbox_inches="tight")
        cg.savefig(out_path_prefix + "_clustered_heatmap.pdf", bbox_inches="tight")
        plt.close(cg.fig)
        return

    # -------------------- #
    # NON-CLUSTER HEATMAP (your existing group-bar layout)
    # -------------------- #
    wt_idx = [i for i, n in enumerate(sample_names) if n in groups["WT"]]
    ko_idx = [i for i, n in enumerate(sample_names) if n in groups["KO"]]
    wt_sorted = sorted(wt_idx, key=lambda i: np.nanmean(M[i]))
    ko_sorted = sorted(ko_idx, key=lambda i: np.nanmean(M[i]))
    order = wt_sorted + ko_sorted

    M_ord = M[np.ix_(order, order)]
    names_ord = [sample_names[i] for i in order]
    glab_ord = [("WT" if n in groups["WT"] else "KO") for n in names_ord]
    grgb_ord = np.array([mcolors.to_rgb(GROUP_LUT[g]) for g in glab_ord], dtype=float)

    fig = plt.figure(figsize=(10.6, 8.2))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.35, 8.0],
        width_ratios=[0.35, 8.0],
        hspace=0.05, wspace=0.05
    )

    ax_top = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_hm = fig.add_subplot(gs[1, 1])

    # Top group bar
    ax_top.imshow(np.array([grgb_ord], dtype=float), aspect="auto")
    ax_top.set_xticks([]); ax_top.set_yticks([])
    ax_top.set_xlim(-0.5, len(names_ord) - 0.5)

    # Left group bar
    ax_left.imshow(np.array([grgb_ord]).transpose(1, 0, 2), aspect="auto")
    ax_left.set_xticks([]); ax_left.set_yticks([])
    ax_left.set_ylim(len(names_ord) - 0.5, -0.5)

    hm = sns.heatmap(
        M_ord,
        ax=ax_hm,
        cmap=cmap,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
        xticklabels=names_ord,
        yticklabels=names_ord,
        square=True,
        cbar=True,
        cbar_kws={
            "label": "Median cosine (sampled pairs)",
            "ticks": HEATMAP_TICKS,
            "shrink": 0.92,
            "pad": 0.02,
            "aspect": 30
        },
        linewidths=0.2,
        linecolor="white"
    )

    ax_hm.set_title(title, pad=18)
    plt.setp(ax_hm.get_xticklabels(), rotation=90)
    ax_hm.tick_params(axis="both", labelsize=14)

    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14, width=1.2, length=6)
    cbar.set_label("Median cosine (sampled pairs)", fontsize=18)

    fig.legend(handles=handles, title="Group", frameon=False,
               loc="upper right", bbox_to_anchor=(1.22, 0.92),
               fontsize=18, title_fontsize=20)

    fig.savefig(out_path_prefix + "_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path_prefix + "_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)



# ------------------------------ #
# INPUTS
# ------------------------------ #
sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

selected_clusters_raw = {
    "T1" :[0,1,3,4,6,7],
    "T2" : [0,1,2],
    "T3" : [0,1,4,5],
    "T6" : [0,1,3,4,6],
    "T19": [0,2,3.6,7],
    "T20": [0,1,2,4,5,7],
    "T21": [1,2,3,4],
    "T17": [0,1,2,3,5,6],
    "T10": [0,2,4,5,6,7],
    "T11": [0,1,2,3,6,7],
    "T12": [0,1,2,3,5,7],
    "T13": [1,2,5,6,7],
    "T14": [0,1,4,5,7],
    "T15": [0,2,3,4,5,7],
    "T22": [0,1,4,5,6],
}
selected_clusters = _clean_selected_clusters(selected_clusters_raw, n_clusters=8)

sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]


# ------------------------------ #
# Similarity computations
# ------------------------------ #
raw_matrix, names, raw_sims, raw_pairs = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=False,
    subsample_size=subsample_size,
    random_state=rng_master_seed,
    normalize_spectra=True,
    fix_derivative_sign=False,
    abs_cosine=False,
    max_pairs_dist=max_pairs_dist,
    exclude_self_from_distributions=True
)

sec_matrix, _, sec_sims, sec_pairs = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=True,
    subsample_size=subsample_size,
    random_state=rng_master_seed + 1,
    normalize_spectra=True,
    fix_derivative_sign=True,
    abs_cosine=False,
    max_pairs_dist=max_pairs_dist,
    exclude_self_from_distributions=True
)


# ------------------------------ #
# Stats: cosine similarity + log(1-cos)
# ------------------------------ #
summarize_delta(
    "RAW cosine similarity",
    raw_sims, balance=True, seed=123,
    csv_path=out_prefix + "_RAW_cliffs_cosine.csv"
)

summarize_delta(
    "SEC 2nd-derivative cosine similarity",
    sec_sims, balance=True, seed=123,
    csv_path=out_prefix + "_SEC_cliffs_cosine.csv"
)

log1m = lambda x: np.log10(np.clip(1.0 - x, EPS_DIST, None))

summarize_delta(
    "RAW log10(1 - cosine + eps)",
    raw_sims, transform=log1m, balance=True, seed=123,
    csv_path=out_prefix + "_RAW_cliffs_log1mcos.csv"
)

summarize_delta(
    "SEC log10(1 - cosine + eps)",
    sec_sims, transform=log1m, balance=True, seed=123,
    csv_path=out_prefix + "_SEC_cliffs_log1mcos.csv"
)


# ------------------------------ #
# Plots: cosine meta-hue + log(1-cos) meta-hue
# ------------------------------ #
plot_meta_color_scatter(
    raw_pairs, groups,
    y_mode="cosine",
    fraction=0.30, random_state=rng_master_seed,
    figsize=(8.5, 7.5),
    out_path=out_prefix + "_RAW_meta_cosine",
    title="RAW spectra — cosine similarities (meta-hue)"
)

plot_meta_color_scatter(
    sec_pairs, groups,
    y_mode="cosine",
    fraction=0.30, random_state=rng_master_seed + 1,
    figsize=(8.5, 7.5),
    out_path=out_prefix + "_SEC_meta_cosine",
    title="2nd-derivative — cosine similarities (meta-hue)"
)

plot_meta_color_scatter(
    raw_pairs, groups,
    y_mode="log1mcos",
    fraction=0.30, random_state=rng_master_seed,
    figsize=(8.5, 7.5),
    out_path=out_prefix + "_RAW_meta_log1mcos",
    title="RAW spectra — log10(1 − cosine) (meta-hue)"
)

plot_meta_color_scatter(
    sec_pairs, groups,
    y_mode="log1mcos",
    fraction=0.30, random_state=rng_master_seed + 1,
    figsize=(8.5, 7.5),
    out_path=out_prefix + "_SEC_meta_log1mcos",
    title="2nd-derivative — log10(1 − cosine) (meta-hue)"
)


# ------------------------------ #
# Heatmaps (median sampled cosine) — UPDATED with WT/KO group bars
# ------------------------------ #
M_raw = pair_median_matrix(raw_pairs, sample_names)
M_sec = pair_median_matrix(sec_pairs, sample_names)

plot_similarity_heatmap_with_groupbars(
    M_raw, sample_names, groups,
    title="RAW — Pairwise median cosine (sampled pairs)",
    out_path_prefix=out_prefix + "_raw",
    cluster=False,
    cmap=HEATMAP_CMAP
)

plot_similarity_heatmap_with_groupbars(
    M_sec, sample_names, groups,
    title="2nd-derivative — Pairwise median cosine (sampled pairs)",
    out_path_prefix=out_prefix + "_sec",
    cluster=False,
    cmap=HEATMAP_CMAP
)

plot_similarity_heatmap_with_groupbars(
    M_sec, sample_names, groups,
    title="2nd-derivative — Clustered median cosine (sampled pairs)",
    out_path_prefix=out_prefix + "_sec",
    cluster=True,
    cmap=HEATMAP_CMAP
)
