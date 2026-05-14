# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 01:41:44 2025

@author: hrazeghikondela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.signal import savgol_filter

# ------------------------------ #
# Paths & basic params (edit as needed)
# ------------------------------ #
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
save_dir = os.path.join(input_dir, "UMAP_clustering_8Cluster")
os.makedirs(save_dir, exist_ok=True)

out_prefix = os.path.join(save_dir, "similarity_scatter_pub1")

# Savitzky–Golay (2nd-deriv) params
savgol_window = 11
polyorder = 3

# Subsampling of pixels per sample (for speed/visualization)
subsample_size = 750

# ------------------------------ #
# Utility functions
# ------------------------------ #

def blend_colors(color1, color2):
    rgb1 = np.array(mcolors.to_rgb(color1))
    rgb2 = np.array(mcolors.to_rgb(color2))
    return tuple((rgb1 + rgb2) / 2)

def balance_sims(sims_dict, random_state=50):
    """Equalize list lengths across dict keys by downsampling to min length."""
    rng = np.random.default_rng(random_state)
    min_len = min(len(v) for v in sims_dict.values() if len(v) > 0)
    balanced_sims = {}
    for k, v in sims_dict.items():
        v = np.array(v)
        if len(v) == 0:
            balanced_sims[k] = v
            continue
        if len(v) > min_len:
            balanced_sims[k] = rng.choice(v, size=min_len, replace=False)
        else:
            balanced_sims[k] = v
    return balanced_sims

def _orient_sign(spectra, ref_vec):
    """Flip any spectrum whose cosine to ref is negative (sign-consistent derivatives)."""
    if spectra.size == 0:
        return spectra
    ref_norm = np.linalg.norm(ref_vec)
    if not np.isfinite(ref_norm) or ref_norm == 0:
        # Nothing we can do—return unchanged.
        return spectra
    ref = ref_vec / ref_norm
    dots = spectra @ ref
    flip = dots < 0
    if np.any(flip):
        spectra[flip] *= -1.0
    return spectra

# ------------------------------ #
# Core: build similarities (with optional 2nd-deriv + sign fix)
# ------------------------------ #
def build_similarity_matrices_custom(
    sample_paths,
    sample_names,
    groups,
    selected_clusters,
    second_derivative=False,
    subsample_size=750,
    random_state=50,
    normalize_spectra=False,
    fix_derivative_sign=True,
    abs_cosine=False,
):
    """
    Returns:
      matrix:      (n_samples x n_samples) mean cosine similarity matrix
      sample_names (unchanged)
      sims:        dict of flattened similarities by pair-group
      sim_pairs:   dict mapping group -> list of (name1, name2, [pairwise sims])
    """
    rng = np.random.default_rng(random_state)
    spectra_all = {}
    staged = {}

    # 1) Load → (optional) 2nd-deriv → select clusters → NaN/Inf mask → subsample
    for path, name in zip(sample_paths, sample_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        data = np.load(path)
        spec = data["spectra"]  # (n_pixels, n_bands)

        if second_derivative:
            # SG 2nd derivative along spectral axis
            spec = savgol_filter(
                spec,
                window_length=savgol_window,
                polyorder=polyorder,
                deriv=2,
                axis=1
            )

        clusters = selected_clusters[name]
        mask = np.isin(data["cluster_labels"], clusters)
        spec_sel = spec[mask]

        # keep only finite rows
        mask_valid = np.all(np.isfinite(spec_sel), axis=1)
        spec_sel = spec_sel[mask_valid]

        # subsample for speed/plotting
        if len(spec_sel) > subsample_size:
            idx = rng.choice(len(spec_sel), size=subsample_size, replace=False)
            spec_sel = spec_sel[idx]

        staged[name] = spec_sel  # keep pre-normalization

    # 2) Build robust reference and orient signs (derivative case only)
    if fix_derivative_sign and second_derivative:
        all_for_ref = np.vstack([v for v in staged.values() if len(v) > 0])
        if all_for_ref.shape[0] >= 5:
            ref_vec = np.median(all_for_ref, axis=0)
        elif all_for_ref.shape[0] >= 1:
            ref_vec = np.mean(all_for_ref, axis=0)
        else:
            ref_vec = None

        if ref_vec is not None and np.all(np.isfinite(ref_vec)):
            for name, spec_sel in staged.items():
                if spec_sel.size:
                    staged[name] = _orient_sign(spec_sel.copy(), ref_vec)

    # 3) Optional L2 normalization (row-wise), preserves orientation
    for name, spec_sel in staged.items():
        if normalize_spectra and len(spec_sel) > 0:
            spec_sel = normalize(spec_sel, axis=1)
        spectra_all[name] = spec_sel

    # 4) Compute similarities
    n = len(sample_names)
    matrix = np.zeros((n, n))
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}
    sim_pairs = {"WT-WT": [], "KO-KO": [], "WT-KO": []}

    for i, name1 in enumerate(sample_names):
        for j, name2 in enumerate(sample_names):
            s1 = spectra_all[name1]
            s2 = spectra_all[name2]
            if len(s1) == 0 or len(s2) == 0:
                matrix[i, j] = np.nan
                continue

            sim_matrix = cosine_similarity(s1, s2)
            if abs_cosine:
                sim_matrix = np.abs(sim_matrix)
            sims_pair = sim_matrix.ravel().tolist()
            matrix[i, j] = np.nanmean(sims_pair)

            g1 = "WT" if name1 in groups["WT"] else "KO"
            g2 = "WT" if name2 in groups["WT"] else "KO"
            key = f"{g1}-{g2}" if g1 == g2 else "WT-KO"
            sims[key].extend(sims_pair)
            sim_pairs[key].append((name1, name2, sims_pair))

    return matrix, sample_names, sims, sim_pairs

# --- Build per-sample matrices in exactly the same way as your analysis ---
def build_signed_secderiv_matrices(sample_paths, sample_names, selected_clusters,
                                   window=savgol_window, poly=polyorder,
                                   subsample_size=subsample_size, random_state=40,
                                   do_normalize=True):
    rng = np.random.default_rng(random_state)

    # 1) load & 2nd-deriv & select & subsample (pre-normalization)
    staged = {}
    for path, name in zip(sample_paths, sample_names):
        data = np.load(path)
        spec = data["spectra"]
        spec = savgol_filter(spec, window_length=window, polyorder=poly, deriv=2, axis=1)

        mask = np.isin(data["cluster_labels"], selected_clusters[name])
        spec = spec[mask]
        spec = spec[np.all(np.isfinite(spec), axis=1)]
        if len(spec) > subsample_size:
            spec = spec[rng.choice(len(spec), size=subsample_size, replace=False)]

        staged[name] = spec

    # 2) make global robust reference and orient signs
    all_for_ref = np.vstack([v for v in staged.values() if len(v) > 0])
    ref_vec = np.median(all_for_ref, axis=0) if all_for_ref.shape[0] >= 5 else np.mean(all_for_ref, axis=0)
    ref = ref_vec / (np.linalg.norm(ref_vec) + 1e-12)

    X = {}
    for name, S in staged.items():
        if S.size == 0:
            X[name] = S
            continue
        dots = S @ ref
        S[dots < 0] *= -1.0
        if do_normalize:
            S = normalize(S, axis=1)
        X[name] = S
    return X  # dict: name -> (n_pixels, n_bands)


# ------------------------------ #
# Plotting (meta-hue overlay)
# ------------------------------ #
def plot_meta_color_scatter(
    raw_sims_pairs,
    sec_sims_pairs,
    groups,
    fraction=0.9,
    random_state=50,
    figsize=(7, 7),
    font_family="Arial",
    base_fontsize=11,
    out_prefix=None
):
    rng = np.random.default_rng(random_state)

    # Set up per-sample colors
    wt_colors = sns.color_palette("Greens", n_colors=len(groups["WT"]))
    ko_colors = sns.color_palette("Reds",   n_colors=len(groups["KO"]))
    sample_color = {name: col for name, col in zip(groups["WT"], wt_colors)}
    sample_color.update({name: col for name, col in zip(groups["KO"], ko_colors)})

    # Aesthetic defaults
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "figure.dpi": 300,
    })

    def scatter_with_meta(ax, sim_pairs, title):
        jitter_scale = 0.06
        marker_s = 6
        alpha = 0.35
        positions = {"WT-WT": 1, "KO-KO": 2, "WT-KO": 3}

        for key, pairs in sim_pairs.items():
            pos = positions[key]
            for n1, n2, sims_list in pairs:
                sims = np.array(sims_list, dtype=float)
                if sims.size == 0:
                    continue
                sims = sims[np.isfinite(sims)]
                if sims.size == 0:
                    continue

                subsample_n = max(10, int(len(sims) * fraction))
                subsampled = rng.choice(sims, size=min(subsample_n, len(sims)), replace=False)

                x = np.random.normal(pos, jitter_scale, size=subsample_n)
                color_pair = blend_colors(sample_color[n1], sample_color[n2])
                ax.scatter(x, subsampled, color=color_pair, alpha=alpha,
                           s=marker_s, edgecolors='none')

        ax.set_xlim(0.4, 3.6)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(-1.0, 1.0)
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ---- Raw spectra ----
    fig1, ax1 = plt.subplots(figsize=figsize)
    scatter_with_meta(ax1, raw_sims_pairs, "Raw Spectra Similarities (Meta-hue Overlay)")
    plt.tight_layout()
    if out_prefix:
        fig1.savefig(out_prefix + "_raw_meta.pdf", bbox_inches='tight')
        fig1.savefig(out_prefix + "_raw_meta.png", bbox_inches='tight', dpi=600)
        print(f"Saved: {out_prefix}_raw_meta.*")
    plt.show()
    plt.close(fig1)

    # ---- Second Derivative ----
    fig2, ax2 = plt.subplots(figsize=figsize)
    scatter_with_meta(ax2, sec_sims_pairs, "2nd Derivative Similarities (Meta-hue Overlay)")
    plt.tight_layout()
    if out_prefix:
        fig2.savefig(out_prefix + "_sec_meta.pdf", bbox_inches='tight')
        fig2.savefig(out_prefix + "_sec_meta.png", bbox_inches='tight', dpi=600)
        print(f"Saved: {out_prefix}_sec_meta.*")
    plt.show()
    plt.close(fig2)
    


def pair_median_matrix(sim_pairs, sample_names):
    """
    Build a symmetric matrix M[i,j] = median(pixelwise cosine similarities)
    using the (name1, name2, sims_list) tuples in sim_pairs[*].
    """
    name_to_idx = {n:i for i,n in enumerate(sample_names)}
    n = len(sample_names)
    M = np.full((n, n), np.nan, dtype=float)
    # Fill from all groups
    for _, pairs in sim_pairs.items():
        for n1, n2, sims in pairs:
            if n1 not in name_to_idx or n2 not in name_to_idx: 
                continue
            v = np.asarray(sims, float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            m = float(np.median(v))
            i, j = name_to_idx[n1], name_to_idx[n2]
            M[i, j] = M[j, i] = m
    np.fill_diagonal(M, 1.0)
    return M

def plot_similarity_heatmap(M, sample_names, groups, title, out_path_prefix,
                            cluster=False, cmap="vlag"):
    """
    If cluster=False: order samples by group (WT first), then by mean similarity.
    If cluster=True: hierarchical cluster rows/cols (keeps labels).
    Saves PNG/PDF.
    """
    # Build group colors for side bar
    group_label = ["WT" if n in groups["WT"] else "KO" for n in sample_names]
    lut = {"WT":"#2ca02c", "KO":"#d62728"}  # green/red
    row_colors = [lut[g] for g in group_label]

    if cluster:
        cg = sns.clustermap(
            M, method="average", metric="euclidean", cmap=cmap, center=np.nanmedian(M),
            xticklabels=sample_names, yticklabels=sample_names,
            row_colors=row_colors, col_colors=row_colors, linewidths=0.3, figsize=(9,8)
        )
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=90)
        cg.ax_heatmap.set_title(title)
        cg.savefig(out_path_prefix + "_clustered_heatmap.png", dpi=300, bbox_inches="tight")
        cg.savefig(out_path_prefix + "_clustered_heatmap.pdf", bbox_inches="tight")
        plt.close(cg.fig)
    else:
        # Order by group then mean similarity (descending)
        order = np.argsort([0 if g=="WT" else 1 for g in group_label]
                           + np.zeros(len(sample_names)))  # initial WT/KO block
        # within each block, sort by row mean
        wt_idx = [i for i,n in enumerate(sample_names) if n in groups["WT"]]
        ko_idx = [i for i,n in enumerate(sample_names) if n in groups["KO"]]
        wt_sorted = sorted(wt_idx, key=lambda i: -np.nanmean(M[i]), reverse=False)
        ko_sorted = sorted(ko_idx, key=lambda i: -np.nanmean(M[i]), reverse=False)
        order = wt_sorted + ko_sorted

        M_ord = M[np.ix_(order, order)]
        names_ord = [sample_names[i] for i in order]
        row_colors_ord = [("WT" if n in groups["WT"] else "KO") for n in names_ord]
        row_colors_ord = [lut[g] for g in row_colors_ord]

        plt.figure(figsize=(9,8))
        ax = sns.heatmap(
            M_ord, cmap=cmap, center=np.nanmedian(M), vmin=0, vmax=1,
            xticklabels=names_ord, yticklabels=names_ord,
            square=True, cbar_kws={"label":"Median cosine"}
        )
        plt.setp(ax.get_xticklabels(), rotation=90)
        ax.set_title(title)
        # Add a simple legend for group colors
        for spine in ["top","right"]:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_path_prefix + "_heatmap.png", dpi=300, bbox_inches="tight")
        plt.savefig(out_path_prefix + "_heatmap.pdf", bbox_inches="tight")
        plt.close()


# ------------------------------
# Example Usage
# ------------------------------
sample_names = ["T1","T3","T6","T17","T19","T20","T21",
                "T10","T11","T12","T13","T14","T15","T22"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T3","T6","T17","T19","T20","T21"],
    "KO": ["T10","T11","T12","T13","T14","T15","T22"]
}

#Left
#selected_clusters = {
#    "T1":[0,7],
#   "T2": [3],
#    "T3": [2],
#    "T6": [1],
#    "T10": [1],
#    "T11": [2,4],
#    "T12": [2,3],
#    "T13": [0,4],
#    "T14": [1,5],
#    "T15": [5],
#}

#Left
selected_clusters = {
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
    "T14": [0,1,4,5,7],# 
    "T15": [0,2,3,4,5,7],
    "T22": [0,1,4,5,6],
}
#Right
#selected_clusters = {
#    "T1":[0,7],
#    "T2": [7],
#    "T3": [0],
    #"T4": [5,7],
    #"T5": [1],
#    "T6": [2,4],
    #T7": [1,2,3,4,5],
    #"T8": [1,2,3,4,5],
    #T9": [1,2,3,4,5],
#    "T10": [2,6],
#    "T11": [2,7],
#    "T12": [3,6],
#    "T13": [3,5],
#    "T14": [2,7],
#    "T15": [2],
   #"T16": [1,2,3,4,5]
#}

#Right
#selected_clusters = {
#    "T1":[1,2,3,4,5,6],
#    "T2": [1,2,3,4,5,6],
#    "T3": [1,2,3,4,5,6,7],
##    #"T4": [5,7],
#    "T5": [0,2,4,5,6,7],
#    "T6": [0,1,3,5,6,7],
#    "T17": [1,3,5,6,7],
    #"T18": [0,1,2,4,7],
#    "T19": [1,2,3,4.6,7],
#    "T20": [0,2,3,5,7],
    #T7": [1,2,3,4,5],
    #"T8": [1,2,3,4,5],
#   #T9": [1,2,3,4,5],
#   "T10": [0,1,3,4,5,7],
#    "T11": [0,1,3,4,5,6],
#    "T12": [0,1,2,4,5,7],
#    "T13": [0,1,2,4,6,7],
#    "T14": [0,1,3,4,5,6],
#    "T15": [0,1,3,4,5,6,7]
    
#}

# Compute similarity matrices
raw_matrix, names, raw_sims, raw_pairs = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=False,
    subsample_size=subsample_size,
    random_state=50,
    normalize_spectra=True,
    fix_derivative_sign=False,   # raw: no sign fix needed
    abs_cosine=False
)

sec_matrix, _, sec_sims, sec_pairs = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=True,
    subsample_size=subsample_size,
    random_state=40,
    normalize_spectra=True,
    fix_derivative_sign=True,    # <-- enable the fix here
    abs_cosine=False             # set True only for a quick sanity check
)

def mean_spec(S):  # S: (pixels x bands)
    return np.nanmean(S, axis=0)

# Build matrices
X = build_signed_secderiv_matrices(sample_paths, sample_names, selected_clusters)

# Centroids from sample means (each sample contributes equally)
WT_means = [mean_spec(X[n]) for n in sample_names if n in groups["WT"] and X[n].size]
KO_means = [mean_spec(X[n]) for n in sample_names if n in groups["KO"] and X[n].size]
WT_centroid = np.nanmean(WT_means, axis=0)
KO_centroid = np.nanmean(KO_means, axis=0)

def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return np.nan
    return float((a @ b) / (na * nb))

# Score each sample by cosine to WT and KO centroids
scores = {}
for n in sample_names:
    m = mean_spec(X[n]) if X[n].size else None
    if m is None: 
        scores[n] = (np.nan, np.nan)
        continue
    scores[n] = (cos(m, WT_centroid), cos(m, KO_centroid))

# Print a compact table (sorted by WT−KO score difference)
rows = []
for n in sample_names:
    wt, ko = scores[n]
    rows.append((n, "WT" if n in groups["WT"] else "KO", wt, ko, (wt - ko)))
rows.sort(key=lambda r: r[4])  # most KO-like at top if negative
print("Sample  Group   cos→WT   cos→KO   (WT−KO)")
for n,g,wt,ko,d in rows:
    print(f"{n:6s}  {g:3s}   {wt: .4f}   {ko: .4f}    {d: .4f}")
    
# Convert centroid cosines to angles (radians and milliradians)
ang = {}
for n,(cwt, cko) in scores.items():
    if not np.isfinite(cwt) or not np.isfinite(cko):
        ang[n] = (np.nan, np.nan, np.nan)
        continue
    th_wt = np.arccos(np.clip(cwt, -1, 1))
    th_ko = np.arccos(np.clip(cko, -1, 1))
    # positive => WT-like (smaller angle to WT)
    margin_mrad = (th_ko - th_wt) * 1e3
    ang[n] = (th_wt*1e3, th_ko*1e3, margin_mrad)

# Build a sorted bar plot by angle margin (mrad)
bar = sorted([(n, ("WT" if n in groups["WT"] else "KO"),
               ang[n][0], ang[n][1], ang[n][2])  # (θWT_mrad, θKO_mrad, margin_mrad)
              for n in sample_names if np.isfinite(ang[n][2])],
             key=lambda r: r[4], reverse=True)

print("\nSample  Group   θWT(mrad)  θKO(mrad)  margin(mrad = θKO−θWT)")
for n,g,twt,tko,mm in bar:
    print(f"{n:6s}  {g:3s}   {twt:8.3f}   {tko:8.3f}    {mm:8.3f}")

# Visual
plt.figure(figsize=(8.5,4.5))
x = np.arange(len(bar))
margins = [mm for _,_,_,_,mm in bar]
cols = ["#2ca02c" if g=="WT" else "#d62728" for _,g,_,_,_ in bar]
plt.axhline(0, ls="--", lw=1)
plt.bar(x, margins, color=cols, edgecolor="k", alpha=0.85)
plt.xticks(x, [n for n,_,_,_,_ in bar], rotation=90)
plt.ylabel("Angle margin (mrad): θKO − θWT\n(+ ⇒ WT-like, − ⇒ KO-like)")
plt.title("Centroid leaning (2nd-deriv, sign-corrected)")
plt.tight_layout()
plt.savefig(out_prefix + "_centroid_margins.png", dpi=300)
plt.show()


# Plot
plot_meta_color_scatter(
    raw_pairs, sec_pairs, groups,
    fraction=0.6, random_state=50,
    figsize=(10, 9), font_family='Arial',
    base_fontsize=11, out_prefix=out_prefix
)

# Build matrices of per-pair medians
M_raw = pair_median_matrix(raw_pairs, sample_names)
M_sec = pair_median_matrix(sec_pairs, sample_names)

# Plot & save (group-ordered)
plot_similarity_heatmap(
    M_raw, sample_names, groups,
    title="Raw Spectra – Pairwise Median Similarities",
    out_path_prefix=out_prefix + "_raw",
    cluster=False
)

plot_similarity_heatmap(
    M_sec, sample_names, groups,
    title="2nd Derivative – Pairwise Median Similarities",
    out_path_prefix=out_prefix + "_sec",
    cluster=False
)

# (Optional) clustered versions
plot_similarity_heatmap(
    M_sec, sample_names, groups,
    title="2nd Derivative – Clustered Median Similarities",
    out_path_prefix=out_prefix + "_sec",
    cluster=True
)

#%%
# 2nd Derivative plot

# ---------------------------
# Directories and sample info
# ---------------------------
cluster_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L\UMAP_clustering_8Cluster"
cube_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
save_dir = os.path.join(cube_dir, "WT_KO_pubplot1")
os.makedirs(save_dir, exist_ok=True)

savgol_window = 11
polyorder = 3
ALPHA  = 0.10      # significance level
BPERM  = 5000      # number of permutations

# ---------------------------
# Helper function
# ---------------------------
def get_sample_mean(sample):
    """
    Compute mean 2nd derivative spectrum for selected clusters.
    Automatically pads or trims spectra to match expected wavenumber length.
    """
    cluster_file = os.path.join(cluster_dir, f"{sample}_umap_kmeans.npz")
    cube_file = os.path.join(cube_dir, f"masked_cube_{sample}.npz")

    if not os.path.exists(cluster_file) or not os.path.exists(cube_file):
        print(f"⚠️ Missing files for {sample}")
        return None

    # Load cluster info
    clust = np.load(cluster_file)
    labels = clust["cluster_labels"]          # shape = (n_clustered_pixels,)
    indices = clust["pixel_indices"]          # shape = (n_clustered_pixels, 2) or flat
    sel_clusters = selected_clusters.get(sample, [])

    # Load cube
    cube = np.load(cube_file)["data"]         # shape = (Y, X, n_bands)
    Y, X, Z = cube.shape

    # --- Align spectra length to expected 426 ---
    expected_bands = 426
    if Z < expected_bands:
        missing = expected_bands - Z
        print(f"⚠️ {sample}: missing {missing} bands – padding with edge values at the beginning")
        edge = cube[:, :, :1]                 # take first band
        pad  = np.repeat(edge, missing, axis=2)
        cube = np.concatenate([pad, cube], axis=2)
        Z = expected_bands
    
    elif Z > expected_bands:
        print(f"⚠️ {sample}: has {Z - expected_bands} extra bands – trimming from start")
        cube = cube[:, :, -expected_bands:]
        Z = expected_bands

    # Convert 2D indices to flat indices if needed
    if indices.ndim == 2 and indices.shape[1] == 2:
        rows = indices[:, 0]
        cols = indices[:, 1]
        flat_indices = rows * X + cols
    else:
        flat_indices = indices

    # Mask pixels by cluster selection
    mask_clusters = np.isin(labels, sel_clusters)
    selected_indices = flat_indices[mask_clusters]

    if len(selected_indices) == 0:
        print(f"⚠️ No pixels in selected clusters for {sample}")
        return None

    # Flatten cube spatially
    flat_cube = cube.reshape(-1, Z)
    selected_pixels = flat_cube[selected_indices, :]

    # Remove NaN rows
    selected_pixels = selected_pixels[~np.isnan(selected_pixels).any(axis=1), :]

    if selected_pixels.size == 0:
        print(f"⚠️ No valid pixels for {sample}")
        return None

    # 2nd derivative
    sec_deriv = savgol_filter(selected_pixels, window_length=savgol_window,
                              polyorder=polyorder, deriv=2, axis=1)
    mean_spec = np.nanmean(sec_deriv, axis=0)

    wn = np.linspace(950, 1800, expected_bands)
    return mean_spec, wn

# 1) Gather sample means from your get_sample_mean()
# --- after you defined `get_sample_mean` and after `selected_clusters` ---
# Make sure these point to the folders holding your files:

# ^^^ change these two lines if your files live elsewhere

# Build sample-level mean 2nd-derivative spectra + shared wavenumber axis
means = {}
wn_shared = None
for n in sample_names:
    out = get_sample_mean(n)            # <-- pass ONLY the sample name
    if out is None:
        continue
    m, wn = out
    means[n] = m
    if wn_shared is None:
        wn_shared = wn

print(f"Built means for {len(means)} samples:", sorted(means.keys()))

# 2) Split KO into KO-A (KO-like) vs KO-B (WT-like KO) from centroid margins
#    (θKO − θWT); positive => WT-like, negative => KO-like
KO_like = []   # KO-A
WT_like = []   # KO-B
for n in groups["KO"]:
    cwt, cko = scores[n]
    th_wt = np.arccos(np.clip(cwt, -1, 1))
    th_ko = np.arccos(np.clip(cko, -1, 1))
    if (th_ko - th_wt) < 0:
        KO_like.append(n)
    else:
        WT_like.append(n)

print("KO-A (KO-like):", KO_like)
print("KO-B (WT-like KO):", WT_like)

# 3) Stack helpers
def stack(names):
    return np.vstack([means[n] for n in names if n in means and np.all(np.isfinite(means[n]))])

WT_stack   = stack(groups["WT"])
ALLKO_stack= stack(groups["KO"])
KOA_stack  = stack(KO_like)
KOB_stack  = stack(WT_like)

def maxT_permutation(groupA, groupB, B=5000, strata=None, stat="diff", rng_seed=0):
    """
    groupA/B: arrays shaped (n_samples, n_bands) for the two groups.
    strata: optional 1D array of batch labels per *sample* (len = nA + nB). If given,
            labels are permuted *within* each stratum (batch-controlled).
    stat: "diff" = mean(A) - mean(B); "welch" = Welch t (unequal vars).
    Returns:
      obs: observed stat per band
      p_maxT: max-T FWER-corrected p-values per band
    """
    rng = np.random.default_rng(rng_seed)
    X = np.vstack([groupA, groupB])                          # (n, p)
    nA, nB = groupA.shape[0], groupB.shape[0]
    idxA = np.arange(nA); idxB = np.arange(nA, nA+nB)

    def compute_stat(A, B):
        if stat == "welch":
            mA, mB = A.mean(0), B.mean(0)
            vA, vB = A.var(0, ddof=1), B.var(0, ddof=1)
            nA_, nB_ = A.shape[0], B.shape[0]
            denom = np.sqrt(vA/nA_ + vB/nB_)
            denom[denom == 0] = np.nan
            return (mA - mB) / denom
        else:  # simple difference of means
            return A.mean(0) - B.mean(0)

    obs = compute_stat(X[idxA], X[idxB])

    # build permutation plan
    if strata is None:
        strata = np.zeros(X.shape[0], dtype=int)

    unique_batches = np.unique(strata)
    max_stats = np.empty(B)
    for b in range(B):
        perm_idxA = []
        perm_idxB = []
        # permute labels within each batch
        for bt in unique_batches:
            in_batch = np.where(strata == bt)[0]
            # how many A's originally in this batch?
            nA_bt = np.sum(np.isin(in_batch, idxA))
            # permute within-batch indices and split
            perm = rng.permutation(in_batch)
            perm_idxA.extend(perm[:nA_bt])
            perm_idxB.extend(perm[nA_bt:])
        Aperm = X[np.array(perm_idxA)]
        Bperm = X[np.array(perm_idxB)]
        stat_perm = compute_stat(Aperm, Bperm)
        max_stats[b] = np.nanmax(np.abs(stat_perm))

    # max-T p-values
    p_maxT = (np.sum(np.abs(obs)[None, :] <= max_stats[:, None], axis=0))
    p_maxT = 1.0 - (p_maxT / B)  # convert "count of <= max" to tail prob
    # safer, standard estimate:
    p_maxT = (np.sum(max_stats[:, None] >= np.abs(obs)[None, :], axis=0) + 1) / (B + 1)
    return obs, p_maxT

def plot_sig(wn, obs, pvals, alpha=0.05, title="", outpath=None):
    sig = pvals < alpha
    plt.figure(figsize=(10,3))
    plt.plot(wn, obs, lw=1)
    if np.any(sig):
        plt.fill_between(wn, 0, obs, where=sig, alpha=0.3, step="mid", label=f"FWER<{alpha}")
    plt.axhline(0, ls="--", lw=0.8, alpha=0.7)
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Mean diff (WT − KO)")
    plt.title(title)
    if np.any(sig): plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=300)
    plt.show()
    return sig


# ---- derive KO-A vs KO-B from your centroid margins (already computed) ----
# margin_mrad = θKO − θWT; positive => WT-like, negative => KO-like
# Use the 'ang' dict from earlier; otherwise compute from 'scores' here:
ang = {}
for n,(cwt, cko) in scores.items():
    th_wt = np.arccos(np.clip(cwt, -1, 1))
    th_ko = np.arccos(np.clip(cko, -1, 1))
    ang[n] = (th_wt, th_ko, (th_ko - th_wt))  # radians

KO_like = [n for n in groups["KO"] if ang[n][2] < 0]   # KO-A (main KO)
WT_like = [n for n in groups["KO"] if ang[n][2] >= 0]  # KO-B (WT-like KO)
print("KO-A:", KO_like)
print("KO-B:", WT_like)

# ---- build matrices per comparison ----
def stack(names):
    arrs = [means[n] for n in names if n in means and np.all(np.isfinite(means[n]))]
    return np.vstack(arrs) if len(arrs) else None

WT_stack    = stack(groups["WT"])
ALLKO_stack = stack(groups["KO"])
KOA_stack   = stack(KO_like)   # from your centroid split
KOB_stack   = stack(WT_like)   # from your centroid split

comparisons = [
    ("WT − All KO", WT_stack, ALLKO_stack, out_prefix + "_maxt_WT_vs_AllKO.png"),
    ("WT − KO-A",   WT_stack, KOA_stack,   out_prefix + "_maxt_WT_vs_KOA.png"),
    ("WT − KO-B",   WT_stack, KOB_stack,   out_prefix + "_maxt_WT_vs_KOB.png"),
]

for title, A_stack, B_stack, path in comparisons:
    # skip if either group is missing or too small
    if A_stack is None or B_stack is None or A_stack.shape[0] < 2 or B_stack.shape[0] < 2:
        print(f"⚠️ Skipping '{title}' (insufficient samples): "
              f"A={None if A_stack is None else A_stack.shape}, "
              f"B={None if B_stack is None else B_stack.shape}")
        continue

    # run max-T on these stacks
    obs, p = maxT_permutation(A_stack, B_stack, B=BPERM, strata=None, stat="diff")

    # plot with the chosen alpha
    _ = plot_sig(
        wn_shared, obs, p, alpha=ALPHA,
        title=title + f" (max-T, α={ALPHA})",
        outpath=path
    )
# (optional) batch labels for batch-controlled permutations
# Provide a dict meta[name]['batch'] -> any hashable batch id
# If you don't have it, leave 'strata=None'.
strata_WT_ALLKO = None
# Example if you do have meta:
# strata_WT_ALLKO = np.array([meta[n]['batch'] for n in (groups['WT']+groups['KO'])])

# ---- run max-T permutations ----
B = 5000
obs_WT_vs_ALLKO, p_WT_vs_ALLKO = maxT_permutation(WT_stack, ALLKO_stack, B=B, strata=strata_WT_ALLKO, stat="diff")
obs_WT_vs_KOA,   p_WT_vs_KOA   = maxT_permutation(WT_stack, KOA_stack,   B=B, strata=None,                 stat="diff")
obs_WT_vs_KOB,   p_WT_vs_KOB   = maxT_permutation(WT_stack, KOB_stack,   B=B, strata=None,                 stat="diff")

sig1 = plot_sig(wn, obs_WT_vs_ALLKO, p_WT_vs_ALLKO, alpha=0.05,
                title="WT − All KO (max-T significant bands)",
                outpath=out_prefix + "_maxt_WT_vs_AllKO.png")

sig2 = plot_sig(wn, obs_WT_vs_KOA, p_WT_vs_KOA, alpha=0.05,
                title="WT − KO-A (KO-like) (max-T significant bands)",
                outpath=out_prefix + "_maxt_WT_vs_KOA.png")

sig3 = plot_sig(wn, obs_WT_vs_KOB, p_WT_vs_KOB, alpha=0.05,
                title="WT − KO-B (WT-like KO) (max-T significant bands)",
                outpath=out_prefix + "_maxt_WT_vs_KOB.png")

# 4) Run max-T permutations (mean-difference statistic)


print(f"Significant bands (FWER<0.05): "
      f"WT vs AllKO={sig1.sum()}, WT vs KO-A={sig2.sum()}, WT vs KO-B={sig3.sum()}")

# ---------------------------
# Process all samples
# ---------------------------
sample_means = {}
wn = None

for group, samples in groups.items():
    for s in samples:
        res = get_sample_mean(s)
        if res is not None:
            mean_spec, wn = res
            sample_means[s] = mean_spec

# ---------------------------
# Compute group mean & std
# ---------------------------
group_data = {}
for g, samples in groups.items():
    spectra_list = [sample_means[s] for s in samples if s in sample_means]
    if len(spectra_list) == 0:
        print(f"⚠️ No valid spectra for {g}")
        continue
    spectra_array = np.vstack(spectra_list)  # shape = n_samples x n_bands
    group_data[g] = {
        "mean": np.nanmean(spectra_array, axis=0),
        "std": np.nanstd(spectra_array, axis=0)
    }

# ---------------------------
# Plot
# ---------------------------
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "figure.dpi": 300})
fig, ax = plt.subplots(figsize=(10,6))
colors = {"WT": "#2ca02c", "KO": "#d62728"}

for g in group_data:
    mean = group_data[g]["mean"]
    std = group_data[g]["std"]
    ax.plot(wn, mean, color=colors[g], label=g)
    ax.fill_between(wn, mean-std, mean+std, color=colors[g], alpha=0.3)

ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("2nd derivative absorbance")
ax.set_title("WT vs KO - 2nd Derivative Spectra (Selected Clusters)")
# ax.invert_xaxis()  # <-- remove this line to have 950 → 1800
ax.legend()
ax.grid(linestyle='--', linewidth=0.4, alpha=0.5)

plt.tight_layout()

# Save figure
pdf_file = os.path.join(save_dir, "WT_KO_2nd_deriv.pdf")
png_file = os.path.join(save_dir, "WT_KO_2nd_deriv.png")
fig.savefig(pdf_file, bbox_inches='tight')
fig.savefig(png_file, bbox_inches='tight', dpi=600)
print(f"Saved: {pdf_file}\n       {png_file}")
plt.show()

# ---------------------------
# Plot subregions separately
# ---------------------------
subregions = {
    "950–1300 cm⁻¹": (950, 1300),
    "1300–1800 cm⁻¹": (1300, 1800)
}

for title, (start, end) in subregions.items():
    mask = (wn >= start) & (wn <= end)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for g in group_data:
        mean = group_data[g]["mean"][mask]
        std = group_data[g]["std"][mask]
        wn_sub = wn[mask]
        ax.plot(wn_sub, mean, color=colors[g], label=g)
        ax.fill_between(wn_sub, mean - std, mean + std, color=colors[g], alpha=0.3)

    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("2nd derivative absorbance")
    ax.set_title(f"WT vs KO - {title}")
    ax.legend()
    ax.grid(linestyle='--', linewidth=0.4, alpha=0.5)
    # ax.invert_xaxis()  # keep commented to maintain left-to-right order (950→1800)

    plt.tight_layout()
    
    # Save separate region plots
    fname_base = f"WT_KO_2nd_deriv_{int(start)}_{int(end)}"
    pdf_path = os.path.join(save_dir, f"{fname_base}.pdf")
    png_path = os.path.join(save_dir, f"{fname_base}.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight', dpi=600)
    print(f"Saved subregion: {pdf_path}\n                {png_path}")

    plt.show()

