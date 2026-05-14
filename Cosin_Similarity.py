# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 11:27:52 2025

@author: hrazeghikondela
"""

#%%
# With Normalization
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter

# ===============================
# SETTINGS
# ===============================
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
save_dir = os.path.join(input_dir, "")
os.makedirs(save_dir, exist_ok=True)


# Derivative filter
savgol_window = 10
polyorder = 3

# Grayscale visualization band index (≈1301 cm⁻¹)
grayscale_band_index = 353


# -------------------------
# User-editable: set this
save_dir = os.path.join(input_dir, "UMAP_clustering_8Cluster")   # <<< change this to the folder with your T*_umap_kmeans.npz files
out_prefix = os.path.join(save_dir, "similarity_scatter_pub1")
# -------------------------

def balance_sims(sims_dict, random_state=40):
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

def build_similarity_matrices_custom(sample_paths, sample_names, groups, selected_clusters, 
                                     second_derivative=False, subsample_size=1300, random_state=40,
                                     normalize_spectra=False):
    rng = np.random.default_rng(random_state)
    spectra_all = {}

    for path, name in zip(sample_paths, sample_names):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        data = np.load(path)
        spec = data['spectra']  # expected shape (n_pixels, n_bands)

        if second_derivative:
            # 2nd derivative along spectral axis
            spec = savgol_filter(spec, window_length=savgol_window, polyorder=polyorder, deriv=2, axis=1)

        clusters = selected_clusters[name]
        mask = np.isin(data['cluster_labels'], clusters)
        spec_sel = spec[mask]

        # ---- exclude rows with NaN/Inf ----
        mask_valid = np.all(np.isfinite(spec_sel), axis=1)
        spec_sel = spec_sel[mask_valid]

        if len(spec_sel) > subsample_size:
            idx = rng.choice(len(spec_sel), size=subsample_size, replace=False)
            spec_sel = spec_sel[idx]

        if normalize_spectra and len(spec_sel) > 0:
            spec_sel = normalize(spec_sel, axis=1)

        spectra_all[name] = spec_sel

    n = len(sample_names)
    matrix = np.zeros((n, n))
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}

    for i, name1 in enumerate(sample_names):
        for j, name2 in enumerate(sample_names):
            s1 = spectra_all[name1]
            s2 = spectra_all[name2]

            if len(s1) == 0 or len(s2) == 0:
                matrix[i, j] = np.nan  # no valid spectra
                continue

            sim_matrix = cosine_similarity(s1, s2)
            sims_pair = sim_matrix.flatten().tolist()
            matrix[i, j] = np.mean(sims_pair)

            g1 = 'WT' if name1 in groups['WT'] else 'KO'
            g2 = 'WT' if name2 in groups['WT'] else 'KO'
            if g1 == g2 == 'WT':
                sims['WT-WT'].extend(sims_pair)
            elif g1 == g2 == 'KO':
                sims['KO-KO'].extend(sims_pair)
            else:
                sims['WT-KO'].extend(sims_pair)

    return matrix, sample_names, sims

def plot_publication_scatter(raw_sims, sec_sims, fraction=0.6, random_state=40,
                            figsize=(6, 6), font_family='Arial', base_fontsize=11,
                            out_prefix=None):
    rng = np.random.default_rng(random_state)

    # --- Publication style setup ---
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

    colors = {"WT-WT": "#2ca02c", "KO-KO": "#d62728", "WT-KO": "#1f77b4"}
    keys_order = ["WT-WT", "KO-KO", "WT-KO"]

    # Balance and optionally subsample
    raw_sims_bal = balance_sims(raw_sims, random_state=random_state)
    sec_sims_bal = balance_sims(sec_sims, random_state=random_state)

    # Shared plotting parameters
    jitter_scale = 0.06
    marker_s = 6
    alpha = 0.25

    # =============================
    # --- Plot 1: Raw Similarities
    # =============================
    fig1, ax1 = plt.subplots(figsize=figsize)
    for idx, key in enumerate(keys_order, start=1):
        sims = np.array(raw_sims_bal[key])
        if sims.size == 0:
            continue
        subsample_n = max(10, int(len(sims) * fraction))
        subsampled = rng.choice(sims, size=subsample_n, replace=False)
        x = np.random.normal(idx, jitter_scale, size=subsample_n)
        ax1.scatter(x, subsampled, color=colors[key], alpha=alpha, s=marker_s, edgecolors='none')

    ax1.set_xlim(0.4, 3.6)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
    ax1.set_ylabel("Cosine similarity")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_title("Raw spectra similarities (normalized; 60% sampled)")
    ax1.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    handles = [
        plt.Line2D([], [], marker='o', color=colors[k], linestyle='',
                   markersize=6, label=k)
        for k in keys_order
    ]
    ax1.legend(handles=handles, frameon=True, loc='center left',
               bbox_to_anchor=(1.02, 0.5), borderpad=0.4, handletextpad=0.6)

    plt.tight_layout()
    if out_prefix:
        raw_pdf = out_prefix + "_raw.pdf"
        raw_png = out_prefix + "_raw.png"
        fig1.savefig(raw_pdf, bbox_inches='tight')
        fig1.savefig(raw_png, bbox_inches='tight', dpi=600)
        print(f"Saved: {raw_pdf}\n       {raw_png}")
    plt.show()
    plt.close(fig1)

    # =====================================
    # --- Plot 2: Second Derivative Similarities
    # =====================================
    fig2, ax2 = plt.subplots(figsize=figsize)
    for idx, key in enumerate(keys_order, start=1):
        sims = np.array(sec_sims_bal[key])
        if sims.size == 0:
            continue
        subsample_n = max(10, int(len(sims) * fraction))
        subsampled = rng.choice(sims, size=subsample_n, replace=False)
        x = np.random.normal(idx, jitter_scale, size=subsample_n)
        ax2.scatter(x, subsampled, color=colors[key], alpha=alpha, s=marker_s, edgecolors='none')

    ax2.set_xlim(0.4, 3.6)
    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
    ax2.set_ylabel("Cosine similarity")
    ax2.set_ylim(-1.0, 1.0)
    ax2.set_title("2nd derivative similarities (normalized; 60% sampled)")
    ax2.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.4)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    handles = [
        plt.Line2D([], [], marker='o', color=colors[k], linestyle='',
                   markersize=6, label=k)
        for k in keys_order
    ]
    ax2.legend(handles=handles, frameon=True, loc='center left',
               bbox_to_anchor=(1.02, 0.5), borderpad=0.4, handletextpad=0.6)

    plt.tight_layout()
    if out_prefix:
        sec_pdf = out_prefix + "_secderiv.pdf"
        sec_png = out_prefix + "_secderiv.png"
        fig2.savefig(sec_pdf, bbox_inches='tight')
        fig2.savefig(sec_png, bbox_inches='tight', dpi=600)
        print(f"Saved: {sec_pdf}\n       {sec_png}")
    plt.show()
    plt.close(fig2)



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
    "T6" : [0,1,3,4,6,7],
    "T19": [0,2,3.6,7],
   "T20": [0,1,2,4,5,7],
   "T21": [0,1,2,3,4],
    "T17": [0,1,2,3,5,6],
    "T10": [0,2,4,5,6,7],
    "T11": [0,1,2,3,6,7],
    "T12": [0,1,2,3,5,7],
    "T13": [1,2,4,5,6,7],
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



# Build sims (normalize_spectra=True like your original "Normalized")
raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=False, subsample_size=1300, random_state=40,
    normalize_spectra=True
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=True, subsample_size=1300, random_state=40,
    normalize_spectra=True
)

# Plot and save publication-ready figure
plot_publication_scatter(raw_sims, sec_sims, fraction=0.60, random_state=40,
                         figsize=(10,9), font_family='Arial', base_fontsize=11,
                         out_prefix=out_prefix)

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
        print(f"⚠️ {sample}: missing {missing} bands – padding with NaNs at the beginning")
        pad = np.full((Y, X, missing), np.nan)
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

