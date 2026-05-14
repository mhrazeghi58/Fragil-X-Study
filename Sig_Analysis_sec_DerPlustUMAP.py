# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 18:35:03 2025

@author: hrazeghikondela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import ListedColormap
import umap

# ===============================
# SETTINGS
# ===============================
input_dir = r"D:\Filiz Lab\Data_2025\Mid-IR Data\11_18_2025\Area_EX"
save_dir = os.path.join(input_dir, "UMAP_clustering_12Cluster")
os.makedirs(save_dir, exist_ok=True)

# Full wavenumber axis
wavenumbers = np.linspace(950, 1800, 426)

# Target bands ± window
targets = {
    "C=O stretch (ester, lipid)": 1740,
    "Amide I band": 1655,
    "Amide II band": 1545,
    "CH₂ scissoring": 1464,
    "CH₃ symmetric bending": 1375
}
window = 20  # cm⁻¹

# Build mask for selected bands
mask = np.zeros_like(wavenumbers, dtype=bool)
for _, center in targets.items():
    mask |= (wavenumbers >= (center - window)) & (wavenumbers <= (center + window))
selected_wavenumbers = wavenumbers[mask]

# UMAP parameters
n_neighbors = 9
min_dist = 0.2
n_clusters = 10

# Derivative filter
savgol_window = 10
polyorder = 3

# Grayscale visualization band index (≈1301 cm⁻¹)
grayscale_band_index = 353

# ===============================
# PART 1: PROCESS EACH SAMPLE
# ===============================
for fname in sorted(os.listdir(input_dir)):
    if not fname.startswith("masked_cube_T") or not fname.endswith(".npz"):
        continue

    sample_name = fname.replace("masked_cube_", "").replace(".npz", "")
    file_path = os.path.join(input_dir, fname)

    try:
        with np.load(file_path) as npzfile:
            data = npzfile["data"]  # shape: (H, W, Bands)
    except Exception as e:
        print(f"❌ Skipping {fname}: {e}")
        continue

    print(f"\n🚀 Starting {sample_name} – data shape: {data.shape}")
    H, W, B = data.shape
    flat_features, pixel_indices = [], []

    # Compute 2nd derivative + extract only selected bands
    print("   🔹 Extracting spectra and computing 2nd derivative...")
    for i in range(H):
        for j in range(W):
            spectrum = data[i, j, :][mask]  # only selected bands
            if np.isnan(spectrum).any():
                continue
            second_deriv = savgol_filter(spectrum, window_length=savgol_window,
                                         polyorder=polyorder, deriv=2)
            flat_features.append(second_deriv)
            pixel_indices.append((i, j))

    flat_features = np.array(flat_features)
    if flat_features.shape[0] == 0:
        print(f"⚠ No valid pixels in {sample_name}")
        continue

    # --- UMAP & KMeans ---
    print("   🔹 Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=3, random_state=0,
                        n_neighbors=n_neighbors, min_dist=min_dist)
    umap_result = reducer.fit_transform(flat_features)

    print("   🔹 Running KMeans clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(umap_result)
    cluster_labels = kmeans.labels_

    # --- Build 2D cluster map ---
    print("   🔹 Building cluster map...")
    cluster_map = np.full((H, W), np.nan)
    for (i, j), label in zip(pixel_indices, cluster_labels):
        cluster_map[i, j] = label

    # --- Plot and save cluster map ---
    print("   📊 Saving cluster map image...")
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cluster_map, cmap="tab10", interpolation="nearest")
    plt.title(f"{sample_name} – UMAP + KMeans Clustering", fontsize=14)
    plt.colorbar(im, label="Cluster Label")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{sample_name}_cluster_map.tiff"),
                dpi=600, bbox_inches="tight")
    plt.close()

    # --- Plot and save UMAP scatter ---
    print("   📊 Saving UMAP scatter plot...")
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1],
                          c=cluster_labels, cmap="tab10", s=0.5, alpha=0.4)
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.title(f"{sample_name} – UMAP Scatter", fontsize=14)
    plt.colorbar(scatter, label="Cluster Label")
    plt.savefig(os.path.join(save_dir, f"{sample_name}_umap_scatter.tiff"),
                dpi=600, bbox_inches="tight")
    plt.close()

    # --- Prepare grayscale base image ---
    print("   🔹 Preparing grayscale overlay images...")
    grayscale_img = data[:, :, grayscale_band_index]
    grayscale_img = np.nan_to_num(grayscale_img)
    grayscale_img = (grayscale_img - np.min(grayscale_img)) / (np.max(grayscale_img) - np.min(grayscale_img))

    # --- Overlay each cluster separately ---
    for k in range(n_clusters):
        overlay = np.where(cluster_map == k, k + 1, 0)
        masked_overlay = np.ma.masked_where(overlay == 0, overlay)

        plt.figure(figsize=(8, 6))
        plt.imshow(grayscale_img, cmap="gray", interpolation="nearest")
        plt.imshow(masked_overlay, cmap=ListedColormap(["red"]), alpha=0.5, interpolation="nearest")
        plt.title(f"{sample_name} – Cluster {k} Overlay", fontsize=12)
        plt.axis("off")
        overlay_path = os.path.join(save_dir, f"{sample_name}_cluster_{k}_overlay.tiff")
        plt.savefig(overlay_path, dpi=600, bbox_inches="tight")
        plt.close()

    # --- Save results (npz) ---
    print("   💾 Saving spectra and cluster assignments...")
    cluster_spectra = []
    cluster_ids = []
    for (i, j), label in zip(pixel_indices, cluster_labels):
        spectrum = data[i, j, :][mask]  # restricted to bands
        cluster_spectra.append(spectrum)
        cluster_ids.append(label)

    cluster_spectra = np.array(cluster_spectra)
    cluster_ids = np.array(cluster_ids)

    umap_outfile = os.path.join(save_dir, f"{sample_name}_umap_kmeans.npz")
    np.savez_compressed(
        umap_outfile,
        cluster_labels=cluster_ids,
        umap_result=umap_result,
        pixel_indices=np.array(pixel_indices),
        spectra=cluster_spectra,
        wavenumbers=selected_wavenumbers
    )
    print(f"✅ Finished {sample_name} – results saved to {umap_outfile}")

#%%
# ===============================
# PART 2: COSINE SIMILARITY TEST
# ===============================

def compute_second_derivative(spectrum, window_length=11, polyorder=3):
    return savgol_filter(spectrum, window_length, polyorder, deriv=2)

def compute_average_spectrum_per_sample(npz_path, clusters_to_use=[0, 1], second_derivative=False):
    with np.load(npz_path) as f:
        spectra = f["spectra"]
        labels = f["cluster_labels"]
        wn = f["wavenumbers"]

    mask = np.isin(labels, clusters_to_use)
    selected_spectra = spectra[mask]
    if selected_spectra.shape[0] == 0:
        raise ValueError(f"No spectra found for clusters {clusters_to_use} in {npz_path}")

    avg = np.mean(selected_spectra, axis=0)
    if second_derivative:
        avg = compute_second_derivative(avg)
    return wn, avg

def build_similarity_matrices_custom(sample_paths, sample_names, groups, selected_clusters, second_derivative=False):
    avg_spectra = {}
    for path, name in zip(sample_paths, sample_names):
        clusters_to_use = selected_clusters.get(name, [0, 1])
        wn, avg = compute_average_spectrum_per_sample(path, clusters_to_use, second_derivative)
        avg_spectra[name] = avg

    names = list(avg_spectra.keys())
    matrix = cosine_similarity([avg_spectra[n] for n in names])

    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = matrix[i, j]
            if names[i] in groups["WT"] and names[j] in groups["WT"]:
                sims["WT-WT"].append(sim)
            elif names[i] in groups["KO"] and names[j] in groups["KO"]:
                sims["KO-KO"].append(sim)
            else:
                sims["WT-KO"].append(sim)

    return matrix, names, sims


from scipy.stats import ttest_ind

# ------------------------------
# Helper: add significance stars
# ------------------------------
def add_significance(ax, x1, x2, y, pval, h=0.002):
    """Draws a significance bar with stars between two boxplots."""
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    elif pval < 0.1:
        stars = "*"   # treat p<0.1 as *
    else:
        stars = "ns"

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c="k")
    ax.text((x1+x2)*0.5, y+h, stars, ha="center", va="bottom", color="k", fontsize=12)

# ------------------------------
# Main plotting function
# ------------------------------
def plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- Heatmap Raw ---
    im1 = axs[0, 0].imshow(raw_matrix, cmap="Greens", vmin=.99, vmax=1)
    axs[0, 0].set_xticks(range(len(names)))
    axs[0, 0].set_yticks(range(len(names)))
    axs[0, 0].set_xticklabels(names, rotation=90)
    axs[0, 0].set_yticklabels(names)
    axs[0, 0].set_title("Cosine Similarity – Raw Spectra")
    fig.colorbar(im1, ax=axs[0, 0], fraction=0.046)

    # --- Heatmap 2nd Derivative ---
    im2 = axs[0, 1].imshow(sec_matrix, cmap="Greens", vmin=0.99, vmax=1)
    axs[0, 1].set_xticks(range(len(names)))
    axs[0, 1].set_yticks(range(len(names)))
    axs[0, 1].set_xticklabels(names, rotation=90)
    axs[0, 1].set_yticklabels(names)
    axs[0, 1].set_title("Cosine Similarity – 2nd Derivative Spectra")
    fig.colorbar(im2, ax=axs[0, 1], fraction=0.046)

    # ---------------------------
    # Boxplot with points (Raw)
    # ---------------------------
    data_raw = [raw_sims["WT-WT"], raw_sims["KO-KO"], raw_sims["WT-KO"]]
    axs[1, 0].boxplot(data_raw, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)

    for i, d in enumerate(data_raw, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 0].plot(x, d, "o", color="black", alpha=0.6, markersize=4)

    axs[1, 0].set_title("Group Similarities – Raw Spectra")
    axs[1, 0].set_ylabel("Cosine Similarity")
    axs[1, 0].set_ylim(0.994, 1.005)

    # significance tests raw
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_max = max([max(d) for d in data_raw])
    for (i, j) in pairs:
        stat, p = ttest_ind(data_raw[i], data_raw[j], equal_var=False)
        add_significance(axs[1, 0], i+1, j+1, y_max + (pairs.index((i, j)) * 0.001), p)

    # ---------------------------
    # Boxplot with points (2nd Derivative)
    # ---------------------------
    data_sec = [sec_sims["WT-WT"], sec_sims["KO-KO"], sec_sims["WT-KO"]]
    axs[1, 1].boxplot(data_sec, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)

    for i, d in enumerate(data_sec, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 1].plot(x, d, "o", color="black", alpha=0.6, markersize=4)

    axs[1, 1].set_title("Group Similarities – 2nd Derivative Spectra")
    axs[1, 1].set_ylabel("Cosine Similarity")
    axs[1, 1].set_ylim(0.994, 1.005)

    # significance tests sec
    y_max = max([max(d) for d in data_sec])
    for (i, j) in pairs:
        stat, p = ttest_ind(data_sec[i], data_sec[j], equal_var=False)
        add_significance(axs[1, 1], i+1, j+1, y_max + (pairs.index((i, j)) * 0.001), p)

    plt.tight_layout()
    plt.show()


# ===============================
# Example Usage
# ===============================
sample_names = ["T1","T2","T3","T5","T6",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T5","T6"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
}

selected_clusters = {
    "T1":[1,2,3,4,5],
    "T2": [1,2,3,4,5],
    "T3": [1,2,3,4,5],
   #"T4": [1,2,3,4,5],
    "T5": [1,2,3,4,5],
    "T6": [1,2,3,4,5],
    #T7": [1,2,3,4,5],
   #"T8": [1,2,3,4,5],
   #T9": [1,2,3,4,5],
    "T10": [1,2,3,4,5],
    "T11": [1,2,3,4,5],
    "T12": [1,2,3,4,5],
    "T13": [1,2,3,4,5],
    "T14": [1,2,3,4,5],
    "T15": [1,2,3,4,5],
   #"T16": [1,2,3,4,5]
}

raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=False
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=True
)

plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims)


#%%



# ===============================
# Target bands
# ===============================
targets = {
   #"C=O stretch (ester, lipid)": 1740,
    "Amide I band": 1545,
   "Amide II band": 1545,
   "CH2 scissoring": 1464,
   "CH3 symmetric bending": 1375
}
window = 50 # cm⁻¹ ± window

# ------------------------------
# Helper: add significance stars
# ------------------------------
def add_significance(ax, x1, x2, y, pval, h=0.002):
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    elif pval < 0.1:
        stars = "*"   # permissive
    else:
        stars = "ns"

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c="k")
    ax.text((x1+x2)*0.5, y+h, stars, ha="center", va="bottom", color="k", fontsize=12)

# ------------------------------
# Select bands from spectra
# ------------------------------
def select_bands(spectrum, wn, targets, window):
    """Extracts spectral regions around target bands."""
    mask = np.zeros_like(wn, dtype=bool)
    for center in targets.values():
        mask |= (wn >= center - window) & (wn <= center + window)
    return spectrum[mask], wn[mask]

# ------------------------------
# Main plotting function
# ------------------------------
def plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims, wn=None):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- Heatmap Raw ---
    im1 = axs[0, 0].imshow(raw_matrix, cmap="Greens", vmin=.95, vmax=1)
    axs[0, 0].set_xticks(range(len(names)))
    axs[0, 0].set_yticks(range(len(names)))
    axs[0, 0].set_xticklabels(names, rotation=90)
    axs[0, 0].set_yticklabels(names)
    title_raw = "Cosine Similarity – Raw Spectra (Target Bands)"
    axs[0, 0].set_title(title_raw)
    fig.colorbar(im1, ax=axs[0, 0], fraction=0.046)

    # --- Heatmap 2nd Derivative ---
    im2 = axs[0, 1].imshow(sec_matrix, cmap="Greens", vmin=0.95, vmax=1)
    axs[0, 1].set_xticks(range(len(names)))
    axs[0, 1].set_yticks(range(len(names)))
    axs[0, 1].set_xticklabels(names, rotation=90)
    axs[0, 1].set_yticklabels(names)
    title_sec = "Cosine Similarity – 2nd Derivative Spectra (Target Bands)"
    axs[0, 1].set_title(title_sec)
    fig.colorbar(im2, ax=axs[0, 1], fraction=0.046)

    # ---------------------------
    # Boxplot with points (Raw)
    # ---------------------------
    data_raw = [raw_sims["WT-WT"], raw_sims["KO-KO"], raw_sims["WT-KO"]]
    axs[1, 0].boxplot(data_raw, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)

    for i, d in enumerate(data_raw, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 0].plot(x, d, "o", color="black", alpha=0.6, markersize=4)

    axs[1, 0].set_title("Group Similarities – Raw Spectra (Target Bands)")
    axs[1, 0].set_ylabel("Cosine Similarity")
    axs[1, 0].set_ylim(0.99, 1.005)

    # significance tests raw
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_max = max([max(d) for d in data_raw])
    for k, (i, j) in enumerate(pairs):
        stat, p = ttest_ind(data_raw[i], data_raw[j], equal_var=False)
        add_significance(axs[1, 0], i+1, j+1, y_max + (k * 0.001), p)

    # ---------------------------
    # Boxplot with points (2nd Derivative)
    # ---------------------------
    data_sec = [sec_sims["WT-WT"], sec_sims["KO-KO"], sec_sims["WT-KO"]]
    axs[1, 1].boxplot(data_sec, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)

    for i, d in enumerate(data_sec, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 1].plot(x, d, "o", color="black", alpha=0.6, markersize=4)

    axs[1, 1].set_title("Group Similarities – 2nd Derivative Spectra (Target Bands)")
    axs[1, 1].set_ylabel("Cosine Similarity")
    axs[1, 1].set_ylim(0.99, 1.005)

    # significance tests sec
    y_max = max([max(d) for d in data_sec])
    for k, (i, j) in enumerate(pairs):
        stat, p = ttest_ind(data_sec[i], data_sec[j], equal_var=False)
        add_significance(axs[1, 1], i+1, j+1, y_max + (k * 0.001), p)

    plt.tight_layout()
    plt.show()



# ===============================
# Example Usage
# ===============================
sample_names = ["T1","T2","T3","T5","T6",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T5","T6"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
}

selected_clusters = {
    "T1":[1,2,3,4,5],
    "T2": [1,2,3,4,5],
    "T3": [1,2,3,4,5],
   #"T4": [1,2,3,4,5],
    "T5": [1,2,3,4,5],
    "T6": [1,2,3,4,5],
    #T7": [1,2,3,4,5],
   #"T8": [1,2,3,4,5],
   #T9": [1,2,3,4,5],
    "T10": [1,2,3,4,5],
    "T11": [1,2,3,4,5],
    "T12": [1,2,3,4,5],
    "T13": [1,2,3,4,5],
    "T14": [1,2,3,4,5],
    "T15": [1,2,3,4,5],
   #"T16": [1,2,3,4,5]
}

raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=False
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=True
)

plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims)

#%%

#%%

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import savgol_filter

# ===============================
# Target bands
# ===============================
targets = {
    "Amide I band": 1545
}
window = 100  # cm⁻¹ ± window

# ------------------------------
# Helper: add significance stars
# ------------------------------
def add_significance(ax, x1, x2, y, pval, h=0.002):
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    elif pval < 0.1:
        stars = "*"   # permissive
    else:
        stars = "ns"

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c="k")
    ax.text((x1+x2)*0.5, y+h, stars, ha="center", va="bottom", color="k", fontsize=12)

# ------------------------------
# Select bands from spectra
# ------------------------------
def select_bands(spectrum, wn, targets, window):
    """Extracts spectral regions around target bands."""
    mask = np.zeros_like(wn, dtype=bool)
    for center in targets.values():
        mask |= (wn >= center - window) & (wn <= center + window)
    return spectrum[mask], wn[mask]

# ===============================
# Build similarity matrices
# ===============================
def build_similarity_matrices_custom(sample_paths, sample_names, groups, selected_clusters, second_derivative=False):
    avg_spectra = {}
    wn_all = None

    for path, name in zip(sample_paths, sample_names):
        data = np.load(path)
        wn = data['wavenumbers']
        wn_all = wn
        spec_raw = data['spectra']

        # compute second derivative if requested
        if second_derivative:
            # Savitzky-Golay smoothing + 2nd derivative along wavenumber axis
            spec = savgol_filter(spec_raw, window_length=11, polyorder=3, deriv=2, axis=1)
        else:
            spec = spec_raw

        # select clusters of interest
        clusters = selected_clusters[name]
        mask = np.isin(data['cluster_labels'], clusters)
        spec_sel = spec[mask]

        # select target bands
        spec_sel_bands = []
        for s in spec_sel:
            s_band, _ = select_bands(s, wn, targets, window)
            spec_sel_bands.append(s_band)

        # truncate to shortest length
        min_len = min([len(s) for s in spec_sel_bands])
        spec_truncated = np.array([s[:min_len] for s in spec_sel_bands])

        # average across clusters for this sample
        avg_spectra[name] = np.mean(spec_truncated, axis=0)

    # compute cosine similarity between average spectra
    names = list(avg_spectra.keys())
    matrix = cosine_similarity([avg_spectra[n] for n in names])

    # group-based similarities
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = matrix[i, j]
            if names[i] in groups["WT"] and names[j] in groups["WT"]:
                sims["WT-WT"].append(sim)
            elif names[i] in groups["KO"] and names[j] in groups["KO"]:
                sims["KO-KO"].append(sim)
            else:
                sims["WT-KO"].append(sim)

    return matrix, names, sims

# ===============================
# Plotting function
# ===============================
def plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- Heatmap Raw ---
    im1 = axs[0, 0].imshow(raw_matrix, cmap="Greens", vmin=.95, vmax=1)
    axs[0, 0].set_xticks(range(len(names)))
    axs[0, 0].set_yticks(range(len(names)))
    axs[0, 0].set_xticklabels(names, rotation=90)
    axs[0, 0].set_yticklabels(names)
    axs[0, 0].set_title("Cosine Similarity – Raw Spectra (Target Bands)")
    fig.colorbar(im1, ax=axs[0, 0], fraction=0.046)

    # --- Heatmap 2nd Derivative ---
    im2 = axs[0, 1].imshow(sec_matrix, cmap="Greens", vmin=0.95, vmax=1)
    axs[0, 1].set_xticks(range(len(names)))
    axs[0, 1].set_yticks(range(len(names)))
    axs[0, 1].set_xticklabels(names, rotation=90)
    axs[0, 1].set_yticklabels(names)
    axs[0, 1].set_title("Cosine Similarity – 2nd Derivative Spectra (Target Bands)")
    fig.colorbar(im2, ax=axs[0, 1], fraction=0.046)

    # Boxplot Raw
    data_raw = [raw_sims["WT-WT"], raw_sims["KO-KO"], raw_sims["WT-KO"]]
    axs[1, 0].boxplot(data_raw, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)
    for i, d in enumerate(data_raw, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 0].plot(x, d, "o", color="black", alpha=0.6, markersize=4)
    axs[1, 0].set_title("Group Similarities – Raw Spectra (Target Bands)")
    axs[1, 0].set_ylabel("Cosine Similarity")
    axs[1, 0].set_ylim(0.99, 1.005)

    # significance tests raw
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_max = max([max(d) for d in data_raw])
    for k, (i, j) in enumerate(pairs):
        stat, p = ttest_ind(data_raw[i], data_raw[j], equal_var=False)
        add_significance(axs[1, 0], i+1, j+1, y_max + (k * 0.001), p)

    # Boxplot 2nd derivative
    data_sec = [sec_sims["WT-WT"], sec_sims["KO-KO"], sec_sims["WT-KO"]]
    axs[1, 1].boxplot(data_sec, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)
    for i, d in enumerate(data_sec, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 1].plot(x, d, "o", color="black", alpha=0.6, markersize=4)
    axs[1, 1].set_title("Group Similarities – 2nd Derivative Spectra (Target Bands)")
    axs[1, 1].set_ylabel("Cosine Similarity")
    axs[1, 1].set_ylim(0.99, 1.005)

    # significance tests sec
    y_max = max([max(d) for d in data_sec])
    for k, (i, j) in enumerate(pairs):
        stat, p = ttest_ind(data_sec[i], data_sec[j], equal_var=False)
        add_significance(axs[1, 1], i+1, j+1, y_max + (k * 0.001), p)

    plt.tight_layout()
    plt.show()

# ===============================
# Example Usage
# ===============================
save_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Masked_Cubes"

sample_names = ["T1","T2","T3","T5","T6",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T5","T6"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
}


selected_clusters = {
    "T1":[1,2,3,4,5],
    "T2": [1,2,3,4,5],
    "T3": [1,2,3,4,5],
   #"T4": [1,2,3,4,5],
    "T5": [1,2,3,4,5],
    "T6": [1,2,3,4,5],
    #T7": [1,2,3,4,5],
   #"T8": [1,2,3,4,5],
   #T9": [1,2,3,4,5],
    "T10": [1,2,3,4,5],
    "T11": [1,2,3,4,5],
    "T12": [1,2,3,4,5],
    "T13": [1,2,3,4,5],
    "T14": [1,2,3,4,5],
    "T15": [1,2,3,4,5],
   #"T16": [1,2,3,4,5]
}

raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=False
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=True
)

plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims)

raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=False
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=True
)

plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims)


#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
#from sklearn.metrics.pairwise import cosine_similarity

# ====================================================
# Helper functions
# ====================================================
def compute_second_derivative(spectrum, window_length=11, polyorder=3):
    return savgol_filter(spectrum, window_length, polyorder, deriv=2)

def compute_average_spectrum_per_sample(npz_path, exclude_clusters=None, second_derivative=False):
    with np.load(npz_path) as f:
        spectra = f["spectra"]
        labels = f["cluster_labels"]
        wn = f["wavenumbers"]

    # Exclude selected clusters
    if exclude_clusters is None:
        mask = np.ones_like(labels, dtype=bool)  # keep all clusters
    else:
        mask = ~np.isin(labels, exclude_clusters)  # drop clusters

    selected_spectra = spectra[mask]
    if selected_spectra.shape[0] == 0:
        raise ValueError(f"No spectra left after excluding {exclude_clusters} in {npz_path}")

    avg = np.mean(selected_spectra, axis=0)
    if second_derivative:
        avg = compute_second_derivative(avg)
    return wn, avg

def build_similarity_matrices_custom(sample_paths, sample_names, groups, excluded_clusters, second_derivative=False):
    avg_spectra = {}
    for path, name in zip(sample_paths, sample_names):
        clusters_to_exclude = excluded_clusters.get(name, [])
        wn, avg = compute_average_spectrum_per_sample(path, clusters_to_exclude, second_derivative)
        avg_spectra[name] = avg

    names = list(avg_spectra.keys())
    matrix = cosine_similarity([avg_spectra[n] for n in names])

    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sim = matrix[i, j]
            if names[i] in groups["WT"] and names[j] in groups["WT"]:
                sims["WT-WT"].append(sim)
            elif names[i] in groups["KO"] and names[j] in groups["KO"]:
                sims["KO-KO"].append(sim)
            else:
                sims["WT-KO"].append(sim)

    return matrix, names, sims

def plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims, title_suffix=""):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- Heatmap for raw spectra ---
    im1 = axs[0, 0].imshow(raw_matrix, cmap="viridis", vmin=0.95, vmax=1)
    axs[0, 0].set_xticks(range(len(names)))
    axs[0, 0].set_yticks(range(len(names)))
    axs[0, 0].set_xticklabels(names, rotation=90)
    axs[0, 0].set_yticklabels(names)
    axs[0, 0].set_title(f"Cosine Similarity – Raw Spectra {title_suffix}")
    fig.colorbar(im1, ax=axs[0, 0], fraction=0.046)

    # --- Heatmap for 2nd derivative spectra ---
    im2 = axs[0, 1].imshow(sec_matrix, cmap="viridis", vmin=0.95, vmax=1)
    axs[0, 1].set_xticks(range(len(names)))
    axs[0, 1].set_yticks(range(len(names)))
    axs[0, 1].set_xticklabels(names, rotation=90)
    axs[0, 1].set_yticklabels(names)
    axs[0, 1].set_title(f"Cosine Similarity – 2nd Derivative {title_suffix}")
    fig.colorbar(im2, ax=axs[0, 1], fraction=0.046)

    # --- Boxplot for raw similarities ---
    axs[1, 0].boxplot([raw_sims["WT-WT"], raw_sims["KO-KO"], raw_sims["WT-KO"]],
                      labels=["WT-WT", "KO-KO", "WT-KO"])
    axs[1, 0].set_title(f"Group Similarities – Raw {title_suffix}")
    axs[1, 0].set_ylabel("Cosine Similarity")

    # --- Boxplot for 2nd derivative similarities ---
    axs[1, 1].boxplot([sec_sims["WT-WT"], sec_sims["KO-KO"], sec_sims["WT-KO"]],
                      labels=["WT-WT", "KO-KO", "WT-KO"])
    axs[1, 1].set_title(f"Group Similarities – 2nd Derivative {title_suffix}")
    axs[1, 1].set_ylabel("Cosine Similarity")

    plt.tight_layout()
    plt.show()


# ====================================================
# Example Usage
# ====================================================
sample_names = ["T1","T2","T3","T4","T6","T5","T8","T9",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T4","T5","T6","T8","T9"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
}

selected_clusters = {
    "T1":[0,7],
    "T2": [5,7],
    "T3": [0],
    "T4": [5,7],
    "T5": [1,3],
    "T6": [2,4],
    #"T7": [0, 2],
    "T8": [5,1],
    "T9": [0],
    "T10": [2,6],
    "T11": [2,5,7],
    "T12": [3,6],
    "T13": [3,5],
    "T14": [2,5,7],
    "T15": [2],
    #"T16": [0,6]
}
# Exclude these clusters per sample
excluded_clusters = {
    "T1":[0,7],
    "T2": [5,7],
    "T3": [0],
    "T4": [5,7],
    "T5": [1,3],
    "T6": [2,4],
    #"T7": [0, 2],
    "T8": [5,1],
    "T9": [0],
    "T10": [2,6],
    "T11": [2,5,7],
    "T12": [3,6],
    "T13": [3,5],
    "T14": [2,5,7],
    "T15": [2],
    #"T16": [0,6]
}

# --- Run analysis excluding selected clusters ---
raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, excluded_clusters, second_derivative=False
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, excluded_clusters, second_derivative=True
)
plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims, title_suffix="(excluded clusters)")

# --- Run analysis keeping all clusters (baseline comparison) ---
raw_matrix_all, names_all, raw_sims_all = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, excluded_clusters={}, second_derivative=False
)
sec_matrix_all, _, sec_sims_all = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, excluded_clusters={}, second_derivative=True
)
plot_results(raw_matrix_all, sec_matrix_all, names_all, raw_sims_all, sec_sims_all, title_suffix="(all clusters)")

#%%

import os
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# SETTINGS
# ===============================
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_R\UMAP_clustering_test"

# Group assignments (update with your real mapping)
wt_samples = ["T1", "T2", "T3", "T4", "T6", "T14", "T15", "T16"]
ko_samples = ["T5", "T8", "T9", "T10", "T11", "T12", "T13"]

# ===============================
# STEP 1: Collect average spectra per sample
# ===============================
all_avg_spectra = {}
wavenumbers = None

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith("_umap_kmeans.npz"):
        continue

    sample_name = fname.replace("_umap_kmeans.npz", "").replace("masked_cube_", "")
    file_path = os.path.join(input_dir, fname)

    with np.load(file_path) as f:
        spectra = f["spectra"]  # (n_pixels, n_bands)
        wn = f["wavenumbers"]
        avg_spec = np.mean(spectra, axis=0)
        all_avg_spectra[sample_name] = avg_spec
        if wavenumbers is None:
            wavenumbers = wn

print(f"✅ Collected average spectra for {len(all_avg_spectra)} samples")

# ===============================
# STEP 2: Group into WT and KO
# ===============================
wt_spectra = [all_avg_spectra[s] for s in wt_samples if s in all_avg_spectra]
ko_spectra = [all_avg_spectra[s] for s in ko_samples if s in all_avg_spectra]

wt_spectra = np.array(wt_spectra)
ko_spectra = np.array(ko_spectra)

print(f"WT samples: {wt_spectra.shape}, KO samples: {ko_spectra.shape}")

# ===============================
# STEP 3: Variance across groups
# ===============================
wt_var = np.var(wt_spectra, axis=0)
ko_var = np.var(ko_spectra, axis=0)

# ===============================
# STEP 4: Plot variance spectra
# ===============================
plt.figure(figsize=(10,6))
plt.plot(wavenumbers, wt_var, label="WT variance", color="blue")
plt.plot(wavenumbers, ko_var, label="KO variance", color="red")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Variance (a.u.)")
plt.title("Spectral Variance Across Samples – WT vs KO")
plt.legend()
plt.tight_layout()
plt.show()

# ===============================
# STEP 5: Summary bar plot
# ===============================
plt.figure(figsize=(6,5))
plt.bar(["WT", "KO"], [np.mean(wt_var), np.mean(ko_var)], color=["blue","red"])
plt.ylabel("Mean Variance (a.u.)")
plt.title("Average Spectral Variance – WT vs KO")
plt.tight_layout()
plt.show()

#%%

# Random similarity test


def build_similarity_matrices_custom(sample_paths, sample_names, groups, selected_clusters, 
                                     second_derivative=False, subsample_size=1000, random_state=40):
    rng = np.random.default_rng(random_state)
    spectra_all = {}
    wn_all = None

    for path, name in zip(sample_paths, sample_names):
        data = np.load(path)
        wn = data['wavenumbers']
        wn_all = wn
        spec = data['spectra']

        # ---- Compute 2nd derivative if requested ----
        if second_derivative:
            spec = np.gradient(np.gradient(spec, axis=1), axis=1)

        clusters = selected_clusters[name]
        mask = np.isin(data['cluster_labels'], clusters)
        spec_sel = spec[mask]

        # ---- Random subsampling ----
        if len(spec_sel) > subsample_size:
            idx = rng.choice(len(spec_sel), size=subsample_size, replace=False)
            spec_sel = spec_sel[idx]

        # ---- Select target bands ----
        spec_sel_bands = []
        for s in spec_sel:
            s_band, _ = select_bands(s, wn, targets, window)
            spec_sel_bands.append(s_band)
        spectra_all[name] = np.array(spec_sel_bands)

    # compute cosine similarity matrix
    n = len(sample_names)
    matrix = np.zeros((n, n))
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}

    for i, name1 in enumerate(sample_names):
        for j, name2 in enumerate(sample_names):
            s1 = spectra_all[name1]
            s2 = spectra_all[name2]

            # vectorized cosine similarity
            sim_matrix = cosine_similarity(s1, s2)
            sims_pair = sim_matrix.flatten().tolist()
            mean_sim = np.mean(sims_pair)
            matrix[i, j] = mean_sim

            # assign group sims
            g1 = 'WT' if name1 in groups['WT'] else 'KO'
            g2 = 'WT' if name2 in groups['WT'] else 'KO'
            if g1 == g2 == 'WT':
                sims['WT-WT'].extend(sims_pair)
            elif g1 == g2 == 'KO':
                sims['KO-KO'].extend(sims_pair)
            else:
                sims['WT-KO'].extend(sims_pair)

    return matrix, sample_names, sims

def add_significance(ax, x1, x2, y, pval, h=0.002):
    """Draws a significance bar with stars between two boxplots."""
    if pval < 0.001:
        stars = "***"
    elif pval < 0.01:
        stars = "**"
    elif pval < 0.05:
        stars = "*"
    elif pval < 0.1:
        stars = "*"   # treat p<0.1 as *
    else:
        stars = "ns"

    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.2, c="k")
    ax.text((x1+x2)*0.5, y+h, stars, ha="center", va="bottom", color="k", fontsize=12)

# ------------------------------
# Main plotting function
# ------------------------------
def plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- Heatmap Raw ---
    im1 = axs[0, 0].imshow(raw_matrix, cmap="Greens", vmin=.99, vmax=1)
    axs[0, 0].set_xticks(range(len(names)))
    axs[0, 0].set_yticks(range(len(names)))
    axs[0, 0].set_xticklabels(names, rotation=90)
    axs[0, 0].set_yticklabels(names)
    axs[0, 0].set_title("Cosine Similarity – Raw Spectra")
    fig.colorbar(im1, ax=axs[0, 0], fraction=0.046)

    # --- Heatmap 2nd Derivative ---
    im2 = axs[0, 1].imshow(sec_matrix, cmap="Greens", vmin=0.99, vmax=1)
    axs[0, 1].set_xticks(range(len(names)))
    axs[0, 1].set_yticks(range(len(names)))
    axs[0, 1].set_xticklabels(names, rotation=90)
    axs[0, 1].set_yticklabels(names)
    axs[0, 1].set_title("Cosine Similarity – 2nd Derivative Spectra")
    fig.colorbar(im2, ax=axs[0, 1], fraction=0.046)

    # ---------------------------
    # Boxplot with points (Raw)
    # ---------------------------
    data_raw = [raw_sims["WT-WT"], raw_sims["KO-KO"], raw_sims["WT-KO"]]
    axs[1, 0].boxplot(data_raw, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)

    for i, d in enumerate(data_raw, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 0].plot(x, d, "o", color="black", alpha=0.2, markersize=1)

    axs[1, 0].set_title("Group Similarities – Raw Spectra")
    axs[1, 0].set_ylabel("Cosine Similarity")
    axs[1, 0].set_ylim(0.0, 1.02)

    # significance tests raw
    pairs = [(0, 1), (0, 2), (1, 2)]
    y_max = max([max(d) for d in data_raw])
    for (i, j) in pairs:
        stat, p = ttest_ind(data_raw[i], data_raw[j], equal_var=False)
        add_significance(axs[1, 0], i+1, j+1, y_max + (pairs.index((i, j)) * 0.001), p)

    # ---------------------------
    # Boxplot with points (2nd Derivative)
    # ---------------------------
    data_sec = [sec_sims["WT-WT"], sec_sims["KO-KO"], sec_sims["WT-KO"]]
    axs[1, 1].boxplot(data_sec, labels=["WT-WT", "KO-KO", "WT-KO"], patch_artist=True)

    for i, d in enumerate(data_sec, start=1):
        x = np.random.normal(i, 0.05, size=len(d))
        axs[1, 1].plot(x, d, "o", color="black", alpha=0.2, markersize=1)

    axs[1, 1].set_title("Group Similarities – 2nd Derivative Spectra")
    axs[1, 1].set_ylabel("Cosine Similarity")
    axs[1, 1].set_ylim(0.70, 1.02)

    # significance tests sec
    y_max = max([max(d) for d in data_sec])
    for (i, j) in pairs:
        stat, p = ttest_ind(data_sec[i], data_sec[j], equal_var=False)
        add_significance(axs[1, 1], i+1, j+1, y_max + (pairs.index((i, j)) * 0.001), p)

    plt.tight_layout()
    plt.show()


# ===============================
# Example Usage
# ===============================
sample_names = ["T1","T2","T3","T6",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T6"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
}

selected_clusters = {
    "T1":[3],
    "T2": [3],
    "T3": [2],
   #"T4": [1,2,3,4,5],
    #"T5": [5],
    "T6": [1],
    #T7": [1,2,3,4,5],
   #"T8": [1,2,3,4,5],
   #T9": [1,2,3,4,5],
    "T10": [1],
    "T11": [2,4],
    "T12": [2,3],
    "T13": [0,4],
    "T14": [1,5],
    "T15": [5],
   #"T16": [1,2,3,4,5]
}

raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=False
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=True
)

plot_results(raw_matrix, sec_matrix, names, raw_sims, sec_sims)


#%%

# Balanced samples


# ------------------------------
# Helper functions
# ------------------------------
def balance_sims(sims_dict, random_state=50):
    """
    Subsample each similarity group to the same number of points for fair comparison.
    """
    rng = np.random.default_rng(random_state)
    min_len = min(len(v) for v in sims_dict.values())
    balanced_sims = {}
    for k, v in sims_dict.items():
        if len(v) > min_len:
            balanced_sims[k] = rng.choice(v, size=min_len, replace=False)
        else:
            balanced_sims[k] = np.array(v)
    return balanced_sims

# ------------------------------
# Build similarity matrices
# ------------------------------
def build_similarity_matrices_custom(sample_paths, sample_names, groups, selected_clusters, 
                                     second_derivative=False, subsample_size=600, random_state=50):
    rng = np.random.default_rng(random_state)
    spectra_all = {}

    for path, name in zip(sample_paths, sample_names):
        data = np.load(path)
        spec = data['spectra']

        if second_derivative:
            spec = np.gradient(np.gradient(spec, axis=1), axis=1)

        clusters = selected_clusters[name]
        mask = np.isin(data['cluster_labels'], clusters)
        spec_sel = spec[mask]

        if len(spec_sel) > subsample_size:
            idx = rng.choice(len(spec_sel), size=subsample_size, replace=False)
            spec_sel = spec_sel[idx]

        spectra_all[name] = spec_sel

    n = len(sample_names)
    matrix = np.zeros((n, n))
    sims = {"WT-WT": [], "KO-KO": [], "WT-KO": []}

    for i, name1 in enumerate(sample_names):
        for j, name2 in enumerate(sample_names):
            s1 = spectra_all[name1]
            s2 = spectra_all[name2]

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

# ------------------------------
# Scatter plotting
# ------------------------------
def plot_results_scatter(raw_matrix, sec_matrix, names, raw_sims, sec_sims):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10))


    # --- Scatter points Raw ---
    raw_sims_bal = balance_sims(raw_sims)
    colors = {"WT-WT": "green", "KO-KO": "red", "WT-KO": "blue"}
    for idx, (key, color) in enumerate(colors.items(), 1):
        sims = raw_sims_bal[key]
        x = np.random.normal(idx, 0.05, size=len(sims))
        axs[0].scatter(x, sims, color=color, alpha=0.3, s=2, label=key)
    axs[0].set_xticks([1, 2, 3])
    axs[0].set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
    axs[0].set_ylabel("Cosine Similarity")
    axs[0].set_title("Raw Spectra Similarities")
    axs[0].set_ylim(-0.2, 1.02)
    axs[0].legend()

    # --- Scatter points 2nd Derivative ---
    sec_sims_bal = balance_sims(sec_sims)
    for idx, (key, color) in enumerate(colors.items(), 1):
        sims = sec_sims_bal[key]
        x = np.random.normal(idx, 0.05, size=len(sims))
        axs[1].scatter(x, sims, color=color, alpha=0.3, s=2, label=key)
    axs[1].set_xticks([1, 2, 3])
    axs[1].set_xticklabels(["WT-WT", "KO-KO", "WT-KO"])
    axs[1].set_ylabel("Cosine Similarity")
    axs[1].set_title("2nd Derivative Similarities")
    axs[1].set_ylim(0.7, 1.02)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

# ------------------------------
# Example Usage
# ------------------------------
sample_names = ["T1","T2","T3","T6",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T6"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
}

#selected_clusters = {
#    "T1":[3],
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

selected_clusters = {
    "T1":[0,1,2,3,4,5],
    "T2":[0,1,2,3,4,5],
    "T3":[0,1,2,3,4,5],
    "T6":[0,1,2,3,4,5],
    "T10":[0,1,2,3,4,5],
    "T11":[0,1,2,3,4,5],
    "T12":[0,1,2,3,4,5],
    "T13":[0,1,2,3,4,5],
    "T14":[0,1,2,3,4,5],
    "T15":[0,1,2,3,4,5],
}



raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=False
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=True
)

plot_results_scatter(raw_matrix, sec_matrix, names, raw_sims, sec_sims)


#%%
# With Normalization
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# User-editable: set this
save_dir = os.path.join(input_dir, "UMAP_clustering_8Cluster")   # <<< change this to the folder with your T*_umap_kmeans.npz files
out_prefix = os.path.join(save_dir, "similarity_scatter_pub")
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
                                     second_derivative=False, subsample_size=100, random_state=40,
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
            spec = np.gradient(np.gradient(spec, axis=1), axis=1)

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

def plot_publication_scatter(raw_sims, sec_sims, fraction=0.3, random_state=40,
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
    ax1.set_title("Raw spectra similarities (normalized; 30% sampled)")
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
    ax2.set_title("2nd derivative similarities (normalized; 30% sampled)")
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
sample_names = ["T1","T2","T3","T6","T17",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_umap_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T1","T2","T3","T6","T17"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
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
    "T1":[0,2,3,5,6],
    "T2": [0,2,4,6,7],
    "T3": [1,2,3,5],
    "T6": [0,2,5,6,7],
    "T17": [1,3,4,6,7],
    "T10": [0,2,4,5,6],
    "T11": [1,4,5,7],
    "T12": [0,1,2,5,6,7],
    "T13": [1,2,4,5,6,7],
    "T14": [0,3,4,6,7],
    "T15": [0,2,3,5,6,7],
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
#    #"T4": [5,7],
#    "T5": [0,2,4,5,6,7],
#    "T6": [0,1,3,5,6,7],
#    "T17": [1,3,5,6,7],
#    #"T18": [0,1,2,4,7],
    
    #T7": [1,2,3,4,5],
    #"T8": [1,2,3,4,5],
    #T9": [1,2,3,4,5],
#    "T10": [0,1,3,4,5,7],
#    "T11": [0,1,3,4,5,6],
#    "T12": [0,1,2,4,5,7],
#    "T13": [0,1,2,4,6,7],
#    "T14": [0,1,3,4,5,6],
#    "T15": [0,1,3,4,5,6,7]
    
#}



# Build sims (normalize_spectra=True like your original "Normalized")
raw_matrix, names, raw_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=False, subsample_size=100, random_state=40,
    normalize_spectra=True
)
sec_matrix, _, sec_sims = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters,
    second_derivative=True, subsample_size=100, random_state=40,
    normalize_spectra=True
)

# Plot and save publication-ready figure
plot_publication_scatter(raw_sims, sec_sims, fraction=0.30, random_state=40,
                         figsize=(10,9), font_family='Arial', base_fontsize=11,
                         out_prefix=out_prefix)

    