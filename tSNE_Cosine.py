# -*- coding: utf-8 -*-
"""
Created on Tue Sep 23 13:54:40 2025

@author: hrazeghikondela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import ListedColormap

# ====================================================
# SETTINGS
# ====================================================
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_L"
save_dir = os.path.join(input_dir, "tSNE_clustering")
os.makedirs(save_dir, exist_ok=True)

wavenumbers = np.linspace(950, 1800, 426)
grayscale_band_index = 353
savgol_window = 11
polyorder = 3
n_clusters = 6
tsne_perplexity = 30
tsne_learning_rate = 200

targets = {
    "C=O stretch (ester, lipid)": 1740,
    "Amide I band": 1655,
    "Amide II band": 1545,
    "CH2 scissoring": 1464,
    "CH3 symmetric bending": 1375
}
window = 16
mask = np.zeros_like(wavenumbers, dtype=bool)
for _, center in targets.items():
    mask |= (wavenumbers >= (center - window)) & (wavenumbers <= (center + window))
selected_wavenumbers = wavenumbers[mask]

# ====================================================
# HELPER FUNCTIONS
# ====================================================
def compute_second_derivative(spectrum, window_length=11, polyorder=3):
    return savgol_filter(spectrum, window_length, polyorder, deriv=2)

# ====================================================
# PART 1: t-SNE + KMeans clustering (updated)
# ====================================================
for fname in sorted(os.listdir(input_dir)):
    if not fname.startswith("masked_cube_T") or not fname.endswith(".npz"):
        continue

    sample_name = fname.replace("masked_cube_", "").replace(".npz", "")
    file_path = os.path.join(input_dir, fname)
    print(f"\n🔹 Processing sample: {sample_name}")

    with np.load(file_path) as npzfile:
        data = npzfile["data"]  # (H, W, Bands)
    H, W, B = data.shape
    print(f"   Data shape: {data.shape}")

    flat_features, pixel_indices = [], []
    for i in range(H):
        for j in range(W):
            spectrum = data[i, j, :][mask]
            if np.isnan(spectrum).any():
                continue
            second_deriv = compute_second_derivative(spectrum, savgol_window, polyorder)
            flat_features.append(second_deriv)
            pixel_indices.append((i, j))

    flat_features = np.array(flat_features)
    if flat_features.shape[0] == 0:
        print(f"⚠ No valid pixels in {sample_name}, skipping...")
        continue
    print(f"   Number of valid pixels: {flat_features.shape[0]}")

    # --- t-SNE + KMeans ---
    print("   Running t-SNE...")
    tsne_result = TSNE(n_components=2, perplexity=tsne_perplexity,
                       learning_rate=tsne_learning_rate, random_state=0).fit_transform(flat_features)
    print("   t-SNE done. Running KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(tsne_result)
    cluster_labels = kmeans.labels_
    print("   KMeans done.")

    # --- Build 2D cluster map ---
    cluster_map = np.full((H, W), np.nan)
    for (i, j), label in zip(pixel_indices, cluster_labels):
        cluster_map[i, j] = label

    # --- Plot and save cluster map ---
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cluster_map, cmap="tab10", interpolation="nearest")
    plt.title(f"{sample_name} – t-SNE + KMeans Clustering", fontsize=14)
    plt.colorbar(im, label="Cluster Label")
    plt.axis("off")
    plt.savefig(os.path.join(save_dir, f"{sample_name}_cluster_map.tiff"),
                dpi=600, bbox_inches="tight")
    plt.close()
    print("   Cluster map saved.")

    # --- Plot t-SNE scatter ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1],
                          c=cluster_labels, cmap="tab10", s=0.5, alpha=0.4)
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.title(f"{sample_name} – t-SNE Scatter", fontsize=14)
    plt.colorbar(scatter, label="Cluster Label")
    plt.savefig(os.path.join(save_dir, f"{sample_name}_tsne_scatter.tiff"),
                dpi=600, bbox_inches="tight")
    plt.close()
    print("   t-SNE scatter saved.")

    # --- Prepare grayscale base image ---
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
    print(f"   Cluster overlays saved for {sample_name}.")

    # --- Save results to npz ---
    np.savez_compressed(
        os.path.join(save_dir, f"{sample_name}_tsne_kmeans.npz"),
        cluster_labels=np.array(cluster_labels),
        tsne_result=tsne_result,
        pixel_indices=np.array(pixel_indices),
        spectra=np.array([data[i, j, :][mask] for (i, j) in pixel_indices]),
        wavenumbers=selected_wavenumbers
    )
    print(f"💾 Results saved for {sample_name}.")


# ====================================================
# PART 2: Cosine similarity
# ====================================================
sample_names = ["T2","T3","T4","T6",
                "T10","T11","T12","T13","T14","T15"]
sample_paths = [os.path.join(save_dir, f"{name}_tsne_kmeans.npz") for name in sample_names]

groups = {
    "WT": ["T2","T3","T4","T6"],
    "KO": ["T10","T11","T12","T13","T14","T15"]
}

selected_clusters = {
    "T2": [2],
    "T3": [1],
    "T4": [7],
    "T6": [4,6],
    "T10": [1],
    "T11": [4,7],
    "T12": [2],
    "T13": [3,5],
    "T14": [0,4],
    "T15": [3,7]
}

def compute_average_spectrum_per_sample(npz_path, clusters_to_use=None, second_derivative=False):
    with np.load(npz_path) as f:
        spectra = f["spectra"]
        labels = f["cluster_labels"]
        wn = f["wavenumbers"]

    if clusters_to_use is None:
        mask_use = np.ones_like(labels, dtype=bool)
    else:
        mask_use = np.isin(labels, clusters_to_use)

    selected_spectra = spectra[mask_use]
    if selected_spectra.shape[0] == 0:
        raise ValueError(f"No spectra for clusters {clusters_to_use} in {npz_path}")

    avg = np.mean(selected_spectra, axis=0)
    if second_derivative:
        avg = compute_second_derivative(avg)
    return wn, avg

def build_similarity_matrices_custom(sample_paths, sample_names, groups, selected_clusters, second_derivative=False):
    avg_spectra = {}
    for path, name in zip(sample_paths, sample_names):
        clusters = selected_clusters.get(name, None)
        wn, avg = compute_average_spectrum_per_sample(path, clusters, second_derivative)
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

    im1 = axs[0, 0].imshow(raw_matrix, cmap="viridis", vmin=0, vmax=1)
    axs[0, 0].set_xticks(range(len(names)))
    axs[0, 0].set_yticks(range(len(names)))
    axs[0, 0].set_xticklabels(names, rotation=90)
    axs[0, 0].set_yticklabels(names)
    axs[0, 0].set_title(f"Cosine Similarity – Raw Spectra {title_suffix}")
    fig.colorbar(im1, ax=axs[0, 0], fraction=0.046)

    im2 = axs[0, 1].imshow(sec_matrix, cmap="viridis", vmin=0, vmax=1)
    axs[0, 1].set_xticks(range(len(names)))
    axs[0, 1].set_yticks(range(len(names)))
    axs[0, 1].set_xticklabels(names, rotation=90)
    axs[0, 1].set_yticklabels(names)
    axs[0, 1].set_title(f"Cosine Similarity – 2nd Derivative {title_suffix}")
    fig.colorbar(im2, ax=axs[0, 1], fraction=0.046)

    axs[1, 0].boxplot([raw_sims["WT-WT"], raw_sims["KO-KO"], raw_sims["WT-KO"]],
                      labels=["WT-WT", "KO-KO", "WT-KO"])
    axs[1, 0].set_title(f"Group Similarities – Raw {title_suffix}")
    axs[1, 0].set_ylabel("Cosine Similarity")

    axs[1, 1].boxplot([sec_sims["WT-WT"], sec_sims["KO-KO"], sec_sims["WT-KO"]],
                      labels=["WT-WT", "KO-KO", "WT-KO"])
    axs[1, 1].set_title(f"Group Similarities – 2nd Derivative {title_suffix}")
    axs[1, 1].set_ylabel("Cosine Similarity")

    plt.tight_layout()
    plt.show()

# --- Cosine similarity for selected clusters ---
raw_matrix_sel, names_sel, raw_sims_sel = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=False
)
sec_matrix_sel, _, sec_sims_sel = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, selected_clusters, second_derivative=True
)
plot_results(raw_matrix_sel, sec_matrix_sel, names_sel, raw_sims_sel, sec_sims_sel,
             title_suffix="(selected clusters)")

# --- Cosine similarity for excluded clusters (everything else) ---
excluded_clusters = {k: [c for c in range(n_clusters) if c not in selected_clusters.get(k, [])] for k in sample_names}
raw_matrix_exc, names_exc, raw_sims_exc = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, excluded_clusters, second_derivative=False
)
sec_matrix_exc, _, sec_sims_exc = build_similarity_matrices_custom(
    sample_paths, sample_names, groups, excluded_clusters, second_derivative=True
)
plot_results(raw_matrix_exc, sec_matrix_exc, names_exc, raw_sims_exc, sec_sims_exc,
             title_suffix="(excluded clusters)")
