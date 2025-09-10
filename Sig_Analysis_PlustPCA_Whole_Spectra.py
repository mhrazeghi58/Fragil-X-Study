# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 14:03:58 2025

@author: hrazeghikondela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile
#from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
#from mpl_toolkits.mplot3d import Axes3D

# --- Settings ---
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_Whole_R"
save_dir = os.path.join(input_dir, "roi_amide_maps_original_R_clustering")
os.makedirs(save_dir, exist_ok=True)

# Wavenumber axis
wavenumbers = np.linspace(950, 1800, 426)
print(f"Loaded wavenumber axis with {len(wavenumbers)} bands.")

# Grayscale visualization band index (≈1301 cm⁻¹)
grayscale_band_index = 353

# Mapping: sample name → which cluster labels to combine
sample_cluster_map = {
    "T1": [1, 4], "T2": [2, 3], "T3": [3], "T4": [2, 3], "T5": [1, 2], "T6": [2, 3],
    "T7": [2, 4], "T8": [0, 4], "T9": [0, 5], "T10": [0, 3], "T11": [0, 2], "T12": [2, 4],
}

# Lists to store per-sample mean spectra and names
cluster_spectra_WT, cluster_spectra_KO = [], []
sample_names_WT, sample_names_KO = [], []

# --- Process each file ---
for fname in sorted(os.listdir(input_dir)):
    if not fname.startswith("masked_cube_T") or not fname.endswith(".npz"):
        continue

    sample_name = fname.replace("masked_cube_", "").replace(".npz", "")
    file_path = os.path.join(input_dir, fname)

    try:
        with np.load(file_path) as npzfile:
            if "data" not in npzfile:
                raise KeyError("Missing 'data' key.")
            data = npzfile["data"]  # shape: (H, W, Bands)
    except (IOError, zipfile.BadZipFile, KeyError) as e:
        print(f"❌ Skipping {fname}: {e}")
        continue

    print(f"\n✅ Processing {sample_name} – data shape: {data.shape}")
    H, W, B = data.shape
    flat_features, pixel_indices = [], []

    # --- Collect original spectra directly ---
    for i in range(H):
        for j in range(W):
            spectrum = data[i, j, :]
            if np.isnan(spectrum).any():
                continue
            norm = np.max(np.abs(spectrum))
            if norm > 0:
                flat_features.append(spectrum)
                pixel_indices.append((i, j))

    flat_features = np.array(flat_features)
    if len(flat_features) == 0:
        print(f"⚠ No valid pixels in {sample_name}")
        continue

    # --- PCA + KMeans clustering ---
    n_clusters = 6
    pca_result = PCA(n_components=6, random_state=20).fit_transform(flat_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=20).fit(pca_result)
    cluster_labels = kmeans.labels_

    # Build cluster map
    cluster_map = np.full((H, W), np.nan)
    for (i, j), label in zip(pixel_indices, cluster_labels):
        cluster_map[i, j] = label

    # --- Save cluster map ---
    plt.figure(figsize=(8, 6))
    plt.imshow(cluster_map, cmap='tab10', interpolation='nearest')
    plt.title(f"{sample_name} – PCA + KMeans Clustering", fontsize=14)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{sample_name}_PCA_kmeans_map.tiff"), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Save 3D PCA scatter plot ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(
        pca_result[:, 0], pca_result[:, 1], pca_result[:, 2],
        c=cluster_labels, cmap='tab10', s=0.2, alpha=0.4
    )
    ax.set_xlabel("PCA 1", fontsize=12)
    ax.set_ylabel("PCA 2", fontsize=12)
    ax.set_zlabel("PCA 3", fontsize=12)
    ax.set_title(f"{sample_name} – 3D PCA Scatter", fontsize=14)
    fig.colorbar(scatter, label="Cluster Label", shrink=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{sample_name}_PCA_3D_scatter.tiff"), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Overlay clusters on grayscale in gold ---
    grayscale_img = data[:, :, grayscale_band_index]
    grayscale_img = np.nan_to_num(grayscale_img)
    grayscale_img = (grayscale_img - np.min(grayscale_img)) / (np.max(grayscale_img) - np.min(grayscale_img))

    for k in range(n_clusters):
        overlay = np.where(cluster_map == k, 1, 0)
        masked_overlay = np.ma.masked_where(overlay == 0, overlay)
        plt.figure(figsize=(8, 6))
        plt.imshow(grayscale_img, cmap='gray', interpolation='nearest')
        plt.imshow(masked_overlay, cmap=ListedColormap(['gold']), alpha=0.5, interpolation='nearest')
        plt.title(f"{sample_name} – Cluster {k} Overlay (Gold)", fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{sample_name}_cluster_{k}_overlay_gold.tiff"), dpi=600, bbox_inches='tight')
        plt.close()

    # --- Collect per-sample mean spectra from chosen clusters ---
    selected_labels = sample_cluster_map.get(sample_name, [])
    selected_pixels = [
        flat_features[idx]
        for idx, label in enumerate(cluster_labels)
        if label in selected_labels
    ]
    if len(selected_pixels) > 0:
        mean_spectrum = np.nanmean(selected_pixels, axis=0)
        sample_type = "WT" if sample_name in ["T1", "T2", "T3", "T4", "T5", "T6"] else "KO"
        if sample_type == "WT":
            cluster_spectra_WT.append(mean_spectrum)
            sample_names_WT.append(sample_name)
        else:
            cluster_spectra_KO.append(mean_spectrum)
            sample_names_KO.append(sample_name)

# --- Plot per-sample mean spectra ---
plt.figure(figsize=(10, 6))
for idx, spectrum in enumerate(cluster_spectra_WT):
    plt.plot(wavenumbers, spectrum, label=f"{sample_names_WT[idx]} (WT)", linewidth=1)
for idx, spectrum in enumerate(cluster_spectra_KO):
    plt.plot(wavenumbers, spectrum, label=f"{sample_names_KO[idx]} (KO)", linewidth=1)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Intensity")
plt.title("Mean Original Spectra – Selected Clusters per Sample")
plt.legend()
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "All_samples_selected_clusters_original_spectra.tiff"), dpi=600)
plt.close()

# --- Plot global mean WT vs KO spectra with error ribbon ---
if cluster_spectra_WT and cluster_spectra_KO:
    mean_WT = np.nanmean(cluster_spectra_WT, axis=0)
    std_WT = np.nanstd(cluster_spectra_WT, axis=0)
    mean_KO = np.nanmean(cluster_spectra_KO, axis=0)
    std_KO = np.nanstd(cluster_spectra_KO, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(wavenumbers, mean_WT, label="WT (mean)", color='blue')
    plt.fill_between(wavenumbers, mean_WT - std_WT, mean_WT + std_WT, color='blue', alpha=0.2)
    plt.plot(wavenumbers, mean_KO, label="KO (mean)", color='red')
    plt.fill_between(wavenumbers, mean_KO - std_KO, mean_KO + std_KO, color='red', alpha=0.2)
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title("Mean Original Spectra ± SD – WT vs KO")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "WT_vs_KO_mean_original_spectra_with_ribbon.tiff"), dpi=600)
    plt.close()
    print("\n✅ Saved WT vs KO mean original spectra with error ribbon")

print("\n✅ All done!")
