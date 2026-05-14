# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 23:15:04 2025

@author: hrazeghikondela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

# --- Settings ---
input_dir = r"E:\FILIZ lab\11_25_2024\PFC_KO_1_Results\output_DG_R_EX"
save_dir = os.path.join(input_dir, "PFC_KO_1_clustering_all_spectra")
os.makedirs(save_dir, exist_ok=True)

# Wavenumber axis
wavenumbers = np.linspace(950, 1800, 426)
print(f"Loaded wavenumber axis with {len(wavenumbers)} bands.")

# Grayscale visualization band index
grayscale_band_index = 353  # ≈1301 cm⁻¹

# Mapping: sample name → which cluster labels to combine (not used here but kept)
sample_cluster_map = {
    "T1": [1, 4],
    "T2": [2, 3],
    "T3": [3],
    "T4": [2, 3],
    "T5": [1, 2],
    "T6": [2, 3],
    "T7": [2, 4],
    "T8": [0, 4],
    "T9": [0, 5],
    "T10": [0, 3],
    "T11": [0, 2],
    "T12": [2, 4],
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
            data = npzfile["data"]
    except (IOError, zipfile.BadZipFile, KeyError) as e:
        print(f"❌ Skipping {fname}: {e}")
        continue

    print(f"\n✅ Processing {sample_name} – data shape: {data.shape}")
    H, W, B = data.shape
    flat_features, pixel_indices = [], []

    # --- Extract full spectra for each pixel ---
    for i in range(H):
        for j in range(W):
            spectrum = data[i, j, :]
            if np.isnan(spectrum).any():
                continue
            norm = np.max(np.abs(spectrum))
            if norm > 0:
                spectrum = spectrum / norm  # normalize spectrum
                flat_features.append(spectrum)
                pixel_indices.append((i, j))

    flat_features = np.array(flat_features)
    if len(flat_features) == 0:
        print(f"⚠ No valid pixels in {sample_name}")
        continue

    # --- PCA & clustering ---
    pca_result = PCA(n_components=3, random_state=10).fit_transform(flat_features)
    n_clusters = 8
    kmeans = KMeans(n_clusters=n_clusters, random_state=10).fit(pca_result)
    cluster_labels = kmeans.labels_

    # Build cluster map
    cluster_map = np.full((H, W), np.nan)
    for (i, j), label in zip(pixel_indices, cluster_labels):
        cluster_map[i, j] = label

    # Save full cluster map
    plt.figure(figsize=(8, 6))
    plt.imshow(cluster_map, cmap='tab10', interpolation='nearest')
    plt.title(f"{sample_name} – PCA + KMeans Clustering", fontsize=14)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{sample_name}_PCA_kmeans_map.tiff"),
                dpi=600, bbox_inches='tight')
    plt.close()

    # Save PCA scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1],
                c=cluster_labels, cmap='tab10', s=0.5, alpha=0.4)
    plt.xlabel("PCA 1", fontsize=12)
    plt.ylabel("PCA 2", fontsize=12)
    plt.title(f"{sample_name} – PCA Scatter", fontsize=14)
    plt.colorbar(label="Cluster Label")
    plt.savefig(os.path.join(save_dir, f"{sample_name}_PCA_scatter.tiff"),
                dpi=600, bbox_inches='tight')
    plt.close()

    # --- Overlay clusters on grayscale ---
    grayscale_img = data[:, :, grayscale_band_index]
    grayscale_img = np.nan_to_num(grayscale_img)
    grayscale_img = (grayscale_img - np.min(grayscale_img)) / (np.max(grayscale_img) - np.min(grayscale_img))

    for k in range(n_clusters):
        overlay = np.where(cluster_map == k, 1, 0)  # all cluster pixels → 1, background → 0
        masked_overlay = np.ma.masked_where(overlay == 0, overlay)
        plt.figure(figsize=(8, 6))
        plt.imshow(grayscale_img, cmap='gray', interpolation='nearest')
        plt.imshow(masked_overlay, cmap=ListedColormap(['gold']), alpha=0.5, interpolation='nearest')
        plt.title(f"{sample_name} – Cluster {k} Overlay (Gold)", fontsize=12)
        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"{sample_name}_cluster_{k}_overlay_gold.tiff"),
                    dpi=600, bbox_inches='tight')
        plt.close()
