# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:40:46 2025

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
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\Area_CA"
save_dir = os.path.join(input_dir, "UMAP_clustering_8Cluster")
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
n_neighbors = 12
min_dist = 0.1
n_clusters = 8

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
    reducer = umap.UMAP(n_components=2, random_state=0,
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