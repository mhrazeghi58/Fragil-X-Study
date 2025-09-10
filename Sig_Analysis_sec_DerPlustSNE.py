# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:25:42 2025

@author: hrazeghikondela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from scipy.signal import savgol_filter
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap

# --- Settings ---
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_Whole_R"
save_dir = os.path.join(input_dir, "tSNE_clustering")
os.makedirs(save_dir, exist_ok=True)

# Wavenumber axis
wavenumbers = np.linspace(950, 1800, 426)
print(f"Loaded wavenumber axis with {len(wavenumbers)} bands.")

# Target peaks
targets = {
    "C=O stretch (ester, lipid)": 1740,
    "Amide I band": 1655,
    "Amide II band": 1545,
    "CH₂ scissoring": 1464,
    "CH₃ symmetric bending": 1375
}
window = 7  # ± cm⁻¹ range

# Derivative parameters
savgol_window = 11
polyorder = 3

# Grayscale visualization band index
grayscale_band_index = 353  # ≈1301.2 cm⁻¹

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
    deriv_data = np.full(data.shape, np.nan, dtype=np.float32)
    flat_features = []
    pixel_indices = []

    # Band indices for targets
    target_indices = [np.where((wavenumbers >= c - window) & (wavenumbers <= c + window))[0]
                      for c in targets.values()]

    # --- Compute second derivative & extract features ---
    for i in range(H):
        for j in range(W):
            spectrum = data[i, j, :]
            if np.isnan(spectrum).any():
                continue
            second_deriv = savgol_filter(spectrum, window_length=savgol_window, polyorder=polyorder, deriv=2)
            norm = np.max(np.abs(second_deriv))
            if norm > 0:
                deriv_data[i, j, :] = second_deriv
                selected_bands = np.concatenate([second_deriv[idx] for idx in target_indices])
                flat_features.append(selected_bands)
                pixel_indices.append((i, j))
            
    flat_features = np.array(flat_features)
    if len(flat_features) == 0:
        print(f"⚠ No valid pixels in {sample_name}")
        continue

    # --- t-SNE & Clustering ---
    #perplexity = min(30, max(5, len(flat_features) // 3))
    tsne_result = TSNE(n_components=2, perplexity=30, random_state=0).fit_transform(flat_features)
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(tsne_result)
    cluster_labels = kmeans.labels_

    # Build 2D cluster map
    cluster_map = np.full((H, W), np.nan)
    for (i, j), label in zip(pixel_indices, cluster_labels):
        cluster_map[i, j] = label

    # --- Save full cluster map ---
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cluster_map, cmap='tab10', interpolation='nearest')
    plt.title(f"{sample_name} – t-SNE + KMeans Clustering", fontsize=14)
    plt.colorbar(im, label="Cluster Label")
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f"{sample_name}_tSNE_kmeans_map.tiff"), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Save t-SNE scatter ---
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='tab10', s=0.5, alpha=0.4)
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.title(f"{sample_name} – t-SNE Scatter (Selected Bands)", fontsize=14)
    plt.colorbar(scatter, label="Cluster Label")
    plt.savefig(os.path.join(save_dir, f"{sample_name}_tSNE_scatter.tiff"), dpi=600, bbox_inches='tight')
    plt.close()

    # --- Prepare grayscale base image from band 353 ---
    grayscale_img = data[:, :, grayscale_band_index]
    grayscale_img = np.nan_to_num(grayscale_img)
    grayscale_img = (grayscale_img - np.min(grayscale_img)) / (np.max(grayscale_img) - np.min(grayscale_img))

    # --- Save cluster overlays on grayscale ---
    cmap = plt.get_cmap('tab10', n_clusters)
    for k in range(n_clusters):
        overlay = np.where(cluster_map == k, k + 1, 0)
        masked_overlay = np.ma.masked_where(overlay == 0, overlay)

        plt.figure(figsize=(8, 6))
        plt.imshow(grayscale_img, cmap='gray', interpolation='nearest')
        plt.imshow(masked_overlay, cmap=ListedColormap([cmap(k)]), alpha=0.5, interpolation='nearest')
        plt.title(f"{sample_name} – Cluster {k} Overlay (Band {grayscale_band_index})", fontsize=12)
        plt.axis('off')
        overlay_path = os.path.join(save_dir, f"{sample_name}_cluster_{k}_overlay_band_{grayscale_band_index}.tiff")
        plt.savefig(overlay_path, dpi=600, bbox_inches='tight')
        plt.close()
        
# Save t-SNE and clustering results
tsne_outfile = os.path.join(save_dir, f"{sample_name}_tsne_kmeans.npz")
np.savez_compressed(
    tsne_outfile,
    cluster_labels=cluster_labels,
    tsne_result=tsne_result,
    pixel_indices=np.array(pixel_indices),
    selected_band_features=flat_features
)
print(f"💾 Saved clustering results to {tsne_outfile}")

import os
import numpy as np
import matplotlib.pyplot as plt

# Settings
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_Whole_R\tSNE_clustering"
wavenumbers = np.linspace(950, 1800, 426)

clusters_of_interest = {
    'T1': [0, 1], 'T2': [2, 3], 'T3': [1, 4],
    'T4': [2, 5], 'T5': [3, 4], 'T6': [0, 5],
    'T7': [1, 2], 'T8': [0, 3], 'T9': [1, 5],
    'T10': [0, 2], 'T11': [1, 4], 'T12': [0, 5],
}

for fname in sorted(os.listdir(input_dir)):
    if not fname.endswith("_tsne_kmeans.npz"):
        continue

    sample_name = fname.split("_tsne_kmeans.npz")[0]
    if sample_name not in clusters_of_interest:
        print(f"⚠ Skipping {sample_name} (no selected clusters)")
        continue

    # Load cluster results
    npz = np.load(os.path.join(input_dir, fname))
    cluster_labels = npz["cluster_labels"]
    pixel_indices = npz["pixel_indices"]
    selected_features = npz["selected_band_features"]

    # Load original spectral data
    cube_file = os.path.join(input_dir.replace("tSNE_clustering", ""), f"masked_cube_{sample_name}.npz")
    with np.load(cube_file) as d:
        data = d["data"]

    # Compute second derivative
    from scipy.signal import savgol_filter
    deriv_data = np.full(data.shape, np.nan)
    for (i, j) in pixel_indices:
        spectrum = data[i, j, :]
        if not np.isnan(spectrum).any():
            deriv_data[i, j, :] = savgol_filter(spectrum, 11, 3, deriv=2)

    # Cluster selection and spectra extraction
    selected_clusters = clusters_of_interest[sample_name]
    cluster_spectra = {k: [] for k in selected_clusters}
    cluster_derivs = {k: [] for k in selected_clusters}

    for (ij, label) in zip(pixel_indices, cluster_labels):
        i, j = ij
        if label in selected_clusters:
            spectrum = data[i, j, :]
            deriv = deriv_data[i, j, :]
            if not np.isnan(spectrum).any():
                cluster_spectra[label].append(spectrum)
            if not np.isnan(deriv).any():
                cluster_derivs[label].append(deriv)

    # Plot raw spectra
    plt.figure(figsize=(10, 6))
    for k in selected_clusters:
        if cluster_spectra[k]:
            mean_spec = np.mean(cluster_spectra[k], axis=0)
            plt.plot(wavenumbers, mean_spec, label=f"Cluster {k}")
    plt.title(f"{sample_name} – Mean Spectra (Raw)")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorbance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, f"{sample_name}_cluster_raw_spectra.png"), dpi=300)
    plt.close()

    # Plot 2nd derivative spectra
    plt.figure(figsize=(10, 6))
    for k in selected_clusters:
        if cluster_derivs[k]:
            mean_deriv = np.mean(cluster_derivs[k], axis=0)
            plt.plot(wavenumbers, mean_deriv, label=f"Cluster {k}")
    plt.title(f"{sample_name} – Mean Spectra (2nd Derivative)")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("2nd Derivative")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(input_dir, f"{sample_name}_cluster_derivative_spectra.png"), dpi=300)
    plt.close()
