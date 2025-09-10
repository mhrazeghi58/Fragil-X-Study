# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 12:44:33 2025

@author: hrazeghikondela
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm

# --- Settings ---
input_dir = r"D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)\output_DG_Whole_L"
save_dir = os.path.join(input_dir, "roi_amide_maps_2nd_deriv")
os.makedirs(save_dir, exist_ok=True)

# Wavenumber axis
wavenumbers = np.linspace(950, 1800, 426)
print(f"Loaded wavenumber axis with {len(wavenumbers)} bands.")

# Target bands and visualization parameters
targets = {
    "C=O stretch (ester, lipid)": 1740,
    "Amide I band": 1655,
    "Amide II band": 1545,
    "CH₂ scissoring": 1464,
    "CH₃ symmetric bending": 1375
}
window = 7  # cm⁻¹ ± window

# Custom visualization range per target
visual_params = {
    "C=O stretch (ester, lipid)": {
        "cmap": "turbo", "vmin": -0.6, "vmax": 0.0
    },
    "Amide I band": {
        "cmap": "turbo", "vmin": -0.8, "vmax": -0.0
    },
    "Amide II band": {
        "cmap": "turbo", "vmin": -0.6, "vmax": 0.0
    },
    "CH₂ scissoring": {
        "cmap": "turbo", "vmin": -0.6, "vmax": 0.0
    },
    "CH₃ symmetric bending": {
        "cmap": "turbo", "vmin": -0.6, "vmax": 0.0
    }
}


# Savitzky-Golay filter parameters
savgol_window = 11  # must be odd
polyorder = 3

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

    # Compute second derivative spectrum per pixel
    for i in range(H):
        for j in range(W):
            spectrum = data[i, j, :]
            if np.isnan(spectrum).any():
                continue
            # Apply Savitzky-Golay second derivative
            second_deriv = savgol_filter(spectrum, window_length=savgol_window, polyorder=polyorder, deriv=2)
            # Normalize by max absolute value
            norm = np.max(np.abs(second_deriv))
            if norm > 0:
                deriv_data[i, j, :] = second_deriv # / norm

    # Generate and save heatmaps
    for label, center in targets.items():
        idx_range = np.where((wavenumbers >= center - window) & (wavenumbers <= center + window))[0]
        if len(idx_range) == 0:
            print(f"⚠ No indices found for {label} at {center} ± {window}")
            continue

        # Use sum instead of mean
        slice_data = deriv_data[:, :, idx_range]
        valid_mask = ~np.isnan(slice_data)
        has_valid = np.any(valid_mask, axis=2)
        
        sum_map = np.where(has_valid, np.nansum(slice_data, axis=2), np.nan)

        # Normalize by max absolute value (preserve sign)
        max_abs = np.nanmax(np.abs(sum_map))
        if max_abs > 0:
         sum_map = sum_map / max_abs

       # Apply Gaussian smoothing
        smoothed_map = gaussian_filter(sum_map, sigma=3)
        masked_map = smoothed_map #np.ma.masked_where(np.isnan(smoothed_map) | (np.abs(smoothed_map) <= 0.4), smoothed_map)
        # Get visualization params and set colormap with white NaN
        params = visual_params[label]
        base_cmap = cm.get_cmap(params["cmap"]).copy()
        base_cmap.set_bad(color='white')

        # Plot
        plt.figure(figsize=(8, 6))
        im = plt.imshow(masked_map, cmap=base_cmap, interpolation='nearest',
                        vmin=params["vmin"], vmax=params["vmax"])
        plt.title(f"{sample_name} – {label} (2nd Derivative Sum, Smoothed)", fontsize=14)
        plt.colorbar(im, label=f"Sum 2nd Deriv. Intensity ({center} ± {window} cm⁻¹)")
        plt.axis('off')

        # Save
        out_path = os.path.join(save_dir, f"{sample_name}_{label}_2nd_sum_smoothed.tiff")
        plt.savefig(out_path, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()
        print(f"💾 Saved: {out_path}")
