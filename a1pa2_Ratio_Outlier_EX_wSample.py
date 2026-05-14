# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:02:19 2026

@author: mhraz
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------- FILE READING ---------------- #
def read_pixel_data(file_path):
    """Read 2D pixel data from ASC and flatten."""
    try:
        data = np.loadtxt(file_path, skiprows=6)
        return data.flatten()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def find_files_with_pattern(group_path, pattern):
    """Return list of ASC files in S_* folders, sorted numerically."""
    matching_files = []
    subfolders = [os.path.join(group_path, d) for d in os.listdir(group_path)
                  if os.path.isdir(os.path.join(group_path, d)) and d.startswith("S_")]

    def extract_index(name):
        try:
            return int(name.split("_")[1])
        except:
            return 9999

    subfolders.sort(key=lambda x: extract_index(os.path.basename(x)))

    for folder in subfolders:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if os.path.isfile(full) and f.endswith(pattern):
                matching_files.append(full)
    return matching_files

# ---------------- OUTLIER REMOVAL ---------------- #
def exclude_outliers_joint(a1, a2, tm=None, lower=2.5, upper=97.5):
    """Remove pixel-level outliers across a1, a2, tm jointly."""
    if a1.size == 0 or a2.size == 0:
        return a1, a2, tm
    lo_a1, hi_a1 = np.percentile(a1, [lower, upper])
    lo_a2, hi_a2 = np.percentile(a2, [lower, upper])
    mask = (a1 >= lo_a1) & (a1 <= hi_a1) & (a2 >= lo_a2) & (a2 <= hi_a2)
    if tm is not None:
        lo_tm, hi_tm = np.percentile(tm, [lower, upper])
        mask = mask & (tm >= lo_tm) & (tm <= hi_tm)
        return a1[mask], a2[mask], tm[mask]
    return a1[mask], a2[mask], tm

# ---------------- CLIFF'S DELTA ---------------- #
def cliffs_delta(x, y):
    """Compute Cliff's Delta."""
    x, y = np.array(x), np.array(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, "undefined"
    greater = np.sum(x[:, None] > y)
    smaller = np.sum(x[:, None] < y)
    delta = (greater - smaller) / (nx * ny)
    abs_d = abs(delta)
    if abs_d < 0.147: mag = "negligible"
    elif abs_d < 0.33: mag = "small"
    elif abs_d < 0.474: mag = "medium"
    else: mag = "large"
    return delta, mag

# ---------------- LOAD GROUP DATA ---------------- #
def load_group_data(a1_files, a2_files, tm_files=None):
    all_a1, all_a2, all_tm = [], [], []
    per_mouse_ratio = []

    for f1, f2, ftm in zip(a1_files, a2_files, tm_files if tm_files else [None]*len(a1_files)):
        a1 = read_pixel_data(f1)
        a2 = read_pixel_data(f2)
        tm = read_pixel_data(ftm) if ftm else None
        if a1 is None or a2 is None:
            continue
        valid = ~np.isnan(a1) & ~np.isnan(a2)
        if tm is not None: 
            valid = valid & ~np.isnan(tm)
        a1, a2 = a1[valid], a2[valid]
        tm = tm[valid] if tm is not None else None
        ratio = np.divide(a1, a2, out=np.zeros_like(a1), where=a2!=0)
        all_a1.append(a1)
        all_a2.append(a2)
        if tm is not None: 
            all_tm.append(tm)
        per_mouse_ratio.append(np.mean(ratio))

    if all_a1:
        return (np.concatenate(all_a1),
                np.concatenate(all_a2),
                np.concatenate(all_tm) if all_tm else None,
                np.array(per_mouse_ratio))
    else:
        return np.array([]), np.array([]), None, np.array([])

# ---------------- PLOTTING ---------------- #
def plot_all_samples(group_data, title_prefix):
    """Create 3-column plots per group showing a1/a2 vs tm."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (a1, a2, tm) in enumerate(group_data):
        ratio = np.divide(a1, a2, out=np.zeros_like(a1), where=a2 != 0)
        axs[i].scatter(ratio, tm, c=tm, cmap="viridis", s=5, alpha=0.6)
        axs[i].set_xlabel("a1/a2")
        axs[i].set_ylabel("tm (ns)")
        axs[i].set_ylim(0, 10)
        axs[i].set_title(f"{title_prefix} Sample {i+1}")
    
    plt.tight_layout()
    return fig

# ---------------- MAIN ---------------- #
def main():
    root = r"C:\Users\hrazeghikondela\Desktop\Paper\FLIM\Analysis"
    groups = {'WT': os.path.join(root, "WT"),
              'KO': os.path.join(root, "KO")}

    # Each group has Sample1, Sample2, Sample3
    group_sample_data = {}

    for group, path in groups.items():
        samples = [os.path.join(path, d) for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d)) and d.lower().startswith("sample")]
        samples.sort()
        group_data = []

        for sample_path in samples:
            a1 = find_files_with_pattern(sample_path, "_740_a1.asc")
            a2 = find_files_with_pattern(sample_path, "_740_a2.asc")
            tm = find_files_with_pattern(sample_path, "_740_color coded value.asc")

            a1_data, a2_data, tm_data, _ = load_group_data(a1, a2, tm)
            a1_data, a2_data, tm_data = exclude_outliers_joint(a1_data, a2_data, tm_data)
            group_data.append((a1_data, a2_data, tm_data))

        group_sample_data[group] = group_data

    # ---- Plot all ---- #
    fig_wt = plot_all_samples(group_sample_data['WT'], "WT")
    fig_ko = plot_all_samples(group_sample_data['KO'], "KO")
    plt.show()

if __name__ == "__main__":
    main()
