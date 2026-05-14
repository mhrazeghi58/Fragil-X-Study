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
        if tm is not None: valid = valid & ~np.isnan(tm)
        a1, a2 = a1[valid], a2[valid]
        tm = tm[valid] if tm is not None else None
        ratio = np.divide(a1, a2, out=np.zeros_like(a1), where=a2!=0)
        all_a1.append(a1)
        all_a2.append(a2)
        if tm is not None: all_tm.append(tm)
        per_mouse_ratio.append(np.mean(ratio))

    if all_a1:
        return (np.concatenate(all_a1),
                np.concatenate(all_a2),
                np.concatenate(all_tm) if all_tm else None,
                np.array(per_mouse_ratio))
    else:
        return np.array([]), np.array([]), None, np.array([])

# ---------------- PLOTTING ---------------- #
def plot_metrics(a1_wt, a2_wt, tm_wt,
                 a1_ko, a2_ko, tm_ko,
                 delta_dict):
    """Scatter plots of a1/a2, a1, a2 vs tm colored by tm."""
    fig, axs = plt.subplots(2, 2, figsize=(16,14))

    # ---- a1/a2 ratio ---- #
    ratio_wt = np.divide(a1_wt, a2_wt, out=np.zeros_like(a1_wt), where=a2_wt!=0)
    ratio_ko = np.divide(a1_ko, a2_ko, out=np.zeros_like(a1_ko), where=a2_ko!=0)

    axs[0,0].scatter(ratio_wt, tm_wt, c=tm_wt, cmap='viridis', s=5, alpha=0.6, label='WT')
    axs[0,0].scatter(ratio_ko, tm_ko, c=tm_ko, cmap='viridis', s=5, alpha=0.6, label='KO')
    axs[0,0].set_xlabel('a1/a2')
    axs[0,0].set_ylabel('tm (ns)')
    axs[0,0].set_ylim(0,10)
    axs[0,0].set_title(f"a1/a2 vs tm\nCliff's δ={delta_dict['ratio'][0]:.3f} ({delta_dict['ratio'][1]})")
    axs[0,0].legend()

    # ---- a1 ---- #
    axs[0,1].scatter(a1_wt, tm_wt, c=tm_wt, cmap='plasma', s=5, alpha=0.6, label='WT')
    axs[0,1].scatter(a1_ko, tm_ko, c=tm_ko, cmap='plasma', s=5, alpha=0.6, label='KO')
    axs[0,1].set_xlabel('a1')
    axs[0,1].set_ylabel('tm (ns)')
    axs[0,1].set_ylim(0,10)
    axs[0,1].set_title(f"a1 vs tm\nCliff's δ={delta_dict['a1'][0]:.3f} ({delta_dict['a1'][1]})")
    axs[0,1].legend()

    # ---- a2 ---- #
    axs[1,0].scatter(a2_wt, tm_wt, c=tm_wt, cmap='cool', s=5, alpha=0.6, label='WT')
    axs[1,0].scatter(a2_ko, tm_ko, c=tm_ko, cmap='cool', s=5, alpha=0.6, label='KO')
    axs[1,0].set_xlabel('a2')
    axs[1,0].set_ylabel('tm (ns)')
    axs[1,0].set_ylim(0,10)
    axs[1,0].set_title(f"a2 vs tm\nCliff's δ={delta_dict['a2'][0]:.3f} ({delta_dict['a2'][1]})")
    axs[1,0].legend()

    plt.tight_layout()
    plt.show()

# ---------------- MAIN ---------------- #
def main():
    root = r"C:\Users\hrazeghikondela\Desktop\Paper\FLIM\Analysis_test"
    wt_path = os.path.join(root, "WT")
    ko_path = os.path.join(root, "KO")

    # Find files
    wt_a1 = find_files_with_pattern(wt_path, "_740_a1.asc")
    wt_a2 = find_files_with_pattern(wt_path, "_740_a2.asc")
    wt_tm = find_files_with_pattern(wt_path, "_740_color coded value.asc")

    ko_a1 = find_files_with_pattern(ko_path, "_740_a1.asc")
    ko_a2 = find_files_with_pattern(ko_path, "_740_a2.asc")
    ko_tm = find_files_with_pattern(ko_path, "_740_color coded value.asc")

    # Load data
    a1_wt, a2_wt, tm_wt, _ = load_group_data(wt_a1, wt_a2, wt_tm)
    a1_ko, a2_ko, tm_ko, _ = load_group_data(ko_a1, ko_a2, ko_tm)

    # Remove outliers jointly
    a1_wt, a2_wt, tm_wt = exclude_outliers_joint(a1_wt, a2_wt, tm_wt)
    a1_ko, a2_ko, tm_ko = exclude_outliers_joint(a1_ko, a2_ko, tm_ko)

    # Compute Cliff's Delta for all metrics
    delta_dict = {
        'ratio': cliffs_delta(np.divide(a1_wt, a2_wt, out=np.zeros_like(a1_wt), where=a2_wt!=0),
                              np.divide(a1_ko, a2_ko, out=np.zeros_like(a1_ko), where=a2_ko!=0)),
        'a1': cliffs_delta(a1_wt, a1_ko),
        'a2': cliffs_delta(a2_wt, a2_ko)
    }

    # Plot all metrics
    plot_metrics(a1_wt, a2_wt, tm_wt,
                 a1_ko, a2_ko, tm_ko,
                 delta_dict)

if __name__ == "__main__":
    main()
