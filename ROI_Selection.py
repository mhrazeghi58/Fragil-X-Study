import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import gc  # for memory cleanup

# Setup
os.chdir(r'D:\Filiz Lab\Data_2024\Brain_Samples\Working_Copy\New folder (2)')
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# Sample info
sample_files = {
    "T1": "PFC_WT_1.npy",
    "T2": "PFC_WT_2.npy",
    "T3": "PFC_WT_3.npy",
    "T4": "PFC_WT_4.npy",
    "T5": "PFC_WT_5.npy",
    "T6": "PFC_WT_6.npy",
    "T7": "PFC_KO_1.npy",
    "T8": "PFC_KO_2.npy",
    "T9": "PFC_KO_3.npy",
    "T10": "PFC_KO_4.npy",
    "T11": "PFC_KO_5.npy",
    "T12": "PFC_KO_6.npy",
}

low_threshold = 0.0
slice_idx = 353  # Slice to visualize

# Store processed results
masked_cubes = {}
reshaped_data = {}

# Helper class for ROI
class ROISelector:
    def __init__(self, ax):
        self.ax = ax
        self.roi_coords = None
        self.selector = PolygonSelector(ax, self.onselect, useblit=True)
        self.cid_key_press = plt.connect('key_press_event', self.on_key_press)

    def onselect(self, verts):
        self.roi_coords = np.array(verts)

    def on_key_press(self, event):
        if event.key == "enter":
            plt.disconnect(self.cid_key_press)
            plt.close()

    def get_roi(self):
        plt.show(block=True)
        return self.roi_coords

def create_roi_mask(image, roi_coords):
    ny, nx = image.shape
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    points = np.vstack((x.flatten(), y.flatten())).T
    roi_path = Path(roi_coords)
    return roi_path.contains_points(points).reshape((ny, nx))

# Main loop: Load, process, unload
for sample_name, filename in sample_files.items():
    print(f"\nLoading and processing {sample_name} from {filename}")

    # Load
    T = np.load(filename)

    # Mask based on intensity threshold
    exc = np.percentile(T[:, :, slice_idx], 99.999)
    mask2d = ma.masked_where(
        np.logical_or(T[:, :, slice_idx] <= low_threshold, T[:, :, slice_idx] >= exc),
        T[:, :, slice_idx]
    )
    full_mask = np.zeros(T.shape, dtype=bool)
    full_mask[:, :, :] = mask2d[:, :, np.newaxis].mask
    masked_cube = ma.array(T, mask=full_mask)

    # Visualize and select ROI
    slice_image = masked_cube[:, :, slice_idx].filled(np.nan)
    fig, ax = plt.subplots(figsize=(26, 19))
    ax.imshow(slice_image, cmap='jet')
    ax.set_title(f"{sample_name}: Click to define ROI, press Enter to finish")
    roi_selector = ROISelector(ax)
    roi_coords = roi_selector.get_roi()

    if roi_coords is not None and roi_coords.size > 0:
        roi_mask = create_roi_mask(slice_image, roi_coords)

        # Apply 2D ROI mask to the whole cube
        roi_masked_cube = np.copy(masked_cube.filled(np.nan))
        for i in range(roi_masked_cube.shape[2]):
            roi_masked_cube[:, :, i][~roi_mask] = np.nan

        # Save processed data as compressed .npz file
        output_path = os.path.join(output_dir, f"masked_cube_{sample_name}.npz")
        np.savez_compressed(output_path, data=roi_masked_cube)

        # Store in dictionaries
        masked_cubes[sample_name] = roi_masked_cube
        reshaped_data[f"{sample_name}_m2d"] = roi_masked_cube[~np.isnan(roi_masked_cube)].reshape(-1, roi_masked_cube.shape[-1])

        print(f"Saved and stored {sample_name}")
    else:
        print(f"No ROI selected for {sample_name}, skipping.")

    # Unload raw data immediately
    del T, masked_cube, roi_masked_cube
    gc.collect()

print("\nProcessing complete!")


#%% Reload process
# List of sample names
sample_names = [f"T{i}" for i in range(1, 13)]  # T1 to T12

# Initialize the reshaped_data dictionary
reshaped_data = {}

# Load each .npz file and populate reshaped_data
for sample_name in sample_names:
    file_path = os.path.join(output_dir, f"masked_cube_{sample_name}.npz")
    if os.path.exists(file_path):
        with np.load(file_path) as data:
            roi_masked_cube = data['data']
            reshaped_data[f"{sample_name}_m2d"] = roi_masked_cube[~np.isnan(roi_masked_cube)].reshape(-1, roi_masked_cube.shape[-1])
    else:
        print(f"File {file_path} not found. Skipping.")


# %% List of sample names
sample_names = [f"T{i}" for i in range(1, 13)]  # T1 to T12

# Initialize the reshaped_data dictionary
reshaped_data = {}

# Load each .npz file and populate reshaped_data
# Initialize lists
X_list = []
y_list = []

# Assume: sample_names = list of sample names in the correct order (first 6 WT, last 6 KO)
for idx, sample_name in enumerate(sample_names):
    data_key = f"{sample_name}_m2d"
    if data_key in reshaped_data:
        sample_data = reshaped_data[data_key]  # shape (num_pixels, num_features)
        X_list.append(sample_data)
        
        # Label: 0 if index < 6 else 1
        label = 0 if idx < 6 else 1
        y_list.append(np.full(sample_data.shape[0], label))
    else:
        print(f"Sample {sample_name} not found in reshaped_data. Skipping.")

# Stack all data points
X = np.vstack(X_list)  # shape (total_pixels, num_features)
y = np.concatenate(y_list)  # shape (total_pixels,)



# %%
# Wavenumber range
wn = np.linspace(950, 1800, 426)

# Extract mean and std from reshaped_data (already masked and flattened)
WT_mean = np.mean(reshaped_data["T1_m2d"], axis=0)
WT_std = np.std(reshaped_data["T1_m2d"], axis=0)

WT1_mean = np.mean(reshaped_data["T2_m2d"], axis=0)
WT1_std = np.std(reshaped_data["T2_m2d"], axis=0)

WT2_mean = np.mean(reshaped_data["T3_m2d"], axis=0)
WT2_std = np.std(reshaped_data["T3_m2d"], axis=0)

WT3_mean = np.mean(reshaped_data["T4_m2d"], axis=0)
WT3_std = np.std(reshaped_data["T4_m2d"], axis=0)

WT4_mean = np.mean(reshaped_data["T5_m2d"], axis=0)
WT4_std = np.std(reshaped_data["T5_m2d"], axis=0)

WT5_mean = np.mean(reshaped_data["T6_m2d"], axis=0)
WT5_std = np.std(reshaped_data["T6_m2d"], axis=0)

KO_mean = np.mean(reshaped_data["T7_m2d"], axis=0)
KO_std = np.std(reshaped_data["T7_m2d"], axis=0)

KO1_mean = np.mean(reshaped_data["T8_m2d"], axis=0)
KO1_std = np.std(reshaped_data["T8_m2d"], axis=0)

KO2_mean = np.mean(reshaped_data["T9_m2d"], axis=0)
KO2_std = np.std(reshaped_data["T9_m2d"], axis=0)

KO3_mean = np.mean(reshaped_data["T10_m2d"], axis=0)
KO3_std = np.std(reshaped_data["T10_m2d"], axis=0)

KO4_mean = np.mean(reshaped_data["T11_m2d"], axis=0)
KO4_std = np.std(reshaped_data["T11_m2d"], axis=0)

KO5_mean = np.mean(reshaped_data["T12_m2d"], axis=0)
KO5_std = np.std(reshaped_data["T12_m2d"], axis=0)

# %%
plt.figure(figsize=(14, 9))

for idx, (sample_name, data) in enumerate(reshaped_data.items()):
    # Determine WT or KO by index (first 6 = WT, last 6 = KO)
    if idx < 6:
        color = 'orange'
    else:
        color = 'green'
    
    # Plot each pixel's spectrum
    plt.plot(wn, data.T, color=color, alpha=0.1)

# Labels and settings
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
plt.ylabel('Absorbance (a.u.)', fontsize=14)
plt.title('All Spectra from All Pixels (WT and KO)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.xlim([950, 1800])
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 9))

#%%
# Plot WT samples (orange)
plt.plot(wn, WT_mean, color='orange', linewidth=2, label="WT_1")
plt.fill_between(wn, WT_mean - WT_std, WT_mean + WT_std, color='orange', alpha=0.2)

plt.plot(wn, WT1_mean, color='orange', linewidth=2, label="WT_2")
plt.fill_between(wn, WT1_mean - WT1_std, WT1_mean + WT1_std, color='orange', alpha=0.2)

plt.plot(wn, WT2_mean, color='orange', linewidth=2, label="WT_3")
plt.fill_between(wn, WT2_mean - WT2_std, WT2_mean + WT2_std, color='orange', alpha=0.2)

plt.plot(wn, WT3_mean, color='orange', linewidth=2, label="WT_4")
plt.fill_between(wn, WT3_mean - WT3_std, WT3_mean + WT3_std, color='orange', alpha=0.2)

#plt.plot(wn, WT4_mean, color='orange', linewidth=2, label="WT_5")
#plt.fill_between(wn, WT4_mean - WT4_std, WT4_mean + WT4_std, color='orange', alpha=0.2)

plt.plot(wn, WT5_mean, color='orange', linewidth=2, label="WT_6")
plt.fill_between(wn, WT5_mean - WT5_std, WT5_mean + WT5_std, color='orange', alpha=0.2)

# Plot KO samples (green)
plt.plot(wn, KO_mean, color='green', linewidth=2, label="KO_1")
plt.fill_between(wn, KO_mean - KO_std, KO_mean + KO_std, color='green', alpha=0.2)

plt.plot(wn, KO1_mean, color='green', linewidth=2, label="KO_2")
plt.fill_between(wn, KO1_mean - KO1_std, KO1_mean + KO1_std, color='green', alpha=0.2)

plt.plot(wn, KO2_mean, color='green', linewidth=2, label="KO_3")
plt.fill_between(wn, KO2_mean - KO2_std, KO2_mean + KO2_std, color='green', alpha=0.2)

plt.plot(wn, KO3_mean, color='green', linewidth=2, label="KO_4")
plt.fill_between(wn, KO3_mean - KO3_std, KO3_mean + KO3_std, color='green', alpha=0.2)

plt.plot(wn, KO4_mean, color='green', linewidth=2, label="KO_5")
plt.fill_between(wn, KO4_mean - KO4_std, KO4_mean + KO4_std, color='green', alpha=0.2)

plt.plot(wn, KO5_mean, color='green', linewidth=2, label="KO_6")
plt.fill_between(wn, KO5_mean - KO5_std, KO5_mean + KO5_std, color='green', alpha=0.2)

# Labels and settings
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
plt.ylabel('Absorbance', fontsize=14)
plt.title('Spectral Data (Full ROI, With Shaded Variance)', fontsize=16)
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([950, 1800])
plt.tight_layout()
plt.show()


# Stack all spectra
WT_all = np.vstack([WT_mean, WT1_mean, WT2_mean, WT3_mean, WT4_mean, WT5_mean])
KO_all = np.vstack([KO_mean, KO1_mean, KO2_mean, KO3_mean, KO4_mean, KO5_mean])

# Mean and std
WT_avg = np.mean(WT_all, axis=0)
WT_std_all = np.std(WT_all, axis=0)

KO_avg = np.mean(KO_all, axis=0)
KO_std_all = np.std(KO_all, axis=0)

# Align starting Y (first wavenumber point)
start_diff = WT_avg[0] - KO_avg[0]
KO_avg_aligned = KO_avg + start_diff  # shift KO to match WT at the first point
KO_std_all_aligned = KO_std_all       # std stays same since it's symmetric

# Optional: Shift KO up/down slightly for better separation
vertical_shift = 0.0  # you can adjust this value
KO_avg_shifted = KO_avg_aligned + vertical_shift

# Plot
plt.figure(figsize=(16, 12))

# WT (orange)
plt.plot(wn, WT_avg, color='orange', linewidth=3, label="WT (mean)")
plt.fill_between(wn, WT_avg - WT_std_all, WT_avg + WT_std_all, color='orange', alpha=0.15)

# KO (green) shifted
plt.plot(wn, KO_avg_shifted, color='green', linewidth=3, label="KO (mean, shifted)")
plt.fill_between(wn, KO_avg_shifted - KO_std_all_aligned, KO_avg_shifted + KO_std_all_aligned, color='green', alpha=0.15)

# Labels
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=22)
plt.ylabel('Absorbance (a.u.)', fontsize=22)
plt.title('Averaged Spectra (Aligned and Shifted)', fontsize=16)
plt.legend(loc="upper left", fontsize=20)
#plt.grid(False, alpha=0.3)
plt.xlim([950, 1800])
plt.tight_layout()
plt.show()



 # %%
# --- Calculate first derivatives ---
WT_first = np.gradient(WT_mean, wn)
WT1_first = np.gradient(WT1_mean, wn)
WT2_first = np.gradient(WT2_mean, wn)
WT3_first = np.gradient(WT3_mean, wn)
WT4_first = np.gradient(WT4_mean, wn)
WT5_first = np.gradient(WT5_mean, wn)

KO_first = np.gradient(KO_mean, wn)
KO1_first = np.gradient(KO1_mean, wn)
KO2_first = np.gradient(KO2_mean, wn)
KO3_first = np.gradient(KO3_mean, wn)
KO4_first = np.gradient(KO4_mean, wn)
KO5_first = np.gradient(KO5_mean, wn)

# --- Calculate second derivatives ---
WT_second = np.gradient(WT_first, wn)
WT1_second = np.gradient(WT1_first, wn)
WT2_second = np.gradient(WT2_first, wn)
WT3_second = np.gradient(WT3_first, wn)
WT4_second = np.gradient(WT4_first, wn)
WT5_second = np.gradient(WT5_first, wn)

KO_second = np.gradient(KO_first, wn)
KO1_second = np.gradient(KO1_first, wn)
KO2_second = np.gradient(KO2_first, wn)
KO3_second = np.gradient(KO3_first, wn)
KO4_second = np.gradient(KO4_first, wn)
KO5_second = np.gradient(KO5_first, wn)

from scipy.signal import savgol_filter

# Smoothing parameters
window_length = 15  # must be odd
polyorder = 3       # usually 2 or 3

# Smooth first derivatives
WT_first_smooth = savgol_filter(WT_first, window_length, polyorder)
WT1_first_smooth = savgol_filter(WT1_first, window_length, polyorder)
WT2_first_smooth = savgol_filter(WT2_first, window_length, polyorder)
WT3_first_smooth = savgol_filter(WT3_first, window_length, polyorder)
WT4_first_smooth = savgol_filter(WT4_first, window_length, polyorder)
WT5_first_smooth = savgol_filter(WT5_first, window_length, polyorder)

KO_first_smooth = savgol_filter(KO_first, window_length, polyorder)
KO1_first_smooth = savgol_filter(KO1_first, window_length, polyorder)
KO2_first_smooth = savgol_filter(KO2_first, window_length, polyorder)
KO3_first_smooth = savgol_filter(KO3_first, window_length, polyorder)
KO4_first_smooth = savgol_filter(KO4_first, window_length, polyorder)
KO5_first_smooth = savgol_filter(KO5_first, window_length, polyorder)

# Smooth second derivatives
WT_second_smooth = savgol_filter(WT_second, window_length, polyorder)
WT1_second_smooth = savgol_filter(WT1_second, window_length, polyorder)
WT2_second_smooth = savgol_filter(WT2_second, window_length, polyorder)
WT3_second_smooth = savgol_filter(WT3_second, window_length, polyorder)
WT4_second_smooth = savgol_filter(WT4_second, window_length, polyorder)
WT5_second_smooth = savgol_filter(WT5_second, window_length, polyorder)

KO_second_smooth = savgol_filter(KO_second, window_length, polyorder)
KO1_second_smooth = savgol_filter(KO1_second, window_length, polyorder)
KO2_second_smooth = savgol_filter(KO2_second, window_length, polyorder)
KO3_second_smooth = savgol_filter(KO3_second, window_length, polyorder)
KO4_second_smooth = savgol_filter(KO4_second, window_length, polyorder)
KO5_second_smooth = savgol_filter(KO5_second, window_length, polyorder)


# %%
# --- Plot smoothed first derivatives ---
plt.figure(figsize=(14, 9))

# WT (orange)
plt.plot(wn, WT_first_smooth, color='orange', linewidth=2, label="WT_1")
plt.plot(wn, WT1_first_smooth, color='orange', linewidth=2, label="WT_2", alpha=0.7)
plt.plot(wn, WT2_first_smooth, color='orange', linewidth=2, label="WT_3", alpha=0.7)
plt.plot(wn, WT3_first_smooth, color='orange', linewidth=2, label="WT_4", alpha=0.7)
#plt.plot(wn, WT4_first_smooth, color='orange', linewidth=2, label="WT_5", alpha=0.7)
plt.plot(wn, WT5_first_smooth, color='orange', linewidth=2, label="WT_6", alpha=0.7)

# KO (green)
plt.plot(wn, KO_first_smooth, color='green', linewidth=2, label="KO_1")
plt.plot(wn, KO1_first_smooth, color='green', linewidth=2, label="KO_2", alpha=0.7)
plt.plot(wn, KO2_first_smooth, color='green', linewidth=2, label="KO_3", alpha=0.7)
plt.plot(wn, KO3_first_smooth, color='green', linewidth=2, label="KO_4", alpha=0.7)
plt.plot(wn, KO4_first_smooth, color='green', linewidth=2, label="KO_5", alpha=0.7)
plt.plot(wn, KO5_first_smooth, color='green', linewidth=2, label="KO_6", alpha=0.7)

plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
plt.ylabel('First Derivative (a.u.)', fontsize=14)
plt.title('Smoothed First Derivative of Spectra', fontsize=16)
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([950, 1800])
plt.tight_layout()
plt.show()

# %%
# --- Plot smoothed second derivatives ---
plt.figure(figsize=(14, 9))

# WT (orange)
plt.plot(wn, WT_second_smooth, color='orange', linewidth=2, label="WT_1")
plt.plot(wn, WT1_second_smooth, color='orange', linewidth=2, label="WT_2", alpha=0.7)
#plt.plot(wn, WT2_second_smooth, color='orange', linewidth=2, label="WT_3", alpha=0.7)
plt.plot(wn, WT3_second_smooth, color='orange', linewidth=2, label="WT_4", alpha=0.7)
plt.plot(wn, WT4_second_smooth, color='orange', linewidth=2, label="WT_5", alpha=0.7)
plt.plot(wn, WT5_second_smooth, color='orange', linewidth=2, label="WT_6", alpha=0.7)

# KO (green)
plt.plot(wn, KO_second_smooth, color='green', linewidth=2, label="KO_1")
plt.plot(wn, KO1_second_smooth, color='green', linewidth=2, label="KO_2", alpha=0.7)
plt.plot(wn, KO2_second_smooth, color='green', linewidth=2, label="KO_3", alpha=0.7)
plt.plot(wn, KO3_second_smooth, color='green', linewidth=2, label="KO_4", alpha=0.7)
plt.plot(wn, KO4_second_smooth, color='green', linewidth=2, label="KO_5", alpha=0.7)
plt.plot(wn, KO5_second_smooth, color='green', linewidth=2, label="KO_6", alpha=0.7)

plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
plt.ylabel('Second Derivative (a.u.)', fontsize=14)
plt.title('Smoothed Second Derivative of Spectra', fontsize=16)
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([950, 1800])
plt.tight_layout()
plt.show()

# %%
# --- Plot smoothed second derivatives (with WT shifted up by 0.02 and all alphas = 0.9) ---
plt.figure(figsize=(14, 12))

# Shift amount
shift = 0.002

# WT (orange, shifted up, alpha=0.9)
plt.plot(wn, WT_second_smooth + shift, color='orange', linewidth=2, label="WT_1", alpha=0.9)
plt.plot(wn, WT1_second_smooth + shift, color='orange', linewidth=2, label="WT_2", alpha=0.9)
plt.plot(wn, WT2_second_smooth + shift, color='orange', linewidth=2, label="WT_3", alpha=0.9)
plt.plot(wn, WT3_second_smooth + shift, color='orange', linewidth=2, label="WT_4", alpha=0.9)
plt.plot(wn, WT4_second_smooth + shift, color='orange', linewidth=2, label="WT_5", alpha=0.9)
plt.plot(wn, WT5_second_smooth + shift, color='orange', linewidth=2, label="WT_6", alpha=0.9)

# KO (green, no shift, alpha=0.9)
plt.plot(wn, KO_second_smooth, color='green', linewidth=2, label="KO_1", alpha=0.9)
plt.plot(wn, KO1_second_smooth, color='green', linewidth=2, label="KO_2", alpha=0.9)
plt.plot(wn, KO2_second_smooth, color='green', linewidth=2, label="KO_3", alpha=0.9)
plt.plot(wn, KO3_second_smooth, color='green', linewidth=2, label="KO_4", alpha=0.9)
plt.plot(wn, KO4_second_smooth, color='green', linewidth=2, label="KO_5", alpha=0.9)
plt.plot(wn, KO5_second_smooth, color='green', linewidth=2, label="KO_6", alpha=0.9)

plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
plt.ylabel('Second Derivative (a.u.)', fontsize=14)
plt.title('Smoothed Second Derivative of Spectra (WT shifted up, α=0.9)', fontsize=16)
plt.legend(loc="best", fontsize=10)
plt.grid(True, alpha=0.3)
plt.xlim([950, 1800])
plt.tight_layout()
plt.show()

#%%
#Derivative- Averages
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 12))

# Shift amount
shift = 0.00

# Stack all WT and KO smoothed second derivatives
WT_all = np.vstack([
    WT_second_smooth,
    WT1_second_smooth,
    #WT2_second_smooth,
    WT3_second_smooth,
    WT4_second_smooth,
    WT5_second_smooth
])

KO_all = np.vstack([
    KO_second_smooth,
    KO1_second_smooth,
    KO2_second_smooth,
    KO3_second_smooth,
    KO4_second_smooth,
    KO5_second_smooth
])

# Calculate means
WT_mean = WT_all.mean(axis=0)
KO_mean = KO_all.mean(axis=0)

# Plot mean curves
plt.plot(wn, WT_mean + shift, color='orange', linewidth=3, label="WT (mean)", alpha=0.9)
plt.plot(wn, KO_mean, color='green', linewidth=3, label="KO (mean)", alpha=0.9)

# Labels and settings
plt.xlabel('Wavenumber (cm$^{-1}$)', fontsize=14)
plt.ylabel('Second Derivative (a.u.)', fontsize=14)
plt.title('Mean Smoothed Second Derivative of WT and KO Spectra', fontsize=16)
plt.legend(loc="best", fontsize=12)
#plt.grid(True, alpha=0.3)
plt.xlim([950, 1800])
plt.tight_layout()
plt.show()


#%%

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json  # Ensure that json is imported
import joblib

label_dict = {}

# Define your patch output directory
patch_output_dir = "path_to_your_patch_directory"  # Set the correct path to your patch directory

# Create the directory if it doesn't exist
os.makedirs(patch_output_dir, exist_ok=True)

def extract_and_save_patches(masked_image, sample_name, label, patch_size=(5, 5), num_patches=5000):
    ny, nx, nz = masked_image.shape
    patch_h, patch_w = patch_size
    valid_mask = ~np.isnan(masked_image[:, :, 0])
    valid_y, valid_x = np.where(valid_mask)
    coords = list(zip(valid_y, valid_x))
    np.random.shuffle(coords)
    
    count = 0
    for y, x in coords:
        if (y + patch_h <= ny) and (x + patch_w <= nx):
            patch = masked_image[y:y + patch_h, x:x + patch_w, :]
            if not np.isnan(patch).any():
                patch_filename = f"{sample_name}_patch_{count}.npy"
                np.save(os.path.join(patch_output_dir, patch_filename), patch)
                
                # Store label in dictionary
                label_dict[patch_filename] = label
                
                count += 1
                if count >= num_patches:
                    break

if len(masked_cubes_list) >= 2:
    extract_and_save_patches(roi_masked_cubes_list[0], "T1", 1)
    extract_and_save_patches(roi_masked_cubes_list[1], "T3", 2)
    extract_and_save_patches(roi_masked_cubes_list[2], "T4", 3)
    extract_and_save_patches(roi_masked_cubes_list[3], "T5", 4)
    extract_and_save_patches(roi_masked_cubes_list[4], "T6", 5)
    # Save labels as JSON
    with open(os.path.join(patch_output_dir, "labels.json"), "w") as json_file:
        json.dump(label_dict, json_file, indent=4)

    print(f"Patches and labels saved in '{patch_output_dir}' directory.")
else:
    print("Patch extraction skipped due to missing masked images.") 
#%%

import seaborn as sns
import json
import numpy as np
import os

# =============================================================================
# Load Data and Labels
# =============================================================================
label_json_path = os.path.join(patch_output_dir, "labels.json")

# Load labels from JSON
with open(label_json_path, "r") as json_file:
    label_dict = json.load(json_file)

# Initialize lists to store data
X = []  # Feature vectors (flattened patches)
y = []  # Labels

# Load patches and flatten them into feature vectors
for patch_filename, label in label_dict.items():
    patch_path = os.path.join(patch_output_dir, patch_filename)
    patch = np.load(patch_path)  # Shape: (patch_size, patch_size, spectral_bands)
    
    # Flatten the patch to 1D feature vector
    X.append(patch.flatten())
    y.append(label)  # Corresponding label

X = np.array(X)  # Convert to NumPy array
y = np.array(y)

# Check if data is being loaded correctly
print(f"Loaded {len(X)} samples.")
print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"First 10 labels: {y[:10]}")

# Extract unique class labels dynamically
class_labels = [str(label) for label in np.unique(y)]
print(f"Class labels: {class_labels}")

# =============================================================================
# Split Data into Training and Testing Sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check if data is correctly split
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

# =============================================================================
# Train Random Forest Classifier
# =============================================================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Check if training data is non-empty
if len(X_train) > 0 and len(y_train) > 0:
    rf_model.fit(X_train, y_train)
    print("Model training complete.")
else:
    print("Training data is empty. Check data preprocessing.")

# =============================================================================
# Evaluate the Model
# =============================================================================
y_pred = rf_model.predict(X_test)

# Debugging step: Check contents of y_test and y_pred
print(f"First 10 true labels in y_test: {y_test[:10]}")
print(f"First 10 predicted labels in y_pred: {y_pred[:10]}")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_labels))

# =============================================================================
# Save the Model
# =============================================================================
model_path = os.path.join(patch_output_dir, "random_forest_model.pkl")
joblib.dump(rf_model, model_path)
print(f"Model saved as {model_path}")

# =============================================================================
# Confusion Matrix Visualization (Normalized)
# =============================================================================
# Generate normalized confusion matrix
cm = confusion_matrix(y_test, y_pred, normalize='true')  # Normalize by rows (true labels)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.show()

# Check accuracy from confusion matrix
accuracy_from_cm = np.sum(np.diag(cm)) / np.sum(cm)
print(f"Accuracy from confusion matrix: {accuracy_from_cm:.4f}")

#%%

#import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================================
# Load Data and Labels
# =============================================================================
label_json_path = os.path.join(patch_output_dir, "labels.json")

# Load labels from JSON
with open(label_json_path, "r") as json_file:
    label_dict = json.load(json_file)

# Initialize lists to store data
X = []  # Feature vectors (patches with spatial dimensions)
y = []  # Labels

# Load patches and reshape them into 4D arrays (height, width, channels)
for patch_filename, label in label_dict.items():
    patch_path = os.path.join(patch_output_dir, patch_filename)
    patch = np.load(patch_path)  # Shape: (patch_size, patch_size, spectral_bands)
    
    # Check if the patch is 2D (height, width) and add channel dimension if necessary
    if len(patch.shape) == 2:  # If patch is 2D (height, width), add channel dimension
        patch = np.expand_dims(patch, axis=-1)  # Shape becomes (patch_size, patch_size, 1)
    
    X.append(patch)
    y.append(label)  # Corresponding label

X = np.array(X)  # Convert to NumPy array
y = np.array(y)

# Check if data is being loaded correctly
print(f"Loaded {len(X)} samples.")
print(f"Feature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"First 10 labels: {y[:10]}")

# Extract unique class labels dynamically
class_labels = [str(label) for label in np.unique(y)]
print(f"Class labels: {class_labels}")

# =============================================================================
# Convert Labels to 0-based
# =============================================================================
y = np.array([label_dict[patch_filename] for patch_filename in label_dict.keys()])  # Extract labels

# Normalize labels dynamically
y = y - np.min(y)

# Ensure number of unique classes is correct
num_classes = len(np.unique(y))

print("Transformed unique labels:", np.unique(y))  # Debugging


# =============================================================================
# Split Data into Training and Testing Sets
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =============================================================================
# Build CNN Model
# =============================================================================
cnn_model = models.Sequential()

# Add convolutional layers with padding
cnn_model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train.shape[1:]))
cnn_model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

# Apply Global Average Pooling to avoid spatial dimension issues
cnn_model.add(layers.GlobalAveragePooling2D())  # This layer reduces the spatial dimensions to 1

# Fully connected layers
cnn_model.add(layers.Dense(128, activation='relu'))
cnn_model.add(layers.Dropout(0.5))  # Dropout to avoid overfitting
cnn_model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for multi-class classification

# Compile the model
cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# =============================================================================
# Train the CNN Model
# =============================================================================
try:
    cnn_model.fit(X_train, y_train, epochs=150, batch_size=32, validation_data=(X_test, y_test))
except Exception as e:
    print(f"Error during model training: {e}")
    raise

# =============================================================================
# Evaluate the Model
# =============================================================================
y_pred = np.argmax(cnn_model.predict(X_test), axis=-1)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=np.unique(y).astype(str)))

# =============================================================================
# Confusion Matrix Visualization (Normalized)
# =============================================================================
cm = confusion_matrix(y_test, y_pred, normalize='true')  # Normalize by rows (true labels)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix")
plt.show()

# Check accuracy from confusion matrix
accuracy_from_cm = np.sum(np.diag(cm)) / np.sum(cm)
print(f"Accuracy from confusion matrix: {accuracy_from_cm:.4f}")

