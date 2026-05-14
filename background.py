# -*- coding: utf-8 -*-
"""
Compute background spectrum as the mean of the FIRST N pixels inside the drawn ROI.
- Draw ROI on slice_idx
- Get all ROI pixel spectra: T[roi_mask, :]  -> (Npix, Z)
- Take first N pixels (row-major order) and average them
- Save mean spectrum to .npz
- Pop-up plot (not saved)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import gc

# -------------------- Setup --------------------
os.chdir(r'D:\Filiz Lab\Data_2025\Mid-IR Data\11_18_2025')

output_root = "Area_B"
spec_dir = os.path.join(output_root, "BG_First300ROI_MeanSpectrum")
os.makedirs(spec_dir, exist_ok=True)

sample_files = {
 #   "T1": "HP_WT_1.npy",
 #   "T2": "HP_WT_2.npy",
 #   "T3": "HP_WT_3.npy",
 #   "T4": "HP_WT_4.npy",
 #   "T5": "HP_WT_5.npy",
 #   "T6": "HP_WT_6.npy",
 #   "T7": "HP_WT_7.npy",
 #   "T8": "HP_WT_8.npy",
 #   "T9": "HP_WT_9.npy",
 #   "T10": "HP_KO_1.npy",
 #   "T11": "HP_KO_2.npy",
 #   "T12": "HP_KO_3.npy",
 #   "T13": "HP_KO_4.npy",
 #   "T14": "HP_KO_5.npy",
 #   "T15": "HP_KO_6.npy",
 #   "T20": "HP_WT_14.npy",
 #   "T19": "HP_WT_12.npy",
 #   "T21": "HP_WT_13.npy",
 #   "T22": "HP_KO_8.npy",
 #    "T17": "HP_WT_10.npy"
      "T1_E":"HP_KO_1E.npy",
      "T2_E":"HP_KO_2E.npy",
      "T3_E":"HP_WT_1E.npy",
      "T4_E":"HP_WT_2E.npy"
}

slice_idx = 353
low_threshold = 0.0

N_BG_PIXELS = 30                 # <-- your request
USE_VALID_MASK = True             # restrict ROI to tissue-like pixels
HI_PCT = 99.5                     # for valid mask
LO_PCT = 1.0                      # set 0.0 to disable low clipping


# -------------------- ROI helper --------------------
class ROISelector:
    def __init__(self, ax):
        self.roi_coords = None
        self.selector = PolygonSelector(ax, self.onselect, useblit=True)
        self.cid_key_press = plt.connect("key_press_event", self.on_key_press)

    def onselect(self, verts):
        self.roi_coords = np.array(verts)

    def on_key_press(self, event):
        if event.key == "enter":
            plt.disconnect(self.cid_key_press)
            plt.close()

    def get_roi(self):
        plt.show(block=True)
        return self.roi_coords


def create_polygon_mask(shape_hw, roi_coords):
    H, W = shape_hw
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.vstack((x.ravel(), y.ravel())).T
    return Path(roi_coords).contains_points(pts).reshape((H, W))


# -------------------- Main loop --------------------
for sample_name, filename in sample_files.items():
    print(f"\nLoading {sample_name} from {filename}")
    T = np.load(filename)  # (H, W, Z)

    if slice_idx < 0 or slice_idx >= T.shape[2]:
        print(f"slice_idx={slice_idx} out of range for {sample_name} (Z={T.shape[2]}). Skipping.")
        del T
        gc.collect()
        continue

    slice2d = T[:, :, slice_idx]

    # ----- Build valid mask (optional) -----
    if USE_VALID_MASK:
        hi = np.percentile(slice2d, HI_PCT)
        if LO_PCT > 0:
            lo = max(np.percentile(slice2d, LO_PCT), low_threshold)
        else:
            lo = low_threshold
        valid2d = (slice2d > lo) & (slice2d < hi)

        view_img = np.full(slice2d.shape, np.nan, dtype=np.float32)
        view_img[valid2d] = slice2d[valid2d].astype(np.float32, copy=False)
    else:
        valid2d = np.ones(slice2d.shape, dtype=bool)
        view_img = slice2d.astype(np.float32, copy=False)

    # ----- Draw ROI -----
    fig, ax = plt.subplots(figsize=(26, 19))
    ax.imshow(view_img, cmap="jet")
    ax.set_title(f"{sample_name}: Draw ROI (BG = mean of first {N_BG_PIXELS} ROI pixels), press Enter")
    roi_coords = ROISelector(ax).get_roi()

    if roi_coords is None or roi_coords.size == 0:
        print(f"No ROI selected for {sample_name}, skipping.")
        del T
        gc.collect()
        continue

    roi_mask = create_polygon_mask(T.shape[:2], roi_coords)
    roi_mask = roi_mask & valid2d  # keep only valid pixels if enabled

    roi_count = int(roi_mask.sum())
    print(f"{sample_name}: ROI valid pixels = {roi_count}")

    if roi_count == 0:
        print(f"{sample_name}: ROI has 0 valid pixels. Try drawing over tissue or set USE_VALID_MASK=False.")
        del T
        gc.collect()
        continue

    # ----- Extract ROI spectra -----
    roi_spectra = T[roi_mask, :]  # (Npix, Z) in row-major order

    # Take first N pixels (or fewer if ROI is smaller)
    n_use = min(N_BG_PIXELS, roi_spectra.shape[0])
    bg_spectra = roi_spectra[:n_use, :]

    mean_spectrum = np.nanmean(bg_spectra, axis=0).astype(np.float32, copy=False)

    # ----- Save mean spectrum only -----
    save_path = os.path.join(spec_dir, f"bg_first{N_BG_PIXELS}_mean_spectrum_{sample_name}.npz")
    np.savez_compressed(
        save_path,
        mean_spectrum=mean_spectrum,
        n_bg_pixels_used=int(n_use),
        roi_pixel_count=roi_count,
        slice_idx=slice_idx,
        use_valid_mask=USE_VALID_MASK,
        hi_percentile=HI_PCT,
        lo_percentile=LO_PCT,
        low_threshold=low_threshold,
        source_file=filename,
    )
    print(f"Saved: {save_path}  (used {n_use}/{roi_count} ROI pixels)")

    # ----- Pop-up plot -----
    plt.figure(figsize=(10, 4))
    plt.plot(mean_spectrum)
    plt.title(f"{sample_name} - Mean Spectrum (first {n_use} ROI pixels)")
    plt.xlabel("Spectral index (band)")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.show()

    del T, roi_spectra, bg_spectra, mean_spectrum
    gc.collect()

print("\nDone.")
