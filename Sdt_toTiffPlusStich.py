# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 12:07:53 2025

@author: hrazeghikondela
"""

import os
import glob
import json
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
from sdt_read.read_openscan_sdt import read_sdt_openscan

def convert_sdt_to_tiff(tile_dir):
    os.chdir(tile_dir)
    filelist = glob.glob('*.sdt')
    print(f"Found {len(filelist)} .sdt files")

    for filename in filelist:
        data, _, _ = read_sdt_openscan(filename)

        # Save 3D transposed
        tf.imwrite(filename[:-4] + '_tf.tif', data[0].transpose(2, 0, 1).astype(np.uint16))
        
        # Save scaled 2D projection
        
        img2d = data[0].max(axis=2).astype(np.float32)
        img2d -= img2d.min()
        #if img2d.max() > 0:
            #img2d /= img2d.max()
        gamma = 0.75  # <1 boosts dark areas
        img2d = np.power(img2d, gamma) * 2500
        img2d = img2d.astype(np.uint16)
        tf.imwrite(filename[:-4] + '_intensity_scaled.tif', img2d)

        print(f"{filename} converted: shape={data[0].shape}, 2D max={img2d.max()}")

    print("Conversion complete.\n")

def load_pos_file(pos_filepath):
    with open(pos_filepath, 'r') as f:
        data_json = json.load(f)
    return data_json

def parse_stage_positions(stage_positions):
    positions = []
    for pos in stage_positions['array']:
        xy = next((d['Position_um']['array'] for d in pos['DevicePositions']['array'] if d['Device']['scalar'] == 'XYStage'), [None, None])
        z = next((d['Position_um']['array'][0] for d in pos['DevicePositions']['array'] if d['Device']['scalar'] == 'ZStage'), None)
        grid_col = pos['GridCol']['scalar']
        grid_row = pos['GridRow']['scalar']
        label = pos['Label']['scalar']
        positions.append({
            'xy_um': xy,
            'z_um': z,
            'grid_col': grid_col,
            'grid_row': grid_row,
            'label': label
        })
    return positions

def get_tile_filename_by_index(index, tile_dir):
    number_str = f"{index:04d}"
    return os.path.join(tile_dir, f"20X_740_{number_str}_intensity_scaled.tif")

def stitch_tiles(tile_dir, positions, pixel_size_um, tile_size_px):
    tile_h, tile_w = tile_size_px
    positions_by_row = {}

    for idx, p in enumerate(positions):
        x_px = int(round(p['xy_um'][0] / pixel_size_um))
        y_px = int(round(p['xy_um'][1] / pixel_size_um))
        row = p['grid_row']
        col = p['grid_col']
        if row not in positions_by_row:
            positions_by_row[row] = []
        positions_by_row[row].append({'index': idx, 'x_px': x_px, 'y_px': y_px, 'grid_col': col})

    max_cols = max(len(cols) for cols in positions_by_row.values())
    max_row = max(positions_by_row.keys())

    stitched_w = max_cols * tile_w
    stitched_h = (max_row + 1) * tile_h
    stitched_image = np.zeros((stitched_h, stitched_w), dtype=np.uint16)

    for row in sorted(positions_by_row.keys()):
        tiles = sorted(positions_by_row[row], key=lambda x: x['x_px'])

        # Even-numbered rows go left-to-right, odd-numbered rows go right-to-left
        if row % 1 == 0:
            tiles = tiles[::-1]

        for place_idx, tile_info in enumerate(tiles):
            x_pos = place_idx * tile_w
            y_pos = row * tile_h
            tile_path = get_tile_filename_by_index(tile_info['index'], tile_dir)

            if not os.path.exists(tile_path):
                print(f"Warning: tile file not found: {tile_path}")
                continue

            tile_img = tf.imread(tile_path)
            if tile_img.shape[:2] != (tile_h, tile_w):
                print(f"Warning: size mismatch for {tile_path}. Skipping.")
                continue

            stitched_image[y_pos:y_pos+tile_h, x_pos:x_pos+tile_w] = tile_img

    print(f"Stitched image min: {stitched_image.min()}, max: {stitched_image.max()}")
    return stitched_image

def main():
    # --- Update these paths ---
    tile_dir = r'E:\FLIM_Data\2025_7_8\KO\740\Data\Sample_1\LOC_3'
    pos_filepath = r'E:\FLIM_Data\2025_7_8\KO\740\Data\Sample_1\Location_3.pos'

    pixel_size_um = 0.931
    tile_size_px = (256, 256)

    # --- Convert .sdt to TIFF ---
    convert_sdt_to_tiff(tile_dir)

    # --- Load position data and stitch ---
    pos_data = load_pos_file(pos_filepath)
    stage_positions = pos_data['map']['StagePositions']
    positions = parse_stage_positions(stage_positions)

    print(f"Found {len(positions)} stage positions.")
    stitched_img = stitch_tiles(tile_dir, positions, pixel_size_um, tile_size_px)

    # --- Save stitched image ---
    save_path = os.path.join(tile_dir, 'stitched_image.tif')
    tf.imwrite(save_path, stitched_img)
    print(f"✅ Saved stitched image to: {save_path}")

    # --- Display stitched image ---
    plt.figure(figsize=(12, 12))
    plt.imshow(stitched_img, cmap='Greens_r')
    plt.title("Stitched Image")
    plt.axis('off')
    plt.show()

    # --- Display stitched image with 5x5 grid and numbers ---
    plot_grid_overlay_on_image(stitched_img, grid_size=(5, 5))


def plot_grid_overlay_on_image(image, grid_size=(5, 5), save_path='stitched_overlay.png'):
    """
    Display and save stitched image with a grid and numbered tiles overlayed.
    Numbering pattern:
    - Row 0: right → left
    - Row 1: left → right
    - Row 2: right → left
    - ...
    
    Parameters:
    - image: 2D NumPy array (stitched grayscale image)
    - grid_size: Tuple (rows, cols) defining the grid
    - save_path: Path to save the overlayed image
    """
    rows, cols = grid_size
    height, width = image.shape

    row_height = height / rows
    col_width = width / cols

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(image, cmap='gray')
    ax.set_title("Stitched Image with Grid Overlay")
    ax.axis('off')

    # Draw grid lines
    for i in range(rows + 1):
        ax.axhline(i * row_height, color='red', linewidth=1)
    for j in range(cols + 1):
        ax.axvline(j * col_width, color='red', linewidth=1)

    # Add tile numbers in custom snake pattern
    num = 1
    for i in range(rows):
        if i % 2 == 0:
            col_range = range(cols - 1, -1, -1)  # even rows: right → left
        else:
            col_range = range(cols)              # odd rows: left → right

        for j in col_range:
            y = (i + 0.5) * row_height
            x = (j + 0.5) * col_width
            ax.text(x, y, str(num), color='yellow', fontsize=18,
                    ha='center', va='center', fontweight='bold')
            num += 1

    # Save the overlayed figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Overlayed image saved to: {save_path}")

    plt.show()

            
  


if __name__ == '__main__':
    main()