# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 19:27:06 2025

@author: hrazeghikondela
"""

import json
import os
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt

def load_pos_file(pos_filepath):
    with open(pos_filepath, 'r') as f:
        data_json = json.load(f)
    return data_json

def parse_stage_positions(stage_positions):
    """
    Parse stage positions from JSON structure.
    Returns list of dicts with keys:
    'xy_um', 'z_um', 'grid_col', 'grid_row', 'label'
    """
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
    number_str = f"{index:04d}"  # zero-padded 4-digit number
    filename = f"20X_740_{number_str}_intensity_scaled.tif"
    return os.path.join(tile_dir, filename)

def stitch_tiles(tile_dir, positions, pixel_size_um, tile_size_px):
    tile_h, tile_w = tile_size_px

    # Group positions by row and sort by x position
    positions_by_row = {}
    for idx, p in enumerate(positions):
        x_px = int(round(p['xy_um'][0] / pixel_size_um))
        y_px = int(round(p['xy_um'][1] / pixel_size_um))
        row = p['grid_row']
        col = p['grid_col']
        if row not in positions_by_row:
            positions_by_row[row] = []
        positions_by_row[row].append({'index': idx, 'x_px': x_px, 'y_px': y_px, 'grid_col': col})

    # Determine max tiles per row and total rows
    max_cols = max(len(cols) for cols in positions_by_row.values())
    max_row = max(positions_by_row.keys())

    stitched_w = max_cols * tile_w
    stitched_h = (max_row + 1) * tile_h

    stitched_image = np.zeros((stitched_h, stitched_w), dtype=np.uint16)

    for row in sorted(positions_by_row.keys()):
        tiles_in_row = positions_by_row[row]
        tiles_sorted = sorted(tiles_in_row, key=lambda x: x['x_px'])

        # Even rows reversed, odd rows normal
        if row % 1 == 0:
            tiles_ordered = tiles_sorted[::-1]
        else:
            tiles_ordered = tiles_sorted

        for place_idx, tile_info in enumerate(tiles_ordered):
            x_pos = place_idx * tile_w
            y_pos = row * tile_h
            tile_filename = get_tile_filename_by_index(tile_info['index'], tile_dir)

            if not os.path.exists(tile_filename):
                print(f"Warning: tile file not found: {tile_filename}")
                continue

            tile_img = tf.imread(tile_filename)

            if tile_img.shape[0] != tile_h or tile_img.shape[1] != tile_w:
                print(f"Warning: tile size mismatch for {tile_filename}. Expected {tile_size_px}, got {tile_img.shape[:2]}. Skipping tile.")
                continue

            stitched_image[y_pos:y_pos+tile_h, x_pos:x_pos+tile_w] = tile_img

    print(f"Stitched image min: {stitched_image.min()}, max: {stitched_image.max()}")
    return stitched_image

def main():
    pos_filepath = r'E:\FLIM_Data\2025_7_8\KO\740\Data\Sample_1\Location_1.pos'  # Update this path
    tile_dir = r'E:\FLIM_Data\2025_7_8\KO\740\Data\Sample_1\Loc_1'              # Update this path
    pixel_size_um = 0.931  # update to your pixel size in microns
    tile_size_px = (256, 256)  # update to your tile size

    pos_data = load_pos_file(pos_filepath)
    stage_positions = pos_data['map']['StagePositions']
    positions = parse_stage_positions(stage_positions)

    print(f"Found {len(positions)} stage positions.")

    stitched_img = stitch_tiles(tile_dir, positions, pixel_size_um, tile_size_px)

    save_path = os.path.join(tile_dir, 'stitched_image.tif')
    tf.imwrite(save_path, stitched_img)
    print(f"Saved stitched image to {save_path}")

    plt.figure(figsize=(12, 12))
    plt.imshow(stitched_img, cmap='gray')
    plt.title("Stitched Image")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
