#!/usr/bin/env python3.11

# by Nir Gutman (nirgutman013@gmail.com)

import os
import rasterio
import shutil
import tqdm

# Define directories
dsm_dir = "128overlap+200overlap_train/Mask_tiles"  # Replace with the path to your DSM tiles
mask_dir = "128overlap+200overlap_train/DSM_tiles"  # Replace with the path to your Mask tiles

# List DSM files
dsm_files = [f for f in os.listdir(dsm_dir) if f.endswith('.tif')]

# Iterate over DSM files
for dsm_file in tqdm.tqdm(dsm_files):
    dsm_path = os.path.join(dsm_dir, dsm_file)
    mask_path = os.path.join(mask_dir, dsm_file)  # Assuming masks have the same name

    try:
        # Open the DSM file
        with rasterio.open(dsm_path) as src:
            dsm_data = src.read(1)  # Read the first band

            # Check if any value is outside the range [0, 1]
            if (dsm_data < 0).any() or (dsm_data > 1).any():
                print(f"Invalid values found in {dsm_file}. Deleting DSM and corresponding mask.")

                # Delete the DSM and mask file
                os.remove(dsm_path)
                if os.path.exists(mask_path):
                    os.remove(mask_path)

    except Exception as e:
        print(f"Error processing {dsm_file}: {e}")
