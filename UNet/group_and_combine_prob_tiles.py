#!/usr/bin/env python3.10

# by Nir Gutman (nirgutman013@gmail.com)

import os
import shutil
import geopandas as gpd
import argparse
import logging
import subprocess


def organize_files_by_group(shapefile_path, input_dir, output_dir):
    """
    Organizes files into subdirectories based on the "Group" field in a shapefile.
    After sorting, combines tiles for each group using an external shell script.

    Parameters:
        shapefile_path (str): Path to the shapefile containing "Group" and "filename" fields.
        input_dir (str): Directory containing the input files.
        output_dir (str): Directory where the grouped files will be stored.
    """
    # Load the shapefile
    logging.info("Loading shapefile...")
    gdf = gpd.read_file(shapefile_path)

    # Check required fields
    required_fields = {"Group", "filename"}
    if not required_fields.issubset(gdf.columns):
        logging.error(f"The shapefile must contain the following fields: {required_fields}")
        raise ValueError(f"The shapefile must contain the following fields: {required_fields}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    group_dirs = {}

    # Iterate through the shapefile rows
    logging.info("Organizing files by group...")
    for _, row in gdf.iterrows():
        group = str(row["Group"])
        file_name = row["filename"]

        # Build paths
        source_file = os.path.join(input_dir, file_name)
        group_dir = os.path.join(output_dir, group)
        destination_file = os.path.join(group_dir, file_name)

        # Ensure the group directory exists
        os.makedirs(group_dir, exist_ok=True)
        group_dirs[group] = group_dir  # Store group directory

        # Copy file if it exists
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            logging.info(f"Copied {file_name} to {group_dir}")
        else:
            logging.warning(f"File {file_name} not found in {input_dir}")

    # Combine tiles for each group
    logging.info("Combining tiles for each group...")
    for group, group_dir in group_dirs.items():
        combined_tif_path = os.path.join(output_dir, f"prob_sum_group{group}.tif")
        combine_tiles(group_dir, output_dir, combined_tif_path)
        logging.info(f"Combined TIF created: {combined_tif_path}")

    logging.info("Process complete.")


def combine_tiles(tiles_dir, output_dir, combined_tif_path):
    """
    Uses a shell script to combine tiles into a single TIF file.

    Parameters:
        tiles_dir (str): Directory containing the tiles to combine.
        output_dir (str): Directory where the combined TIF will be stored.
        combined_tif_path (str): Path for the output combined TIF file.
    """
    try:
        script = "./combine_tiles.sh"
        subprocess.run([script, tiles_dir, output_dir, os.path.basename(combined_tif_path)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to combine tiles for {tiles_dir}: {e}")
    except FileNotFoundError:
        logging.error(
            "The combine_tiles.sh script is not found or not executable. Ensure it is in the same directory as this script.")


if __name__ == "__main__":
    # Argument parser for terminal usage
    parser = argparse.ArgumentParser(description="Organize files by group based on a shapefile and combine tiles.")
    parser.add_argument("shapefile", type=str, help="Path to the shapefile with 'Group' and 'file_name' fields.")
    parser.add_argument("input_dir", type=str, help="Directory containing the input files.")
    parser.add_argument("output_dir", type=str,
                        help="Directory where the grouped files and combined TIFs will be stored.")
    parser.add_argument("--log", type=str, default="info", help="Logging level: debug, info, warning, error, critical.")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    logging_level = getattr(logging, args.log.upper(), logging.INFO)
    logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        organize_files_by_group(args.shapefile, args.input_dir, args.output_dir)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
