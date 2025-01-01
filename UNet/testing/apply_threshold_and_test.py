#!/usr/bin/env python3.10

# by Nir Gutman (nirgutman013@gmail.com)

import os
import sys
import argparse
import logging
import numpy as np
from osgeo import gdal
from sklearn.metrics import confusion_matrix
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import pandas as pd
import SinkTools2 as st





# Auxiliary parameters for logging
_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
                              datefmt='%Y-%m-%dT%H:%M:%S')

# Default parameters for logging
VERBOSE = False  # Turn to True for printing to terminal
LOG_FILE = None  # Set a file name for logging into a file
LOG_LEVEL = 'CRITICAL'  # Change to any of the _LOG_LEVEL_STRINGS values. See documentation of built-in logging module
log = logging.getLogger('PROJECTNAME')  # Replace with the name of the project or application
log.setLevel(LOG_LEVEL)  # set at default log level


########################################################################################################################
def apply_threshold_to_rasters(input_folder, output_folder, threshold):
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    print(threshold)

    # Get a list of all .tif files in the input folder
    tif_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

    for file in tif_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        print(f"Processing {input_path}...")

        # Open the input file
        dataset = gdal.Open(input_path)
        if dataset is None:
            print(f"Failed to open {input_path}")
            continue

        # Read the raster data as a NumPy array
        band = dataset.GetRasterBand(1)
        array = band.ReadAsArray()

        # Apply the threshold and convert to binary (1 where above threshold, else 0)
        binary_array = np.where(array > threshold, 1, 0)

        # Save the binary array to a new file
        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(output_path, dataset.RasterXSize, dataset.RasterYSize, 1, gdal.GDT_Byte)
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())
        output_dataset.SetProjection(dataset.GetProjection())

        # Write the binary array to the output raster band
        output_band = output_dataset.GetRasterBand(1)
        output_band.WriteArray(binary_array)
        output_band.SetNoDataValue(0)

        # Flush changes and close datasets
        output_band.FlushCache()
        output_dataset = None
        dataset = None

        print(f"Saved binary raster to {output_path}")


def binary_rasters_to_shapefiles(folder_path, output_folder):
    """Converts binary rasters to shapefiles."""
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.tif'):
            raster_path = os.path.join(folder_path, file_name)
            shapefile_path = os.path.join(output_folder, file_name.replace('.tif', '.shp'))
            st.pred2polygons(raster_path, shapefile_path)
            st.del_0val_polygons(shapefile_path)
            # logging.info(f"Shapefile created: {shapefile_path}")


def combine_shapefiles(input_folder, output_shapefile):
    """Combines multiple shapefiles into one."""
    gdf_list = [gpd.read_file(os.path.join(input_folder, f)) for f in os.listdir(input_folder) if f.endswith('.shp')]
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    combined_gdf.to_file(output_shapefile)
    logging.info(f"Combined shapefile created at: {output_shapefile}")


def calculate_metrics(pred_dir, mask_dir):
    """Calculates metrics for binary raster predictions."""
    global_tp, global_fp, global_fn, global_tn = 0, 0, 0, 0
    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.tif')]

    for file in pred_files:
        pred_path = os.path.join(pred_dir, file)
        mask_path = os.path.join(mask_dir, file)

        pred_ds = gdal.Open(pred_path)
        mask_ds = gdal.Open(mask_path)

        if pred_ds is None or mask_ds is None:
            logging.error(f"Could not open files for {file}")
            continue

        pred_array = pred_ds.GetRasterBand(1).ReadAsArray()
        mask_array = mask_ds.GetRasterBand(1).ReadAsArray()

        pred_binary = (pred_array > 0).astype(int)
        mask_binary = (mask_array > 0).astype(int)

        pred_flat = pred_binary.flatten()
        mask_flat = mask_binary.flatten()

        tn, fp, fn, tp = confusion_matrix(mask_flat, pred_flat, labels=[0, 1]).ravel()
        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

    total_sensitivity = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    total_specificity = global_tn / (global_tn + global_fp) if (global_tn + global_fp) > 0 else 0
    total_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    total_accuracy = (global_tp + global_tn) / (global_tp + global_tn + global_fp + global_fn) if (
            global_tp + global_tn + global_fp + global_fn) > 0 else 0

    logging.info("Overall Metrics:")
    logging.info(f"Sensitivity: {total_sensitivity:.4f}")
    logging.info(f"Specificity: {total_specificity:.4f}")
    logging.info(f"Precision: {total_precision:.4f}")
    logging.info(f"Accuracy: {total_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Pipeline script for thresholding, shapefile generation, and metrics.")
    parser.add_argument("prob_dir", help="Directory of probability rasters")
    parser.add_argument("mask_dir", help="Directory of mask rasters")
    parser.add_argument("threshold", type=float, help="Threshold value for binary classification")
    parser.add_argument("output_dir", help="Main output directory")
    parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=LOG_LEVEL,
                        help=f"Log level (Default: {LOG_LEVEL}). see Python's Logging module for more details")

    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s - %(levelname)s - %(message)s")

    # Create output directories
    binary_output = os.path.join(args.output_dir, "binary_rasters")
    shapefile_output = os.path.join(args.output_dir, "shapefiles")
    combined_shapefile = os.path.join(args.output_dir, "combined_shapefile.shp")
    log.info(f"Output directories: {binary_output}, {shapefile_output}, {combined_shapefile}")

    # Step 1: Apply threshold
    apply_threshold_to_rasters(args.prob_dir, binary_output, float(args.threshold))
    log.info("Thresholding complete.")

    # Step 2: Convert to shapefiles
    binary_rasters_to_shapefiles(binary_output, shapefile_output)
    log.info("Shapefile generation complete.")

    # Step 3: Combine shapefiles
    combine_shapefiles(shapefile_output, combined_shapefile)
    log.info("Shapefile combination complete.")

    # Step 4: Calculate metrics
    calculate_metrics(binary_output, args.mask_dir)
    log.info("Metrics calculation complete.")


if __name__ == "__main__":
    main()
