#!/usr/bin/env python3.10

# by Nir Gutman (nirgutman013@gmail.com)

import os
from osgeo import gdal
import os
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
import SinkTools2 as st
import tqdm
import torch
from osgeo import gdal, ogr
import numpy as np
import torch.nn.functional as F
from model import build_unet
import time
import SinkTools2 as st
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import subprocess

start_time = time.time()

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

DESCRIPTION = """
Script for testing a trained U-Net model on DSM and mask tiles, and evaluating its performance.
"""

# Setting commandline argument using a parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=DESCRIPTION,
    epilog='''Created by Nir Gutman (nirgutman013@gmail.com), 2024''')
# arguments for logging
parser.add_argument('-v', help='verbose - print messages to screen?', action='store_true', default=VERBOSE)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=LOG_LEVEL,
                    help=f"Log level (Default: {LOG_LEVEL}). see Python's Logging module for more details")
parser.add_argument('--logfile', metavar='log file name', help='log to file', default=LOG_FILE)
# arguments for a config file
parser.add_argument('-c', '--configfile', default=None, help='Config file.')
parser.add_argument('-tsx', '--tile_size_x', default=256, help='Tile Size X')
parser.add_argument('-tsy', '--tile_size_y', default=256, help='Tile Size Y')
parser.add_argument('-tox', '--tile_overlap_x', default=0, help='Tile Overlap X')
parser.add_argument('-toy', '--tile_overlap_y', default=0, help='Tile Overlap Y')
parser.add_argument('-th', '--threshold', default=0.5, help='Threshold for binary mask prediction')
parser.add_argument('-Otd', '--output_tiles_dir', default=None, help='Output DSM Tiles Directory Path '
                                                                     '(Default: Input Directory)')
parser.add_argument('-Opd', '--output_pred_dir', default=None, help='Output Prediction Tiles Directory '
                                                                    'Path (Default: Input Directory)')
parser.add_argument('-Om', '--output_mosaic', default=None, help='Output Mosaic File Path '
                                                                 '(Default: Input Directory)')
parser.add_argument('-Os', '--output_shp', default=None, help='Path to the output shapefile '
                                                              '(Default: Input Directory)')

# ADD ANY OTHER ARGUMENTS HERE
parser.add_argument('input_DSM_path', help='Path to the DSM')
parser.add_argument('Model_Path', help='Path to the trained model')





###############################################################################
def set_logger(log, verbose=VERBOSE, log_level=LOG_LEVEL, logfile=LOG_FILE):
    """ A function for setting the logger.
    log: The logger needed to be set. Typically, set this log at the top of the file with logging.get(LOGGERNAME).
    verbose: True (print to screen) or False (do not print to screen)
    log_level: A valid logging level ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')
    logfile: A file name (including path). The file is a daily file and for long runs will change at midnight. See logging documentation for TimedRotatingFileHandler
    """

    log.setLevel(log_level)
    if verbose:
        # create console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        if logging.StreamHandler not in [h.__class__ for h in log.handlers]:
            log.addHandler(ch)
        else:
            log.warning('log Stream handler already applied.')
    if logfile:
        # create file handler
        fh = TimedRotatingFileHandler(logfile,
                                      when='midnight',
                                      utc=True)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if TimedRotatingFileHandler not in [h.__class__ for h in log.handlers]:
            log.addHandler(fh)
        else:
            log.warning('Log file handler already applied.')
        log.info(f'Log file is: {logfile}')
    else:
        log.debug('No Log file was set')


def normalize_array(array):
    """Normalize the array values to the range [0, 1]."""
    array_min = array.min()
    array_max = array.max()
    if array_max - array_min == 0:
        return array  # Avoid division by zero if all values are the same
    normalized_array = (array - array_min) / (array_max - array_min)
    return normalized_array

def divide_raster_to_tiles(input_raster_path, output_dir, tile_size_x, tile_size_y, overlap_x=0, overlap_y=0):
    # Open the input raster
    raster = gdal.Open(input_raster_path)
    if raster is None:
        print(f"Unable to open input raster {input_raster_path}")
        return

    # Get raster information
    raster_band = raster.GetRasterBand(1)
    xsize = raster.RasterXSize
    ysize = raster.RasterYSize
    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()

    # Calculate the number of tiles in x and y directions, considering overlap
    x_tiles = (xsize + tile_size_x - 1) // (tile_size_x - overlap_x)
    y_tiles = (ysize + tile_size_y - 1) // (tile_size_y - overlap_y)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(x_tiles):
        for j in range(y_tiles):
            # Calculate the offsets and tile size
            x_offset = i * (tile_size_x - overlap_x)
            y_offset = j * (tile_size_y - overlap_y)
            x_tile_size = min(tile_size_x, xsize - x_offset)
            y_tile_size = min(tile_size_y, ysize - y_offset)

            # Ensure that the tile dimensions are positive
            if x_tile_size <= 0 or y_tile_size <= 0:
                continue

            # Read the tile data
            tile = raster_band.ReadAsArray(x_offset, y_offset, x_tile_size, y_tile_size)

            # Normalize the tile data
            tile = normalize_array(tile)

            # Create a new raster dataset for the tile
            driver = gdal.GetDriverByName('GTiff')
            output_tile_path = os.path.join(output_dir, f'tile_{i}_{j}.tif')
            tile_dataset = driver.Create(output_tile_path, x_tile_size, y_tile_size, 1, gdal.GDT_Float32)
            tile_dataset.SetGeoTransform((
                geotransform[0] + x_offset * geotransform[1],
                geotransform[1],
                0,
                geotransform[3] + y_offset * geotransform[5],
                0,
                geotransform[5]
            ))
            tile_dataset.SetProjection(projection)
            tile_dataset.GetRasterBand(1).WriteArray(tile)

            # Save and close the tile dataset
            tile_dataset.FlushCache()
            tile_dataset = None

    # Close the original raster
    raster = None

def predict_mask(raster_file, checkpoint_path, device):
    """
    The predict_mask function loads a pre-trained U-Net model from a checkpoint file, processes a raster image to the appropriate input size, performs a forward pass through the model to predict a mask, and returns a binary mask indicating the areas of interest.
    :param raster_file: the tif tile file of DEM
    :param checkpoint_path: saved model path
    :param device: cuda/cpu
    :return:
    """
    """ Load model """
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Reading raster """
    raster = gdal.Open(raster_file)
    raster = raster.ReadAsArray()

    """ Resize raster """
    new_height, new_width = 256, 256
    raster = raster.astype(np.float32)
    raster = torch.from_numpy(raster)
    raster = raster.unsqueeze(0)  # Add batch dimension
    raster = F.interpolate(raster.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

    """ Prediction """
    with torch.no_grad():
        pred_mask = model(raster)
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()
        pred_mask = np.squeeze(pred_mask, axis=0)  # Remove batch dimension
        # make a mask of 0s and 1s
        pred_mask = np.where(pred_mask >= threshold, 1, 0)

    # print(f'pred_mask shape: {pred_mask.shape}')

    return pred_mask


def save_mask_as_raster(pred_mask, raster_file, output_file):
    """
    The save_mask_as_raster function saves a binary mask as a raster file.
    :param pred_mask: the binary mask
    :param raster_file: the tif tile file of DEM
    :param output_file: the output file path
    :return:
    """
    raster = gdal.Open(raster_file)
    driver = gdal.GetDriverByName('GTiff')
    output_raster = driver.Create(output_file, raster.RasterXSize, raster.RasterYSize, 1, gdal.GDT_Byte)
    output_raster.SetGeoTransform(raster.GetGeoTransform())
    output_raster.SetProjection(raster.GetProjection())
    # output_raster.GetRasterBand(1).WriteArray(pred_mask)
    output_raster.FlushCache()
    output_raster = None


def array2raster(raster_path, array, output_path, geotransform, projection):
    """
    Save a NumPy array as a raster file using GDAL.

    Parameters:
        raster_path (str): Path to the original raster file to use as a template.
        array (numpy.ndarray): NumPy array containing the data to be saved as a raster.
        output_path (str): Path to save the raster file.
        geotransform (tuple): Geotransform parameters (top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution).
        projection (str): Projection in WKT format.
    """
    # Open the original raster file
    original_raster = gdal.Open(raster_path)

    # Get the number of rows and columns from the original raster
    array = np.squeeze(array)
    # print(f'array shape: {array.shape}')
    rows, cols = array.shape

    # Create the output raster file
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)

    # Write the array data to the output raster
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(array)

    # Set the geotransform and projection
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(projection)

    # Close the output raster file
    out_band.FlushCache()


def combine_rasters_with_gdal(folder_path, output_file):
    # List all .tif files in the folder
    input_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.tif')]

    # Build the gdal_merge command
    gdal_merge_command = [
                             'gdal_merge.py',
                             '-o', output_file,
                             '-n', '0',  # NoData value to ignore
                             '-a_nodata', '0',  # NoData value to set in the output
                             '-pct',  # Use the maximum value in overlapping regions
                         ] + input_files

    # Execute the command
    subprocess.run(gdal_merge_command)


def raster_to_vector(input_raster_path, output_vector_path):
    # Open the input raster
    raster = gdal.Open(input_raster_path)
    if raster is None:
        print(f"Unable to open input raster {input_raster_path}")
        return

    # Get raster information
    xsize = raster.RasterXSize
    ysize = raster.RasterYSize
    geotransform = raster.GetGeoTransform()
    projection = raster.GetProjection()

    # Create a new vector layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    output_vector = driver.CreateDataSource(output_vector_path)
    layer = output_vector.CreateLayer('output', srs=None)

    # Add a single field to the layer
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))

    # Rasterize the vector layer
    gdal.RasterizeLayer(raster, [1], layer, options=['ATTRIBUTE=value'])

    # Close the vector layer
    output_vector = None
###############################################################################


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    set_logger(log, args.v, args.log_level, args.logfile)

    log.info(f"Arguments: {args}")

    input_raster_path = args.input_DSM_path
    model_path = args.Model_Path

    # if the output_tiles_dir directory is not provided, set it to the directory of the input raster
    if args.output_tiles_dir is None:
        output_tiles_dir = os.path.join(os.path.dirname(input_raster_path), 'tiles')
    else:
        output_tiles_dir = args.output_tiles_dir

    # if the output_pred_dir directory is not provided, set it to the directory of the input raster
    if args.output_pred_dir is None:
        output_pred_dir = os.path.join(os.path.dirname(input_raster_path), 'predictions')
    else:
        output_pred_dir = args.output_pred_dir

    # if the output_mosaic directory is not provided, set it to the directory of the input raster
    if args.output_mosaic is None:
        output_mosaic = os.path.join(os.path.dirname(input_raster_path), 'mosaic.tif')
    else:
        output_mosaic = args.output_mosaic

    # if the output_shp directory is not provided, set it to the directory of the input raster
    if args.output_shp is None:
        output_shp = os.path.join(os.path.dirname(input_raster_path), 'predictions.shp')
    else:
        output_shp = args.output_shp



    tile_size_x = args.tile_size_x  # width of each tile in pixels
    tile_size_y = args.tile_size_y  # height of each tile in pixels

    overlap_x = args.tile_overlap_x  # overlap in x direction in pixels
    overlap_y = args.tile_overlap_y  # overlap in y direction in pixels

    divide_raster_to_tiles(input_raster_path, output_tiles_dir, tile_size_x, tile_size_y, overlap_x, overlap_y)

    # Prediction for each tile
    # using the pre-trained UNet model to predict the tiles, and save the results as binary tiles

    threshold = args.threshold
    save_array = False
    calc_each_samp = False

    for tile_file in tqdm.tqdm(os.listdir(output_tiles_dir)):
        if tile_file.endswith('.tif'):
            # print(tile_file)
            tile_path = os.path.join(output_tiles_dir, tile_file)
            pred_mask = predict_mask(tile_path, model_path, device)
            pred_mask = pred_mask.squeeze()
            output_file = os.path.join(output_pred_dir, tile_file)
            # save_mask_as_raster(pred_mask, tile_path, output_file)
            array2raster(tile_path, pred_mask, output_file, gdal.Open(tile_path).GetGeoTransform(),
                         gdal.Open(tile_path).GetProjection())


    # Combining the tiles to a single raster based on the original raster
    combine_rasters_with_gdal(output_pred_dir, output_mosaic)

    # Binary Mosaic to Shapefile
    raster_to_vector(output_mosaic, output_shp)

    # converting to polygons
    st.pred2polygons(output_mosaic, output_shp)
    st.del_0val_polygons(output_shp)

