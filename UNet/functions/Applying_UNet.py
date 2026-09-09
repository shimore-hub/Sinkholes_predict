#!/usr/bin/env python3
# Optimized for Windowed Streaming Inference (OOM-Safe)
# Original code by Nir Gutman, Adapted for large-scale streaming

import os
import sys
import time
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import tqdm
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from model import build_unet
import SinkTools2 as st

start_time = time.time()

_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d | Sinkhole_UNet_Inference | %(levelname)s | %(message)s',
                              datefmt='%Y-%m-%dT%H:%M:%S')

# Default Parameters
VERBOSE = True
LOG_FILE = None
LOG_LEVEL = 'INFO'
TILE_SIZE = 256 # Tile size in both x and y dimensions (pixels)
THRESHOLD = 0.5 # Threshold for binary classification
BATCH_SIZE = 4 # Batch size for inference
MOSAIC_PATH = None # Output mosaic file path
SHAPEFILE_PATH = None # Output shapefile path
log = logging.getLogger('Sinkhole_UNet_Inference')
log.setLevel(LOG_LEVEL)


def restricted_float(x):
    """
    Check if float is in the range (0.0, 1.0]

    Args:
    x (str): The float value to check.

    Returns:
        float: The validated float value.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid float or not in the range (0.0, 1.0]
    """
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x!r} is not a valid float")
    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x!r} not in range [0.0, 1.0]")
    return x


DESCRIPTION = """
Optimized Script for testing a trained U-Net model on large DSM files.
Uses Windowed Streaming to prevent OOM errors and avoid memory issues.
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
parser.add_argument('-v', help='verbose', action='store_true', default=VERBOSE)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=LOG_LEVEL)
parser.add_argument('--logfile', default=LOG_FILE)
parser.add_argument('-ts', '--tile_size', default=TILE_SIZE, type=int, help='Tile size (pixels)')
parser.add_argument('-th', '--threshold', default=THRESHOLD, type=restricted_float, help='Threshold for binary classification')
parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int, help='Batch size for inference')
parser.add_argument('-Om', '--output_mosaic', default=MOSAIC_PATH, help='Output Mosaic File Path')
parser.add_argument('-Os', '--output_shp', default=SHAPEFILE_PATH, help='Path to the output shapefile')
parser.add_argument('input_DSM_path', help='Path to the DSM (.ers file)')
parser.add_argument('Model_Path', help='Path to the trained model')

    
def set_logger(log, verbose, log_level, logfile):
    """
    Set up the logger with the specified log level and output options.
    """
    log.setLevel(log_level)
    if verbose:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if logfile:
        fh = TimedRotatingFileHandler(logfile, when='midnight', utc=True)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        log.addHandler(fh)


def load_model(checkpoint_path, device):
    """
    Load the U-Net model from the specified checkpoint path and set it for evaluation.

    Args:
        checkpoint_path (str): Path to the model checkpoint file.
        device (torch.device): The device to load the model onto (CPU or GPU).

    Returns:
        torch.nn.Module: The loaded and evaluated U-Net model.
    """
    model = build_unet()
    model = model.to(device) # load model to device (CPU or GPU)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device)) # load model weights
    model.eval() # set model to evaluation mode
    return model


def create_output_profile(src):
    """
    Create an output profile for the output raster.

    Args:
        src (rasterio.DatasetReader): The input raster dataset.

    Returns:
        dict: The updated profile for the output raster.
    """
    profile = src.profile
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0, driver='GTiff')
    return profile


def read_tile(src, window, tile_size, max_elevation=-300, nodata_value=-500):
    """
    Read a tile from the input raster and pad it if necessary.

    Args:
        src (rasterio.DatasetReader): The input raster dataset.
        window (rasterio.windows.Window): The window defining the tile to read.
        tile_size (int): The size of the tile in pixels.

    Returns:
        np.ndarray: The padded tile data.
    """
    tile = src.read(1, window=window)

    # Use the validation function to skip irrelevant tiles
    if not is_valid_tile(tile, max_elevation, nodata_value):
        return None
    
    # Padding for edge tiles in case tile is too small
    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
        padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
        padded_tile[:tile.shape[0], :tile.shape[1]] = tile
        return padded_tile
    
    return tile


def is_valid_tile(tile, max_elevation=-300, nodata_value=-500):
    """
    Check if a tile is valid based on its elevation values.

    Args:
        tile (np.ndarray): The tile data to check.
        max_elevation (float): The maximum elevation value to consider the tile valid.
        nodata_value (float): The NoData value to ignore in the tile.

    Returns:
        bool: True if the tile is valid, False otherwise.
    """
    # Filter out NoData values
    valid_pixels = tile[tile > nodata_value]

    # filter out high altitude pixels
    if valid_pixels.size == 0 or np.min(valid_pixels) > max_elevation:
        return False

    return True


def normalize_tile(tile, nodata_value=-500):
    """
    Normalize a tile to the range [0, 1].

    Args:
        tile (np.ndarray): The tile data to normalize.

    Returns:
        np.ndarray: The normalized tile data.
    """
    valid_mask = tile > nodata_value
    norm_tile = np.zeros_like(tile, dtype=np.float32)
    valid_pixels = tile[valid_mask]
    
    if valid_pixels.size > 0:
        tile_min, tile_max = valid_pixels.min(), valid_pixels.max()
        if tile_max - tile_min > 0:
            norm_tile[valid_mask] = (tile[valid_mask] - tile_min) / (tile_max - tile_min)
            
    return norm_tile


def process_batch(model, batch_data, device, threshold):
    """
    Process a batch of tiles through the U-Net model.

    Args:
        model (torch.nn.Module): The U-Net model for inference.
        batch_data (list): List of normalized tile data.
        device (torch.device): The device to run the model on (CPU or GPU).
        threshold (float): The threshold for binary classification.

    Returns:
        np.ndarray: The predicted masks for the batch of tiles.
    """
    tensor_batch = torch.tensor(np.array(batch_data), dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        preds = model(tensor_batch)
        preds = torch.sigmoid(preds)
        preds = preds.cpu().numpy().squeeze(1)

    preds_mask = np.where(preds >= threshold, 1, 0).astype(np.uint8)
    return preds_mask


def write_predictions(dst, batch_windows, preds_mask, tile_size, height, width):
    """
    Write the predictions to the output raster.

    Args:
        dst (rasterio.DatasetWriter): The output raster dataset.
        batch_windows (list): List of windows corresponding to the batch of tiles.
        preds_mask (np.ndarray): The predicted masks.
        tile_size (int): The size of each tile in pixels.
        height (int): The height of the input raster.
        width (int): The width of the input raster.

    Returns:
        None
    """
    for i, win in enumerate(batch_windows):
        valid_h = min(tile_size, height - win.row_off)
        valid_w = min(tile_size, width - win.col_off)

        out_win = Window(win.col_off, win.row_off, valid_w, valid_h)
        dst.write(preds_mask[i, :valid_h, :valid_w], 1, window=out_win)


def process_raster(input_dem_path, output_mask_path, model, device, batch_size=32, tile_size=256, threshold=0.5):
    """
    Process the input raster in a windowed streaming manner and save the predictions.

    Args:
        input_dem_path (str): Path to the input DSM raster.
        output_mask_path (str): Path to save the output mask raster.
        model (torch.nn.Module): The U-Net model for inference.
        device (torch.device): The device to run the model on (CPU or GPU).
        batch_size (int): The number of tiles to process in a batch.
        tile_size (int): The size of each tile in pixels.
        threshold (float): The threshold for binary classification.

    Returns:
        None
    """
    with rasterio.open(input_dem_path) as src:
        profile = create_output_profile(src)
        height, width = src.height, src.width
        
        with rasterio.open(output_mask_path, 'w', **profile) as dst:
            batch_windows, batch_data = [], []
            total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
            
            with tqdm.tqdm(total=total_tiles, desc="Predicting Tiles") as pbar:
                for row_off in range(0, height, tile_size):
                    for col_off in range(0, width, tile_size):
                        window = Window(col_off, row_off, tile_size, tile_size)
                        
                        # Read, Filter, and Pad
                        tile = read_tile(src, window, tile_size, max_elevation, nodata_value)
                        
                        if tile is None:
                            pbar.update(1)
                            continue
                            
                        # Normalize (ignoring NoData)
                        norm_tile = normalize_tile(tile, nodata_value)
                        
                        batch_windows.append(window)
                        batch_data.append(norm_tile)
                        
                        # Process Full Batch
                        if len(batch_data) == batch_size:
                            preds_mask = process_batch(model, batch_data, device, threshold)
                            write_predictions(dst, batch_windows, preds_mask, tile_size, height, width)
                            batch_windows, batch_data = [], []
                            
                        pbar.update(1)
                        
            # Process Remaining Tiles in the last batch
            if len(batch_data) > 0:
                preds_mask = process_batch(model, batch_data, device, threshold)
                write_predictions(dst, batch_windows, preds_mask, tile_size, height, width)
                
    log.info("Inference complete. Mask saved successfully.")

    
if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_logger(log, args.v, args.log_level, args.logfile)
    
    if not torch.cuda.is_available():
        log.warning("WARNING: No GPU detected! PyTorch will use the CPU. This will be very slow.")

    log.info(f"Arguments: {args}")

    input_raster_path = args.input_DSM_path
    model_path = args.Model_Path

    output_mosaic = args.output_mosaic if args.output_mosaic else os.path.join(os.path.dirname(input_raster_path), 'mosaic.tif')
    output_shp = args.output_shp if args.output_shp else os.path.join(os.path.dirname(input_raster_path), 'predictions.shp')

    log.info(f'Output Mosaic File: {output_mosaic}')
    log.info(f'Output Shapefile: {output_shp}')

    log.info('Loading U-Net Model...')
    model = load_model(model_path, device)

    # Core inference using Windowed Streaming
    process_raster(
        input_dem_path=input_raster_path, 
        output_mask_path=output_mosaic, 
        model=model, 
        device=device, 
        batch_size=args.batch_size, 
        tile_size=args.tile_size, 
        threshold=args.threshold
    )


    try:
        log.info('Converting to polygons and cleaning 0-value polygons...')
        st.pred2polygons(output_mosaic, output_shp)
        st.del_0val_polygons(output_shp)
        log.info('Finished successfully!')
    except Exception as e:
        log.error(f"Error during SinkTools2 execution: {e}")

    log.info(f"Total time: {(time.time() - start_time) / 60:.2f} minutes.")