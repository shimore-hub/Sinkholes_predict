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
from osgeo import gdal, ogr
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
TILE_SIZE_X = 256 # Tile size in X direction
TILE_SIZE_Y = 256 # Tile size in Y direction
THRESHOLD = 0.5 # Threshold for binary classification
BATCH_SIZE = 4 # Batch size for inference
MOSAIC_PATH = None # Output mosaic file path
SHAPEFILE_PATH = None # Output shapefile path
log = logging.getLogger('Sinkhole_UNet_Inference')
log.setLevel(LOG_LEVEL)

DESCRIPTION = """
Optimized Script for testing a trained U-Net model on large DSM files.
Uses Windowed Streaming to prevent OOM errors and avoid.
"""

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=DESCRIPTION)
parser.add_argument('-v', help='verbose', action='store_true', default=VERBOSE)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=LOG_LEVEL)
parser.add_argument('--logfile', default=LOG_FILE)
parser.add_argument('-tsx', '--tile_size_x', default=TILE_SIZE_X, type=int)
parser.add_argument('-tsy', '--tile_size_y', default=TILE_SIZE_Y, type=int)
parser.add_argument('-th', '--threshold', default=THRESHOLD, type=restricted_float, help='Threshold for binary classification')
parser.add_argument('-b', '--batch_size', default=BATCH_SIZE, type=int, help='Batch size for inference')
parser.add_argument('-Om', '--output_mosaic', default=MOSAIC_PATH, help='Output Mosaic File Path')
parser.add_argument('-Os', '--output_shp', default=SHAPEFILE_PATH, help='Path to the output shapefile')
parser.add_argument('input_DSM_path', help='Path to the DSM (.ers file)')
parser.add_argument('Model_Path', help='Path to the trained model')


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
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

def process_raster(input_dem_path, output_mask_path, model, device, batch_size=4, tile_size=256, threshold=0.5):
    """
    Process a raster image using the U-Net model for inference.

    Args:
        input_dem_path (str): Path to the input DEM file.
        output_mask_path (str): Path to the output mask file.
        model (torch.nn.Module): The U-Net model for inference.
        device (torch.device): The device to run the model on (CPU or GPU).
        batch_size (int): The number of tiles to process in a batch.
        tile_size (int): The size of each tile in pixels.
        threshold (float): The threshold for binary classification.

    Returns:
        None
    """
    log.info(f"Starting Windowed Inference on {input_dem_path}")
    
    with rasterio.open(input_dem_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=0, driver='GTiff')
        
        with rasterio.open(output_mask_path, 'w', **profile) as dst:
            height, width = src.height, src.width
            batch_windows, batch_data = [], []
            
            total_tiles = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
            
            with tqdm.tqdm(total=total_tiles, desc="Predicting Tiles") as pbar:
                for row_off in range(0, height, tile_size):
                    for col_off in range(0, width, tile_size):
                        
                        window = Window(col_off, row_off, tile_size, tile_size)
                        tile = src.read(1, window=window)
                        
                        # Nir's filter: skip tiles with NoData values completely
                        if np.min(tile) < -500:
                            pbar.update(1)
                            continue
                            
                        # Padding for edge tiles
                        if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                            padded_tile = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                            padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                            tile = padded_tile
                            
                        # Local Normalization (Nir's method)
                        tile_min, tile_max = tile.min(), tile.max()
                        if tile_max - tile_min == 0:
                            norm_tile = np.zeros_like(tile, dtype=np.float32)
                        else:
                            norm_tile = (tile - tile_min) / (tile_max - tile_min)
                            
                        batch_windows.append(window)
                        batch_data.append(norm_tile)
                        
                        # Process batch
                        if len(batch_data) == batch_size:
                            tensor_batch = torch.tensor(np.array(batch_data), dtype=torch.float32).unsqueeze(1).to(device)
                                
                            with torch.no_grad():
                                preds = model(tensor_batch)
                                preds = torch.sigmoid(preds)
                                preds = preds.cpu().numpy().squeeze(1)
                                    
                                for i, win in enumerate(batch_windows):
                                    valid_h = min(tile_size, height - win.row_off)
                                    valid_w = min(tile_size, width - win.col_off)
                                    
                                    prob_mask = preds[i, :valid_h, :valid_w]
                                    pred_mask = np.where(prob_mask >= threshold, 1, 0).astype(np.uint8)
                                    
                                    out_win = Window(win.col_off, win.row_off, valid_w, valid_h)
                                    dst.write(pred_mask, 1, window=out_win)
                                    
                                batch_windows, batch_data = [], []
                        pbar.update(1)

                if len(batch_data) > 0:
                    tensor_batch = torch.tensor(np.array(batch_data), dtype=torch.float32).unsqueeze(1).to(device)
                    
                    with torch.no_grad():
                        preds = model(tensor_batch)
                        preds = torch.sigmoid(preds)
                        preds = preds.cpu().numpy().squeeze(1)
                        
                    for i, win in enumerate(batch_windows):
                        valid_h = min(tile_size, height - win.row_off)
                        valid_w = min(tile_size, width - win.col_off)
                        
                        prob_mask = preds[i, :valid_h, :valid_w]
                        pred_mask = np.where(prob_mask >= threshold, 1, 0).astype(np.uint8)

                        out_win = Window(win.col_off, win.row_off, valid_w, valid_h)
                        dst.write(pred_mask, 1, window=out_win)

                    batch_windows, batch_data = [], []

def raster_to_vector(input_raster_path, output_vector_path):
    log.info("Converting Binary Mosaic to Shapefile...")
    raster = gdal.Open(input_raster_path)
    if raster is None:
        log.error(f"Unable to open {input_raster_path}")
        return
    driver = ogr.GetDriverByName('ESRI Shapefile')
    output_vector = driver.CreateDataSource(output_vector_path)
    layer = output_vector.CreateLayer('output', srs=None)
    layer.CreateField(ogr.FieldDefn('value', ogr.OFTInteger))
    gdal.RasterizeLayer(raster, [1], layer, options=['ATTRIBUTE=value'])
    output_vector = None

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
        tile_size=args.tile_size_x, 
        threshold=args.threshold
    )

    # Vectorization
    raster_to_vector(output_mosaic, output_shp)
    
    try:
        log.info('Converting to polygons and cleaning 0-value polygons...')
        st.pred2polygons(output_mosaic, output_shp)
        st.del_0val_polygons(output_shp)
        log.info('Finished successfully!')
    except Exception as e:
        log.error(f"Error during SinkTools2 execution: {e}")

    log.info(f"Total time: {(time.time() - start_time) / 60:.2f} minutes.")