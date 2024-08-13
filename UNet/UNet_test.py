#!/usr/bin/env python3.10

# by Nir Gutman (nirgutman013@gmail.com)

import os
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import matplotlib.pyplot as plt
import SinkTools2 as st
import tqdm
import torch
from osgeo import gdal
import numpy as np
import torch.nn.functional as F
from model import build_unet
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import time

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
parser.add_argument('-sa', '--save_array', default=False, help='Save Arrays of Each Prediction')
parser.add_argument('-cs', '--calc_each_samp', default=False, help='Calculate Statistics also for each'
                                                                   ' sample separately')
parser.add_argument('-th', '--threshold', default=0.5, help='Threshold for prediction')
parser.add_argument('-O', '--output_dir', default=None, help='Path to the output directory')

# ADD ANY OTHER ARGUMENTS HERE
parser.add_argument('input_DSM_test_tiles_dir', help='Path to the testing DSM tiles directory')
parser.add_argument('input_mask_test_tiles_dir', help='Path to the testing mask tiles directory')
# add the output directory from the directory above the input directories
parser.add_argument('model_path', help='Path to the trained model file')

args = parser.parse_args()

threshold = float(args.threshold)
save_array = args.save_array
calc_each_samp = args.calc_each_samp
x_size = args.tile_size_x
y_size = args.tile_size_y

rasters_test_path = args.input_DSM_test_tiles_dir
masks_test_path = args.input_mask_test_tiles_dir
model_path = args.model_path
output_folder = args.output_dir
if output_folder is None:
    output_folder = os.path.join(os.path.dirname(os.path.dirname(args.input_DSM_test_tiles_dir)), 'predictions')


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


def calculate_metrics(predictions, labels):
    # Convert predictions and labels to boolean arrays
    predictions = np.array(predictions) > threshold  # Assuming predictions are probabilities
    labels = np.array(labels) > threshold  # Assuming labels are binary (0 or 1)

    # True Positives (TP): Predicted positive and actually positive
    TP = np.sum(np.logical_and(predictions, labels))

    # False Positives (FP): Predicted positive but actually negative
    FP = np.sum(np.logical_and(predictions, np.logical_not(labels)))

    # True Negatives (TN): Predicted negative and actually negative
    TN = np.sum(np.logical_and(np.logical_not(predictions), np.logical_not(labels)))

    # False Negatives (FN): Predicted negative but actually positive
    FN = np.sum(np.logical_and(np.logical_not(predictions), labels))

    return TP, FP, TN, FN


def predict_mask(raster_file, checkpoint_path, device):
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

    return pred_mask


def model_full_test(rasters_folder, masks_folder, output_folder, model_path, save_array=False, calc_each_samp=False):
    # check if the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        log.info(f"Output directory created: {output_folder}")

    # load the model
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # load the rasters
    rasters = os.listdir(rasters_folder)
    rasters = [raster for raster in rasters if raster.endswith('.tif')]

    TP_total = 0
    FP_total = 0
    TN_total = 0
    FN_total = 0
    # use tqdm for progress bar

    for raster in tqdm.tqdm(rasters):
        # load the raster
        raster_path = os.path.join(rasters_folder, raster)
        mask_path = os.path.join(masks_folder, raster)
        # load the raster
        raster_array = gdal.Open(raster_path)
        geotrans = raster_array.GetGeoTransform()
        proj = raster_array.GetProjection()
        raster_array = raster_array.ReadAsArray()
        # load the mask
        mask = gdal.Open(mask_path)
        mask = mask.ReadAsArray()
        # if the shape of the maske is not the same as the predefined sizes, resize it

        # resize the mask
        # new_height, new_width = 256, 256
        mask = mask.astype(np.float32)
        # mask = torch.from_numpy(mask)
        if mask.shape != (x_size, y_size):
            new_height, new_width = 256, 256
            # mask = mask.astype(np.float32)
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0)  # Add batch dimension
            mask = F.interpolate(mask.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)

        # mask = mask.unsqueeze(0)  # Add batch dimension
        # mask = F.interpolate(mask.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False)
        # predict single raster
        pred = predict_mask(raster_path, model_path, device)

        # calculate the metrics
        TP, FP, TN, FN = calculate_metrics(mask, pred)
        # calculate the metrics
        if calc_each_samp is True:
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            # accuracy = (TP + TN) / (TP + FP + TN + FN)
            # precision = TP / (TP + FP)
            # f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
            # jaccard = TP / (TP + FP + FN)
            log.info(f'raster name: {raster} \nSensitivity: {sensitivity}, specificity: {specificity}')
            # save the predicted mask

        if save_array is True:
            pred_path = os.path.join(output_folder, raster.split('.')[0] + '_pred.tif')
            # save array as raster according to the original raster using gdal not with st
            st.array2raster(raster_path, pred, pred_path, geotrans, proj)

        # update the total counts
        TP_total += TP
        FP_total += FP
        TN_total += TN
        FN_total += FN

    # calculate the total metrics
    sensitivity = TP_total / (TP_total + FN_total)
    specificity = TN_total / (TN_total + FP_total)
    accuracy = (TP_total + TN_total) / (TP_total + FP_total + TN_total + FN_total)
    precision = TP_total / (TP_total + FP_total)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    jaccard = TP_total / (TP_total + FP_total + FN_total)
    log.info(
        f'Total sensitivity: {sensitivity}, total specificity: {specificity}, total accuracy: {accuracy}, total precision: {precision}, total f1: {f1}, total jaccard: {jaccard}')


if __name__ == '__main__':
    set_logger(log, args.v, args.log_level, args.logfile)

    log.info(f"Arguments: {args}")

    # Create the output directory if not inserted as input
    if args.output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.input_DSM_test_tiles_dir), 'predictions')
    else:
        output_dir = args.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        log.info(f"Output directory created: {output_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_full_test(rasters_test_path, masks_test_path, output_folder, model_path, save_array, calc_each_samp)

    log.info(f"Script execution time: {time.time() - start_time} seconds")
