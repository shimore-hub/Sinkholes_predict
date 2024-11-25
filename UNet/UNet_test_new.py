#!/usr/bin/env python3.10

# by Nir Gutman (nirgutman013@gmail.com)

import os
import SinkTools2 as st
import tqdm
import torch
from osgeo import gdal
import rasterio
import numpy as np
import pandas as pd
import torch.nn.functional as F
from origin_model import build_unet
import argparse
import sys
import time
import logging
import matplotlib.pyplot as plt
from logging.handlers import TimedRotatingFileHandler
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, confusion_matrix

start_time = time.time()


DESCRIPTION = """
Script for testing a trained U-Net model on DSM and mask tiles, and evaluating its performance.

example run: 
./UNet_test.py DSM_tiles_dir mask_tiles_dir model_path -O output_dir -v -l INFO --logfile log_file.log
"""

########################################################################################################################

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
parser.add_argument('-sa', '--save_array', default=True, help='Save Arrays of Each Prediction')
parser.add_argument('-cm', '--calcu_metrics', default=True, help='Calculate metrics for the model')
parser.add_argument('-cs', '--calc_each_samp', default=False, help='Calculate Statistics also for each'
                                                                   ' sample separately')
parser.add_argument('-th', '--threshold', default=None, help='Threshold for prediction')
parser.add_argument('-O', '--output_dir', default=None, help='Path to the output directory')

# ADD ANY OTHER ARGUMENTS HERE
parser.add_argument('input_DSM_test_tiles_dir', help='Path to the testing DSM tiles directory')
parser.add_argument('input_mask_test_tiles_dir', help='Path to the testing mask tiles directory')
# add the output directory from the directory above the input directories
parser.add_argument('model_path', help='Path to the trained model file')

args = parser.parse_args()

if args.threshold is not None:
    threshold = float(args.threshold)
else:
    threshold = None
save_array = args.save_array
calc_each_samp = args.calc_each_samp
x_size = args.tile_size_x
y_size = args.tile_size_y
calcu_metrics = args.calcu_metrics
rasters_test_path = args.input_DSM_test_tiles_dir
masks_test_path = args.input_mask_test_tiles_dir
model_path = args.model_path
output_folder = args.output_dir

if output_folder is None:
    output_folder = os.path.join(os.path.dirname(os.path.dirname(args.input_DSM_test_tiles_dir)), 'probabilities')

########################################################################################################################


def probabs_mask(raster_file, model_path, device):
    """
    predict a probabilities mask from a raster file
    :param raster_file: the DSM tile we want to predict
    :param model_path: path for the torch model file
    :param device: cuda/cpu
    :return: the probabilities mask
    """
    """ Load model """
    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
        prob_mask = model(raster)
        prob_mask = torch.sigmoid(prob_mask)
        prob_mask = prob_mask.cpu().numpy()
        prob_mask = np.squeeze(prob_mask, axis=0)  # Remove batch dimension

    return prob_mask


def model_full_test(rasters_folder, masks_folder, output_folder, model_path, save_array=True, calc_each_samp=False):
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

        # predict single raster
        probs = probabs_mask(raster_path, model_path, device)

        if save_array is True:
            prob_path = os.path.join(output_folder, raster.split('.')[0] + '.tif')
            # save array as raster according to the original raster using gdal not with st
            st.save_array_as_tiff(probs[0], prob_path, geotrans, proj)
            log.info(f'Probabilities saved to: {prob_path}')
            # save txt file with the name of the model file
            txt_path = os.path.join(output_folder, 'model_name.txt')
            with open(txt_path, 'w') as f:
                f.write(model_path)
            log.info(f'Model name saved to: {txt_path}')


    if calcu_metrics is True:
        # calculate the metrics
        if calc_each_samp is True:
            metrics = calculate_metrics_for_rasters(output_folder, masks_folder, num_thresholds=20, print_each_stat=True,
                                                    save_threshold=threshold, save_dir=output_folder)
            log.info(f"Metrics: {metrics}")
        else:
            metrics = calculate_metrics_for_rasters(output_folder, masks_folder, num_thresholds=20, print_each_stat=False,
                                                    save_threshold=threshold, save_dir=output_folder)
            log.info(f"Metrics: {metrics}")


        # plot the metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['threshold'], metrics['precision'], label='Precision')
        plt.plot(metrics['threshold'], metrics['recall'], label='Recall')
        plt.plot(metrics['threshold'], metrics['f1_score'], label='F1 Score')
        plt.plot(metrics['threshold'], metrics['accuracy'], label='Accuracy')
        plt.plot(metrics['threshold'], metrics['sensitivity'], label='Sensitivity')
        plt.plot(metrics['threshold'], metrics['specificity'], label='Specificity')
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs. Threshold')
        plt.legend()
        plt.grid()
        # save the plot to a folder above the output folder
        plot_path = os.path.join(os.path.dirname(output_folder), 'metrics_plot.pdf')
        plt.savefig(plot_path)
        log.info(f"Metrics plot saved to: {plot_path}")

        # save the metrics to a csv file
        metrics_path = os.path.join(os.path.dirname(output_folder), 'metrics.csv')
        metrics.to_csv(metrics_path, index=False)



def calculate_metrics(predictions, labels, threshold):
    # # Convert predictions and labels to boolean arrays
    predictions = np.array(predictions) > threshold  # Assuming predictions are probabilities
    # labels = np.array(labels) > threshold  # Assuming labels are binary (0 or 1)

    # True Positives (TP): Predicted positive and actually positive
    TP = np.sum(np.logical_and(predictions, labels))

    # False Positives (FP): Predicted positive but actually negative
    FP = np.sum(np.logical_and(predictions, np.logical_not(labels)))

    # True Negatives (TN): Predicted negative and actually negative
    TN = np.sum(np.logical_and(np.logical_not(predictions), np.logical_not(labels)))

    # False Negatives (FN): Predicted negative but actually positive
    FN = np.sum(np.logical_and(np.logical_not(predictions), labels))

    return TP, FP, TN, FN


def calculate_metrics_for_rasters(
    raster_dir,
    mask_dir,
    num_thresholds=10,
    print_each_stat=False,
    save_threshold=None,
    save_dir=None
):
    """
    Calculates metrics for rasters against masks at various thresholds.

    Parameters:
        raster_dir (str): Path to the directory containing probability raster files.
        mask_dir (str): Path to the directory containing corresponding mask files.
        num_thresholds (int): Number of thresholds to evaluate, default is 10.
        print_each_stat (bool): If True, logs metrics for each file individually.
        save_threshold (float): Specific threshold to save binary rasters, default is None.
        save_dir (str): Directory to save binary rasters, required if save_threshold is specified.

    Returns:
        pd.DataFrame: DataFrame containing metrics for all thresholds.
    """
    if save_threshold is not None and save_dir is None:
        raise ValueError("If save_threshold is provided, save_dir must also be specified.")

    thresholds = np.linspace(0, 1, num=num_thresholds)
    closest_threshold = None

    # Find the closest threshold to the specified save_threshold
    if save_threshold is not None:
        closest_threshold = thresholds[np.abs(thresholds - save_threshold).argmin()]
        logging.info(f"Closest threshold to save_threshold ({save_threshold}) is {closest_threshold}.")

    results = []

    for threshold in thresholds:
        cumulative_precision, cumulative_recall, cumulative_f1 = 0, 0, 0
        cumulative_accuracy, cumulative_sensitivity, cumulative_specificity = 0, 0, 0
        num_files = 0

        for file_name in os.listdir(raster_dir):
            raster_path = os.path.join(raster_dir, file_name)
            mask_path = os.path.join(mask_dir, file_name)

            # Ensure corresponding mask exists
            if not os.path.exists(mask_path):
                logging.warning(f"Mask not found for {file_name}, skipping.")
                continue

            # Read raster and mask
            with rasterio.open(raster_path) as raster, rasterio.open(mask_path) as mask:
                raster_data = raster.read(1)  # Read the first band
                mask_data = mask.read(1)     # Read the first band

            # Apply threshold
            predictions = (raster_data >= threshold).astype(int)

            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(mask_data.flatten(), predictions.flatten(), labels=[0, 1]).ravel()

            # Calculate metrics
            precision = precision_score(mask_data.flatten(), predictions.flatten(), zero_division=0)
            recall = recall_score(mask_data.flatten(), predictions.flatten(), zero_division=0)
            f1 = f1_score(mask_data.flatten(), predictions.flatten(), zero_division=0)
            accuracy = accuracy_score(mask_data.flatten(), predictions.flatten())
            sensitivity = recall  # Sensitivity is the same as recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            if print_each_stat:
                logging.info(
                    f"File: {file_name}, Threshold: {threshold:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, "
                    f"F1: {f1:.2f}, Accuracy: {accuracy:.2f}, Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}"
                )

            # Save the raster if the threshold matches the closest threshold
            if save_threshold is not None and np.isclose(threshold, closest_threshold):
                save_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}.tif")
                with rasterio.open(save_path, 'w',
                                   driver='GTiff',
                                   height=raster_data.shape[0],
                                   width=raster_data.shape[1],
                                   count=1,
                                   dtype=np.uint8,
                                   crs=raster.crs,
                                   transform=raster.transform) as dst:
                    dst.write(predictions, 1)

            # Accumulate metrics
            cumulative_precision += precision
            cumulative_recall += recall
            cumulative_f1 += f1
            cumulative_accuracy += accuracy
            cumulative_sensitivity += sensitivity
            cumulative_specificity += specificity
            num_files += 1

        # Store average metrics for the threshold
        if num_files > 0:
            avg_precision = cumulative_precision / num_files
            avg_recall = cumulative_recall / num_files
            avg_f1 = cumulative_f1 / num_files
            avg_accuracy = cumulative_accuracy / num_files
            avg_sensitivity = cumulative_sensitivity / num_files
            avg_specificity = cumulative_specificity / num_files

            results.append({
                "threshold": threshold,
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1,
                "accuracy": avg_accuracy,
                "sensitivity": avg_sensitivity,
                "specificity": avg_specificity
            })
        else:
            logging.warning(f"No valid files for threshold {threshold:.2f}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df

########################################################################################################################

if __name__ == '__main__':
    st.set_logger(log, args.v, args.log_level, args.logfile)

    log.info(f"Arguments: {args}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_full_test(rasters_test_path, masks_test_path, output_folder, model_path, save_array, calc_each_samp)




    log.info(f"Script execution time: {time.time() - start_time} seconds")
    log.info("Script finished successfully")