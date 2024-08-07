#!/usr/bin/env python3.10

# by Nir Gutman (nirgutman013@gmail.com)

import SinkTools2 as st
import os
import time
from osgeo import gdal, gdalconst
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import sys

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
This script automates the processing of Digital Surface Models (DSM) and their corresponding mask files for machine learning applications. The key functionalities include:

1. **Dividing Rasters into Tiles**: 
   The script takes an input DSM file and an input mask file, and divides them into smaller tiles based on specified 
   tile sizes and optional overlap sizes. This is useful for breaking down large raster files into manageable pieces for analysis.

2. **Filtering Tiles**:
   After creating the tiles, the script filters out empty tiles based on the mask file. This ensures that only tiles 
   with relevant data are retained for further processing.

3. **Normalization**:
   The remaining non-empty tiles are normalized to ensure consistent data ranges, which is important for many 
   machine learning algorithms. Separate directories are maintained for normalized empty and non-empty tiles.

4. **Training and Testing Split**:
   The script splits the normalized non-empty tiles and their corresponding mask tiles into training and testing 
   datasets based on a specified fraction. This is a common step in preparing data for machine learning models.

5. **Logging**:
   Comprehensive logging is implemented to provide detailed feedback on the processing steps, 
   including options for verbose output and logging to a file.

The script uses the GDAL library for raster processing tasks and scikit-learn for data normalization. 
Command-line arguments allow for flexible configuration, including input file paths, tile sizes, overlap sizes, 
logging preferences, and the fraction of data to use for training.

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
parser.add_argument('-osx', '--overlap_size_x', default=0, help='Overlap Size X')
parser.add_argument('-osy', '--overlap_size_y', default=0, help='Overlap Size Y')
parser.add_argument('-tf', '--train_fraction', default=0.8, help='Training Data Fraction')

# ADD ANY OTHER ARGUMENTS HERE
parser.add_argument('input_DSM_path', help='Path to the input DSM file')
parser.add_argument('input_mask_path', help='Path to the input mask file')


def set_logger(log, verbose=VERBOSE, log_level=LOG_LEVEL, logfile=LOG_FILE):
    """A function for setting the logger.
    log: The logger needed to be set. Typically set this log at the top of the file with logging.get(LOGGERNAME).
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


if __name__ == '__main__':
    # get user command line values
    args = parser.parse_args()
    # set the logger
    set_logger(log, args.v, args.log_level, args.logfile)
    log.info('Logger is set up.')

    # update default parameters
    if args.configfile:
        try:
            with open(args.configfile, 'r') as f:
                for line in f:
                    exec(line)
            log.info(f'Config file {args.configfile} loaded successfully.')
        except Exception as err:
            log.error(f"Error reading config file {args.configfile}:\n{err}.")
            parser.error(f"Error reading config file {args.configfile}:\n{err}.")

    # Access the required input parameter
    input_DSM_path = args.input_DSM_path
    input_mask_path = args.input_mask_path
    base_dir = os.path.dirname(input_DSM_path)

    # Define directories based on the base directory
    output_DSM_tiles_dir = os.path.join(base_dir, 'crop/DSM_tiles')
    output_notempty_DSM_tiles_dir = os.path.join(output_DSM_tiles_dir, 'not_empty')
    output_empty_DSM_tiles_dir = os.path.join(output_DSM_tiles_dir, 'empty')
    output_mask_tiles_dir = os.path.join(base_dir, 'crop/Mask_tiles')
    output_notempty_mask_tiles_dir = os.path.join(output_mask_tiles_dir, 'not_empty')
    DSM_norm_empty = os.path.join(output_DSM_tiles_dir, 'normalized/empty')
    DSM_norm_notempty = os.path.join(output_DSM_tiles_dir, 'normalized/not_empty')

    # Create directories if they don't exist
    os.makedirs(output_DSM_tiles_dir, exist_ok=True)
    os.makedirs(output_notempty_DSM_tiles_dir, exist_ok=True)
    os.makedirs(output_empty_DSM_tiles_dir, exist_ok=True)
    os.makedirs(output_mask_tiles_dir, exist_ok=True)
    os.makedirs(output_notempty_mask_tiles_dir, exist_ok=True)
    os.makedirs(DSM_norm_empty, exist_ok=True)
    os.makedirs(DSM_norm_notempty, exist_ok=True)

    # Log the directory creation
    log.info(f'Created directories for DSM and mask tiles.')

    # Print paths for verification
    log.info(f'Output DSM Tiles Directory: {output_DSM_tiles_dir}')
    log.info(f'Output Non-empty DSM Tiles Directory: {output_notempty_DSM_tiles_dir}')
    log.info(f'Output Empty DSM Tiles Directory: {output_empty_DSM_tiles_dir}')
    log.info(f'Output Mask Tiles Directory: {output_mask_tiles_dir}')
    log.info(f'Output Non-empty Mask Tiles Directory: {output_notempty_mask_tiles_dir}')
    log.info(f'DSM Normalized Empty Directory: {DSM_norm_empty}')
    log.info(f'DSM Normalized Non-empty Directory: {DSM_norm_notempty}')

    # tiles size and overlap
    tile_size_x = args.tile_size_x
    tile_size_y = args.tile_size_y
    overlap_x = args.overlap_size_x
    overlap_y = args.overlap_size_y

    # creating tiles
    log.info('Starting to divide DSM raster into tiles.')
    st.divide_raster_to_tiles(input_DSM_path, output_DSM_tiles_dir, tile_size_x, tile_size_y, overlap_x, overlap_y)
    log.info('Completed dividing DSM raster into tiles.')

    log.info('Starting to divide mask raster into tiles.')
    st.divide_raster_to_tiles(input_mask_path, output_mask_tiles_dir, tile_size_x, tile_size_y, overlap_x, overlap_y)
    log.info('Completed dividing mask raster into tiles.')

    # Filtering tiles by mask
    log.info('Filtering tiles based on mask.')
    st.check_empty_rasters(output_mask_tiles_dir, 10, output_DSM_tiles_dir)
    log.info('Completed filtering tiles based on mask.')

    log.info('Normalizing non-empty DSM tiles.')
    st.normalize_rasters_in_folder(output_notempty_DSM_tiles_dir, DSM_norm_notempty)
    log.info('Completed normalizing non-empty DSM tiles.')

    log.info('Normalizing empty DSM tiles.')
    st.normalize_rasters_in_folder(output_empty_DSM_tiles_dir, DSM_norm_empty)
    log.info('Completed normalizing empty DSM tiles.')

    log.info('Splitting data into training and testing sets.')
    st.split_data_and_masks(DSM_norm_notempty, output_notempty_mask_tiles_dir, args.train_fraction)
    log.info('Completed splitting data into training and testing sets.')

    log.info('Script completed successfully.')
    log.info(f'Total execution time: {time.time() - start_time} seconds.')
