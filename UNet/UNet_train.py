#!/usr/bin/env python3.10

# by Nir Gutman (nirgutman013@gmail.com)

import os
import time
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import DriveDataset
from model import build_unet, train, evaluate
from loss import DiceLoss, DiceBCELoss
from utils import seeding, create_dir, epoch_time
import random
import argparse
import logging
from logging.handlers import TimedRotatingFileHandler
import sys
import tqdm
import SinkTools2 as st

########################################################################################
DESCRIPTION = """
This script automates the training of a U-Net model for semantic segmentation using PyTorch. 
It includes functionalities for loading datasets, training the model, evaluating performance, 
and saving checkpoints. The key features are:

1. **Dataset Loading**:
   The script loads training and validation datasets from specified directories. The datasets 
   are expected to be in .tif format.

2. **Model Building**:
   A U-Net model is built and initialized for training. The script also supports resuming 
   training from a saved checkpoint.

3. **Training and Evaluation**:
   The script trains the U-Net model for a specified number of epochs, evaluating performance 
   on a validation set after each epoch. The best model based on validation loss is saved.

4. **Logging**:
   Comprehensive logging is implemented to provide detailed feedback on the training process, 
   including options for verbose output and logging to a file.

5. **Command-Line Arguments**:
   The script supports various command-line arguments for configuring the training process, 
   including input directories, tile sizes, batch size, learning rate, number of epochs, 
   and checkpointing options.

Using example:
./UNet_train.py train_data_dir train_masks_dir -l INFO -Cr True -Cp Checkpoint_path
"""
########################################################################################
start_time = time.time()

# Auxiliary parameters for logging
_LOG_LEVEL_STRINGS = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d | %(name)s | %(levelname)s | %(message)s',
                              datefmt='%Y-%m-%dT%H:%M:%S')

# Default parameters for logging
VERBOSE = True  # Turn to True for printing to terminal
LOG_FILE = None  # Set a file name for logging into a file
LOG_LEVEL = 'CRITICAL'  # Change to any of the _LOG_LEVEL_STRINGS values. See documentation of built-in logging module
log = logging.getLogger('PROJECTNAME')  # Replace with the name of the project or application
log.setLevel(LOG_LEVEL)  # set at default log level



# Setting commandline argument using a parser
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=DESCRIPTION,
    epilog='''Created by Nir Gutman (nirgutman013@gmail.com), 2024''')
# arguments for logging
parser.add_argument('-v', help='verbose - print q to screen?', action='store_true', default=VERBOSE)
parser.add_argument('-l', '--log_level', choices=_LOG_LEVEL_STRINGS, default=LOG_LEVEL,
                    help=f"Log level (Default: {LOG_LEVEL}). see Python's Logging module for more details")
parser.add_argument('--logfile', metavar='log file name', help='log to file', default=LOG_FILE)
# arguments for a config file
parser.add_argument('-c', '--configfile', default=None, help='Config file.')
parser.add_argument('-tsx', '--tile_size_x', default=256, help='Tile Size X')
parser.add_argument('-tsy', '--tile_size_y', default=256, help='Tile Size Y')
parser.add_argument('-tr', '--train_ratio', default=0.8, help='Train-Validation data ratio')
parser.add_argument('-bs', '--batch_size', default=64, help='Batch Size')
parser.add_argument('-lr', '--learning_rate', default=1e-4, help='Learning Rate')
parser.add_argument('-ep', '--epochs_number', default=50, help='Number of Epochs')
parser.add_argument('-Cr', '--checkpoint_resume', default=False,  help='Resume from Checkpoint')
parser.add_argument('-Cp', '--checkpoint_path', default=None, help='Model Checkpoint Path')
parser.add_argument('-Cf', '--checkpoint_folder_path', default=None, help='Model Checkpoint Folder Path')

# ADD ANY OTHER ARGUMENTS HERE
parser.add_argument('input_DSM_train_tiles_dir', help='Path to the training DSM tiles directory')
parser.add_argument('input_mask_train_tiles_dir', help='Path to the training mask tiles directory')



if __name__ == '__main__':
    seeding(42)
    args = parser.parse_args()

    # set the logger
    st.set_logger(log, args.v, args.log_level, args.logfile)

    mask_folder = args.input_mask_train_tiles_dir
    DSM_folder = args.input_DSM_train_tiles_dir

    # Validate input arguments
    if not os.path.isdir(mask_folder):
        log.error(f"Mask folder not found: {mask_folder}")
        sys.exit(1)
    if not os.path.isdir(DSM_folder):
        log.error(f"DSM folder not found: {DSM_folder}")
        sys.exit(1)

    # Load and sort dataset paths
    all_train_y = sorted(glob(os.path.join(mask_folder, "*.tif")))
    all_train_x = sorted(glob(os.path.join(DSM_folder, "*.tif")))

    if not all_train_y or not all_train_x:
        log.error("No .tif files found in the specified directories.")
        sys.exit(1)

    # Print dataset size
    data_str = f"Dataset Size:\nTrain: {len(all_train_x)}\n"
    log.info(data_str)

    # Define model and training parameters
    H = int(args.tile_size_y)
    W = int(args.tile_size_x)
    size = (H, W)
    batch_size = int(args.batch_size)
    num_epochs = int(args.epochs_number)
    lr = float(args.learning_rate)
    resume_from_checkpoint = args.checkpoint_resume

    if resume_from_checkpoint is True:
        if args.checkpoint_path is None:
            log.error("Checkpoint path is required when resuming training.")
            sys.exit(1)
        else:
            checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = args.checkpoint_path



    if args.checkpoint_folder_path is None:
        # make the directory that is two steps above DSM_folder
        checkpoint_folder_path = os.path.join(os.path.dirname(os.path.dirname(DSM_folder)), 'Model_Checkpoints_batch'+
                                              str(args.batch_size) + '_lr'+ str(args.learning_rate) + time.strftime("%Y%m%d-%H%M%S"))
        create_dir(checkpoint_folder_path)
    else:
        checkpoint_folder_path = args.checkpoint_folder_path


    # in the checkpoint folder, save a txt file with all of the run details:
    with open(os.path.join(checkpoint_folder_path, 'run_details.txt'), 'w') as f:
        f.write(f'time: {time.strftime("%Y%m%d-%H%M%S")}\n')
        f.write(f'Input DSM tiles directory: {DSM_folder}\n')
        f.write(f'Input mask tiles directory: {mask_folder}\n')
        f.write(f'Arguments: {args}\n')
        f.write(f'Checkpoint folder: {checkpoint_folder_path}\n')
        f.write(f'Tile size: {size}\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Learning rate: {lr}\n')
        f.write(f'Number of epochs: {num_epochs}\n')
        f.write(f'Resume from checkpoint: {resume_from_checkpoint}\n')
        f.write(f'Checkpoint path: {checkpoint_path}\n')

    # Split data into training and validation sets based on the defined ratio
    train_ratio = args.train_ratio  # Define training ratio here or pass as an argument
    num_train_samples = int(train_ratio * len(all_train_x))

    train_indices = random.sample(range(len(all_train_x)), num_train_samples)
    valid_indices = list(set(range(len(all_train_x))) - set(train_indices))

    train_x = [all_train_x[i] for i in train_indices]
    train_y = [all_train_y[i] for i in train_indices]

    valid_x = [all_train_x[i] for i in valid_indices]
    valid_y = [all_train_y[i] for i in valid_indices]

    log.info(f'Train: {len(train_x)} - {len(train_y)}')
    log.info(f'Valid: {len(valid_x)} - {len(valid_y)}')

    # Create custom datasets and dataloaders for training and validation
    train_dataset = DriveDataset(train_x, train_y)
    valid_dataset = DriveDataset(valid_x, valid_y)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # Check for CUDA availability and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize and transfer the model to the device
    model = build_unet().to(device)

    # Set up the optimizer, learning rate scheduler, and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()

    # Load model checkpoint if resume_from_checkpoint is True and checkpoint exists
    if resume_from_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        log.info("Checkpoint loaded successfully. Resuming training.")
    elif resume_from_checkpoint:
        log.error("Checkpoint resume is enabled, but the checkpoint path is invalid or does not exist.")
        sys.exit(1)

    # Train and validate the model for the specified number of epochs
    best_valid_loss = float("inf")
    train_loss_list = []
    valid_loss_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss, precision, recall, f1, accuracy, sensitivity, specificity = evaluate(model, valid_loader, loss_fn, device)
        epoch_checkpoint_path = f"{checkpoint_folder_path}/unet_epoch_{epoch+32}.pth"


        # Saving the model if validation loss improves
        if valid_loss < best_valid_loss:
            log.info(
                f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {epoch_checkpoint_path}")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), epoch_checkpoint_path)

        epoch_end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(epoch_start_time, epoch_end_time)

        # Print epoch results
        log.info(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        log.info(f'\tTrain Loss: {train_loss:.3f}')
        log.info(f'\t Val. Loss: {valid_loss:.3f}')
        log.info(f'\t Precision: {precision:.3f}')
        log.info(f'\t Recall: {recall:.3f}')
        log.info(f'\t F1: {f1:.3f}')
        log.info(f'\t Accuracy: {accuracy:.3f}')
        log.info(f'\t Sensitivity: {sensitivity:.3f}')
        log.info(f'\t Specificity: {specificity:.3f}')

        # Append metrics to lists
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)




    log.info('Script completed successfully.')
    log.info(f'Total execution time: {time.time() - start_time} seconds.')

    # Save metrics to a file
    metrics_path = os.path.join(checkpoint_folder_path, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write('Train Loss,Valid Loss,Precision,Recall,F1,Accuracy,Sensitivity,Specificity\n')
        for i in range(num_epochs):
            f.write(f'{train_loss_list[i]},{valid_loss_list[i]},{precision_list[i]},{recall_list[i]},{f1_list[i]},{accuracy_list[i]},{sensitivity_list[i]},{specificity_list[i]}\n')

    log.info(f'Metrics saved to: {metrics_path}')

    # Save final model
    final_model_path = os.path.join(checkpoint_folder_path, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    log.info(f'Final model saved to: {final_model_path}')

    # save plots of loss and metrics as pdf
    import matplotlib.pyplot as plt
    import numpy as np
    epochs = np.arange(num_epochs)
    plt.figure()
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, valid_loss_list, label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
    plt.savefig(os.path.join(checkpoint_folder_path, 'loss_plot.pdf'))

    plt.figure()
    plt.plot(epochs, precision_list, label='Precision')
    plt.plot(epochs, recall_list, label='Recall')
    plt.plot(epochs, f1_list, label='F1')
    plt.plot(epochs, accuracy_list, label='Accuracy')
    plt.plot(epochs, sensitivity_list, label='Sensitivity')
    plt.plot(epochs, specificity_list, label='Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.legend()
    plt.title('Metrics vs Epoch')
    plt.savefig(os.path.join(checkpoint_folder_path, 'metrics_plot.pdf'))

    log.info('Plots saved to: {checkpoint_folder_path}')