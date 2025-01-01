
import os
import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F




class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, feature=None, images_path2=None):
        self.images_path = images_path
        if feature is not None:
            self.images_path2 = feature
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = gdal.Open(self.images_path[index])
        image = image.ReadAsArray()

        """ Reading mask """
        mask = gdal.Open(self.masks_path[index]).ReadAsArray()

        # Convert numpy arrays to torch tensors
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        # Check the dimensions and add channel dimension if necessary
        if len(image.shape) == 2:  # If single channel image
            image = image.unsqueeze(0)  # Add channel dimension
        elif len(image.shape) == 3 and image.shape[0] != 3:  # If (H, W, C) format
            image = image.permute(2, 0, 1)  # Change to (C, H, W)

        # Ensure mask has a channel dimension
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension

        return image, mask

        return image, mask

    def __len__(self):
        return self.n_samples