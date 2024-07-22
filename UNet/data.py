
import os
import numpy as np
from osgeo import gdal
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F




class DriveDataset(Dataset):
    def __init__(self, images_path, masks_path, feature=None):
        self.images_path = images_path
        if feature is not None:
            self.images_path2 = feature
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading image """
        image = gdal.Open(self.images_path[index])
        image = image.ReadAsArray()
        if self.images_path2 is not None:
            image2 = gdal.Open(self.images_path2[index])
            image2 = image2.ReadAsArray()
            image = np.stack((image, image2), axis=2)

        # image = image.reshape(256, 256, 1)  # Assuming single channel image

        """ Reading mask """
        mask = gdal.Open(self.masks_path[index]).ReadAsArray()
        mask = np.expand_dims(mask, axis=0)  # Add batch dimension

        # Convert numpy arrays to torch tensors
        image = torch.from_numpy(image.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        # Transpose dimensions to (N, C, H, W) format
        image = image.permute(2, 0, 1)
        mask = mask.permute(0, 1, 2)  # Add channel dimension

        # Resize image and mask
        # new_height, new_width = 256, 256  # Specify the desired height and width
        # image = F.interpolate(image.unsqueeze(0),  mode='bilinear', align_corners=False)
        # mask = F.interpolate(mask.unsqueeze(0), mode='nearest')



        return image, mask

    def __len__(self):
        return self.n_samples