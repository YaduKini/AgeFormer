import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from scipy import ndimage
import os
import math
import random
import pandas as pd
from settings import parse_settings
from torchvision import transforms
import logging
import torch


class UKBB(Dataset):
    def __init__(self, data_dir, csv_dir):
        self.data_dir = data_dir
        self.csv_dir = csv_dir

        sets = parse_settings()
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.input_x = sets.input_x
        self.input_y = sets.input_y
        self.input_z = sets.input_z

        # from csv dir, get path of each image and store it in a dataframe
        # use patient id as key and image path as value in a dict
        age_imaging_df = pd.read_csv(csv_dir)

        self.age_imaging_df = age_imaging_df

        # age values are in float
        self.targets_list = self.age_imaging_df[["age"]].values.tolist()


    def __len__(self):
        return len(self.age_imaging_df)

    def __getitem__(self, idx):
        # read image and labels
        image_name = self.age_imaging_df.loc[idx, 'image_path']
        target = self.targets_list[idx]
        target_tensor = torch.tensor(target)
        target_tensor = target_tensor.type(torch.float32)

        # check if image file exists
        assert os.path.isfile(image_name)

        #load image
        image = nib.load(image_name)

        assert image is not None

        image_data = image.get_fdata()

        # call all data pre-processing functions here
        # 1. convert nii to tensor array
        image_tensor = self.transform(image_data)

        image_tensor = self.convert_nii_to_tensor(image_data)

        image_tensor_normalized = self.intensity_normalization(image_tensor)

        return image_tensor_normalized, target_tensor


    def convert_nii_to_tensor(self, image):
        [z,y,x] = image.shape
        image_tensor = np.reshape(image, [1,z,y,x])

        image_tensor = image_tensor.astype("float32")

        return image_tensor
    

    # https://github.com/Tencent/MedicalNet
    def intensity_normalization(self, image):
        """ perform intensity normalization on input image """

        # get non-zero voxels
        pixels = image[image > 0]

        # get mean of non-zero voxels
        mean = pixels.mean()

        # get standard deviation of non-zero voxels
        std  = pixels.std()

        # get Z transformed image
        out = (image - mean)/std

        # generate random numbers from a standard normal distribution with zero mean and unit variance
        out_random = np.random.normal(0, 1, size = image.shape)

        # replace zero voxels with random numbers generated from normal distribution to get intensity normalized image
        out[image == 0] = out_random[image == 0]
        return out




