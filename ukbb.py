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
#from sklearn.preprocessing import OneHotEncoder


class UKBB(Dataset):
    def __init__(self, data_dir, csv_dir):
        self.data_dir = data_dir
        self.csv_dir = csv_dir

        sets = parse_settings()
        #self.phase = sets.phase
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.input_x = sets.input_x
        self.input_y = sets.input_y
        self.input_z = sets.input_z

        # from csv dir, get path of #each image and store it in a dataframe
        # use patient id as key and image path as value in a dict
        age_imaging_df = pd.read_csv(csv_dir)
        #print(age_imaging_df.head())
        #patient_id_df = age_imaging_df['eid']

        #for i in range(len(patient_id_df)):
         #   image_path = os.path.join(data_dir, str(patient_id_df[i]), 'wat.nii.gz')
          #  age_imaging_df.loc[i, 'image_path'] = image_path

        self.age_imaging_df = age_imaging_df

        #self.targets_list = self.age_imaging_df[["sex"]].values.tolist()
        # age values are in float
        self.targets_list = self.age_imaging_df[["age"]].values.tolist()




    def __len__(self):
        return len(self.age_imaging_df)

    def __getitem__(self, idx):
        # read image and labels
        image_name = self.age_imaging_df.loc[idx, 'image_path']
        #print(self.age_imaging_df.head().columns)
        #mistake is here. take only one target value per image
        target = self.targets_list[idx]
        #print("target value: ", target)
        #print("type of target: ", type(target))
        target_tensor = torch.tensor(target)
        target_tensor = target_tensor.type(torch.float32)
        #print("target tensor shape: ", target_tensor.shape)

        # check if image file exists
        assert os.path.isfile(image_name)

        #load image
        image = nib.load(image_name)

        assert image is not None

        image_data = image.get_fdata()

        # call all data pre-processing functions here
        # 1. convert nii to tensor array
        #image_tensor = self.transforms(image_data)
        image_tensor = self.transform(image_data)

        image_tensor = self.convert_nii_to_tensor(image_data)

        #image_tensor = self.transform(image_tensor)


        #print(type(image_tensor))

        image_tensor_normalized = self.intensity_normalization(image_tensor)


        return image_tensor_normalized, target_tensor


    def convert_nii_to_tensor(self, image):
        [z,y,x] = image.shape
        image_tensor = np.reshape(image, [1,z,y,x])

        #print("type of image tensor: ", type(image_tensor))
        #image_tensor = image.astype("float32")
        image_tensor = image_tensor.astype("float32")


        return image_tensor
    

    # https://github.com/Tencent/MedicalNet
    def intensity_normalization(self, image):
        pixels = image[image > 0]
        mean = pixels.mean()
        std  = pixels.std()
        out = (image - mean)/std
        out_random = np.random.normal(0, 1, size = image.shape)
        out[image == 0] = out_random[image == 0]
        return out




