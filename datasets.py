import numpy as np
import nibabel as nib
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from settings import parse_settings
from ukbb import UKBB
from settings import parse_settings
import math
from torch.utils.data.sampler import SubsetRandomSampler

class UKBBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str='/home/guests/projects/ukbb/abdominal/nifti',
                 target_dir: str='/home/guests/projects/ukbb/yadu/age_imaging_filtered_train_val.csv'):
        super().__init__()
        self.data_dir = data_dir

        # length of train val csv is 35122, length of test csv is 8712
        self.target_dir = target_dir
        #self.batch_size = batch_size
        self.sets = parse_settings()
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.0,), std=(1,))])
    

    def setup(self, stage: str = None) -> None:
        # assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.sets.phase == 'train':
                # len of train val is 35122
                ukbb_full = UKBB(self.data_dir, self.target_dir)
                #print("type of ukbb full")
                #print(type(ukbb_full))
                ukbb_dataset_size = len(ukbb_full)


                print("len ukbb full")
                print(ukbb_dataset_size)
                ################################## working code block ##########################
                #train_val_size = int(35122)
                #test_size = int(7000)
                train_size = int(30000)
                val_size = int(5122)


                self.ukbb_train, self.ukbb_val = random_split(ukbb_full, [train_size, val_size])

                ######################## end of working code block##############################

                ###################### begin code for training check ###########################
                # code for training check

                #check_dataset_size = int(5122)
                #check_train_size = int(4000)
                check_train_size = int(1122)
                check_val_size = int(4000)

                #self.check_train, self.check_val = random_split(self.ukbb_val, [check_train_size, check_val_size])
                # do random split twice. in one get thr training set. in the other, get the validation set. 
                

                self.check_train, self.check_val = random_split(self.ukbb_val, [check_train_size, check_val_size])
                

                #self.check_train, self.check_val = random_split(check_train_val, [5000, 1000])
                ##############################################################################################
                #check_subset_size = int(1000)
                #self.subset = self.check_test

                self.subset_train, self.subset_val = random_split(self.check_val, [3200, 800])
                #self.subset_train, self.subset_val = random_split(subset_train_val, [800, 100])



    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.subset_train, batch_size=self.sets.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.subset_val, batch_size=self.sets.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.subset_test, batch_size=self.sets.batch_size, shuffle=False, num_workers=4)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ukbb_full, batch_size=self.sets.batch_Size)