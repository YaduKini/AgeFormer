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
    def __init__(self, data_dir: str='/vol/aimspace/projects/ukbb/abdominal/nifti',
                 train_csv: str='/vol/aimspace/users/kini/yadu/sex_prediction/tabular_data_train_aimspace.csv',
                 val_csv: str='/vol/aimspace/users/kini/yadu/sex_prediction/tabular_data_val_aimspace.csv',
                 test_csv: str='/vol/aimspace/users/kini/yadu/sex_prediction/tabular_data_test_aimspace.csv'):
        super().__init__()
        self.data_dir = data_dir

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.sets = parse_settings()
    

    def setup(self, stage: str = None) -> None:
        # assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.sets.phase == 'train':

                self.ukbb_train = UKBB(self.data_dir, self.train_csv)
                self.ukbb_val = UKBB(self.data_dir, self.val_csv)

                # code below is for sanity check using one data sample

                #train_size = int(2)

                #ukbb_train_dataset_size = len(ukbb_train)
                #print("length of ukbb train dataset size: ", ukbb_train_dataset_size)

                #ukbb_val_dataset_size = len(ukbb_val)
                #print("length of ukbb val dataset size: ", ukbb_val_dataset_size)

                #remainder_train_size = ukbb_train_dataset_size - train_size

                #val_size = int(1)
                #remainder_val_size = ukbb_val_dataset_size - val_size

                #self.ukbb_train, _ = random_split(ukbb_train, [train_size, remainder_train_size])
                #self.ukbb_val, _ = random_split(ukbb_val, [val_size, remainder_val_size])


        if stage == "inference":
            if self.sets.phase == "test":
                self.ukbb_test = UKBB(self.data_dir, self.test_csv)



    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.ukbb_train, batch_size=self.sets.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ukbb_val, batch_size=self.sets.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ukbb_test, batch_size=self.sets.batch_size, shuffle=True, num_workers=8)