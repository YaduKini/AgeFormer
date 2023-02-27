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
                 train_csv: str='/home/guests/projects/ukbb/yadu/tabular_data_train.csv',
                 val_csv: str='/home/guests/projects/ukbb/yadu/tabular_data_val.csv'):
        super().__init__()
        self.data_dir = data_dir

        # length of train val csv is 35122, length of test csv is 8712
        #self.target_dir = target_dir
        self.train_csv = train_csv
        self.val_csv = val_csv
        #self.batch_size = batch_size
        self.sets = parse_settings()
        #self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.0,), std=(1,))])
    

    def setup(self, stage: str = None) -> None:
        # assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            if self.sets.phase == 'train':
                # len of train val is 35122
                # len of train is 28125
                # len of val is 6997

                self.ukbb_train = UKBB(self.data_dir, self.train_csv)
                self.ukbb_val = UKBB(self.data_dir, self.val_csv)
                #ukbb_full = UKBB(self.data_dir, self.target_dir)
                #print("type of ukbb full")
                #print(type(ukbb_full))


                #ukbb_train_dataset_size = len(ukbb_train)
                #ukbb_val_dataset_size = len(ukbb_val)
                #print("len ukbb_train")
                #print(ukbb_train_dataset_size)
                #print("len ukbb_val")
                #print(ukbb_val_dataset_size)

                #train_val_size = int(35122)
                #test_size = int(7000)

                # change the value here to add more training data
                #train_size = int(1536)
                #remainder_train_size = ukbb_train_dataset_size - train_size

                # change the value here to add more val data
                #val_size = int(384)
                #remainder_val_size = ukbb_val_dataset_size - val_size

                ############## get training and val data set ################################################

                #self.ukbb_train, _ = random_split(ukbb_train, [train_size, remainder_train_size])

                #self.ukbb_val, _ = random_split(ukbb_val, [val_size, remainder_val_size])

                #self.subset_train, self.subset_val = random_split(self.check_val, [160, 40])
                #self.subset_train, self.subset_val = random_split(subset_train_val, [800, 100])



    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.ukbb_train, batch_size=self.sets.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.ukbb_val, batch_size=self.sets.batch_size, shuffle=False, num_workers=4)

    # def test_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.subset_test, batch_size=self.sets.batch_size, shuffle=False, num_workers=4)

    # def predict_dataloader(self) -> EVAL_DATALOADERS:
    #     return DataLoader(self.ukbb_full, batch_size=self.sets.batch_Size)