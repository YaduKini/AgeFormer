from settings import parse_settings
from datasets import UKBBDataModule
from model import Classifier
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import torch
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateFinder, QuantizationAwareTraining, StochasticWeightAveraging, GradientAccumulationScheduler
import time

wandb_logger = WandbLogger()
accumulator = GradientAccumulationScheduler(scheduling={1: 32})
callback_list = [EarlyStopping(patience=3, monitor='validation loss'), accumulator]

seed_everything(42, workers=True)
sets = parse_settings()
ukbb_dm = UKBBDataModule()
ukbb_dm.setup(stage="fit")
train_dl = ukbb_dm.train_dataloader()
val_dl = ukbb_dm.val_dataloader()
model = Classifier(sets=sets)
trainer = pl.Trainer(max_epochs=100, default_root_dir='/home/guests/yadunandan_kini/yadu/sex_prediction/checkpoints',logger=wandb_logger, accelerator='cpu', auto_lr_find=True, callbacks=callback_list)
start_time = time.time()
trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
end_time = time.time()
training_time = (end_time - start_time) / 3600
print("training time: ", training_time)
torch.save(trainer.model.state_dict(),'/home/guests/yadunandan_kini/yadu/sex_prediction/checkpoints/resnet183dfeb8_early_stopping_gradaccum_1000.pth')