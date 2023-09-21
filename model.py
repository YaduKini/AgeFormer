import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch
from settings import parse_settings
from torchvision.models.video import R3D_18_Weights
from torchmetrics import Accuracy, MeanAbsoluteError
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics.classification import BinaryAccuracy
import pandas as pd
sets = parse_settings()

class Classifier(pl.LightningModule):
    def __init__(self, sets, age_prediction=True, sex_prediction=False):
        super(Classifier, self).__init__()
        self.save_hyperparameters()
        self.test_predictions = []
        self.test_ground_truth = []
        self.sets = sets
        self.num_classes = sets.num_classes
        self.sex_prediction = sex_prediction
        self.age_prediction = age_prediction
        self.mae = MeanAbsoluteError()
        resnet3d_model = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
        resnet3d_model.stem = nn.Sequential(nn.Conv3d(1, 3, 1), resnet3d_model.stem) # or duplicate the input 3 times
        linear_layer_size = list(resnet3d_model.children())[-1].in_features
        resnet3d_model.fc = nn.Linear(linear_layer_size, 256)

        for params in resnet3d_model.parameters():
            params.requires_grad = True

        self.resnet3d_model = resnet3d_model


        if self.sex_prediction:
            self.resnet_classifier = nn.Linear(256, self.num_classes)

        if self.age_prediction:
            self.resnet_regressor = nn.Linear(256, 1)       

        
    def forward(self, x) -> torch.Tensor:

        result = self.resnet3d_model(x)
        result = self.resnet_regressor(result)
        return result

    

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        error = self.mae(y_hat, y)

        self.log("training error", error, on_epoch=True, prog_bar=True, logger=True, on_step=False)

        return error

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.sets.learning_rate)

        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=3, verbose=True),
            'monitor': 'validation error'
        }

        return [optimizer], [lr_scheduler]

    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        error = self.mae(y_hat, y)
        self.log("validation error", error, on_epoch=True, prog_bar=True, logger=True, on_step=False)

        return error
    

    def on_train_epoch_end(self):


        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.logger.log_hyperparams({'current_lr': current_lr})
        return super().on_train_epoch_end()
    

    def on_validation_epoch_end(self):

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']

        self.logger.log_hyperparams({'current_lr': current_lr})
        return super().on_validation_epoch_end()
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        error = self.mae(y_hat, y)
        self.log("test error", error, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        self.log("predicted age", y_hat, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        

        self.log("ground truth age", y, on_epoch=True, prog_bar=True, logger=True, on_step=True)

        y_hat = y_hat.detach().cpu()
        y = y.detach().cpu()
        print("y: ", y)
        print("y_hat: ", y_hat)

        self.test_predictions.extend(y_hat)
        self.test_ground_truth.extend(y)

        return error


    def on_test_epoch_end(self):
        predictions = self.trainer.callback_metrics['test_predictions']

        with open('test_predictions.txt', 'w') as f:
            for pred in predictions:
                f.write(str(pred) + '\n')


    def on_test_start(self):
        self.test_predictions = []

    def on_test_batch_end(self, outputs, batch, batch_idx):
        x, y = batch
        self.test_predictions.extend(outputs.tolist())
        self.test_ground_truth.extend(y.tolist())

