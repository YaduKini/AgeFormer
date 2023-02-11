import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
import torch
from settings import parse_settings
from torchvision.models.video import R3D_18_Weights
from torchmetrics import Accuracy, AUROC
from pytorch_lightning.loggers.wandb import WandbLogger
from torchmetrics.classification import BinaryAccuracy


class Classifier(pl.LightningModule):
    def __init__(self, sets):
        super(Classifier, self).__init__()
        #resnet3d = models.video.r3d_18(pretrained=True)
        #print(resnet3d.parameters)
        self.save_hyperparameters()
        self.sets = parse_settings()
        self.num_classes = sets.num_classes
        self.accuracy = Accuracy(task='binary')
        self.accuracy_fn = BinaryAccuracy()
        resnet3d_model = models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)
        resnet3d_model.stem = nn.Sequential(nn.Conv3d(1, 3, 1), resnet3d_model.stem) # or duplicate the input 3 times
        linear_layer_size = list(resnet3d_model.children())[-1].in_features
        #num_filters = resnet3d_model.fc.in_features
        #layers = list(resnet3d_model.children())[:-1]
        #self.resnet3d_model = nn.Sequential(*layers)
        resnet3d_model.fc = nn.Linear(linear_layer_size, 256)
        print("linear_layer_size: ", linear_layer_size)
        self.resnet3d_model = resnet3d_model


        freeze = False
        if freeze:
            self.resnet3d_model.eval()
            for layer in list(self.resnet3d_model.children())[:-1]:
                for neuron in layer.parameters():
                    neuron.requires_grad = False

        #self.resnet_classifier1 = nn.Linear(1024, linear_layer_size)
        self.resnet_classifier = nn.Linear(256, self.num_classes)        
        #count = 0
        #self.resnet3d_model.eval()
        #for params in self.resnet3d_model.parameters():
         #   params.requires_grad=False

        # for layer in list(self.resnet3d_model.children())[:-1]:
        #      print("layer children: ", layer.children)
        #      for neuron in layer.parameters():
        #         neuron.requires_grad=False

        #print("linear layer size: ", linear_layer_size)
        #self.l1 = nn.Linear(224*168*363, 2)
        # linear layer size is 512
        #self.resnet3d_model.fc = nn.Linear(linear_layer_size, self.num_classes)
        #self.resnet3d_model.fc = nn.Linear(linear_layer_size, 256)


        #self.resnet3d_model = resnet3d_model
        #for layer in list(self.resnet3d_model.children())[:-1]:
         #    print("layer children: ", layer.children)
          #   for neuron in layer.parameters():
           #      neuron.requires_grad=False
                #count += 1
                #print(count)
                #print(neuron.requires_grad)

        #self.resnet3d_model = resnet3d_model


        #for layer in list(self.resnet3d_model.children())[:-1]:
         #    print("layer children: ", layer.children)
          #   for neuron in layer.parameters():
           #     neuron.requires_grad=False
        
        
    def forward(self, x) -> torch.Tensor:

        #print(x.size())
        result = self.resnet3d_model(x)
        result = self.resnet_classifier(result)
        return result
        #return torch.relu(self.resnet3d_model(x))
        #return torch.relu(self.resnet3d_model(x.view(x.size(0), -1)))
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        #y_hat = self(x) # y_hat is the logits
        y_hat = self.forward(x)
        #y_hat = y_hat.unsqueeze(-1)
        # correct loss func
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        
        self.log("training loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        # accuracy function is wrong?
        training_accuracy = self.accuracy(y_hat, y)     
        self.log("training accuracy", training_accuracy, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.sets.learning_rate)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        #y = y.unsqueeze(-1)
        #y_hat = self(x) # y_hat is the logits
        y_hat = self.forward(x)
        #y_hat = y_hat.unsqueeze(-1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("validation loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        validation_accuracy = self.accuracy(y_hat, y)
        self.log("validation accuracy", validation_accuracy, on_epoch=True, prog_bar=True, logger=True, on_step=False)
        return loss
    

    def test_step(self, batch, batch_idx):
        x, y = batch
        #y_hat = self(x)
        y_hat = self.forward(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        self.log("test loss", loss, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        test_accuracy = self.accuracy(y_hat, y)
        self.log("test accuracy", test_accuracy, on_epoch=True, prog_bar=True, logger=True, on_step=True)
        return loss