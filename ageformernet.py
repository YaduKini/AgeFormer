from backbones.build import build_backbone
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchmetrics import MeanAbsoluteError

class AgeFormerNet(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self._config = config
        self.save_hyperparameters()

        # get backbone
        self._backbone = build_backbone(config['backbone'])

        # get regression head
        self._reg_head = MLP(363, 128, 1, 3) # for c0

        #self._reg_head = MLP(46, 128, 1, 3) # fpr c3
        #self._reg_head = MLP(23, 128, 1, 3) # for c4
        #self._reg_head = MLP(12, 128, 1, 3) # for c5

        self._flatten = nn.Flatten()
        self.regressor = nn.Linear(903168, 1) # for c0

        #self.regressor = nn.Linear(112896, 1) # for c3
        #self.regressor = nn.Linear(59136, 1) # for c4
        #self.regressor = nn.Linear(32256, 1) # for c5


        self.mae = MeanAbsoluteError()


    def forward(self, x):

        #print("x shape in forward: ", x.shape)
        out_backbone = self._backbone(x)
        c0 = out_backbone['C0']
        #c3 = out_backbone['C3']
        #c4 = out_backbone['C4']
        #c5 = out_backbone['C5']

        #print("c4 shape in forward: ", c4.shape)

        #c4 = c4.view(-1)

        
        #print("c4 shape after view: ", c4.shape)
        reg_out = self._reg_head(c0)

        #reg_out = self._reg_head(c3)

        #reg_out = self._reg_head(c4)
        #reg_out = self._reg_head(c5)

        flatten_out = self._flatten(reg_out)
        #print("shape of flattened output: ", flatten_out.shape)

        output = self.regressor(flatten_out)

        return output
    

    def training_step(self, batch, batch_idx):
        x, y = batch

        #print(x.shape)
        #print(y.shape)
        #print("y: ", y)

        y_hat = self.forward(x)

        #print("y hat shape: ", y_hat.shape)
        #print("y_hat: ", y_hat)

        error = self.mae(y_hat, y)

        self.log("training error", error, on_epoch=True, prog_bar=True, logger=True, on_step=False)

        return error

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self.forward(x)

        #print("y hat shape: ", y_hat.shape)
        #print("y_hat: ", y_hat)

        error = self.mae(y_hat, y)

        self.log("validation error", error, on_epoch=True, prog_bar=True, logger=True, on_step=False)

        return error


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self._config['backbone']['lr'])



class MLP(pl.LightningModule):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        # hidden dim = 128, h = 128 * 2 = 256
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


