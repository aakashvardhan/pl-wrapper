import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
import os
from models.resnet import *
from torchsummary import summary
from tqdm import tqdm
# from data_transform import *
from torch.optim.lr_scheduler import OneCycleLR
device = get_device()
class LitModel(pl.LightningModule):
    def __init__(self,BATCH_SIZE,best_lr,epochs=5, num_classes=10):
        super().__init__()
        self.model = ResNet18()
        self.num_classes = num_classes
        # self.lr = lr
        self.BATCH_SIZE = BATCH_SIZE
        self.best_lr = best_lr
        self.epochs = epochs

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1, keepdim=True)
        acc = pred.eq(y.view_as(pred)).float().mean()
        if stage == 'val':
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        else:
            self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')

    def configure_optimizers(self):
        LEARNING_RATE=0.03
        optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=1e-4,
        )

        self.trainer.fit_loop.setup_data()
        dataloader = self.trainer.train_dataloader

        lr_scheduler = OneCycleLR(
          optimizer,
          max_lr=self.best_lr,
          steps_per_epoch=len(dataloader),
          epochs=self.epochs,
          pct_start=5/24,
          div_factor=100,
          three_phase=False,
          final_div_factor=100,
          anneal_strategy='linear'
        )

        scheduler = {"scheduler": lr_scheduler, "interval" : "step"}

        return [optimizer], [scheduler]

    def on_train_start(self):
        model_summary(self.model, input_size=(3, 32, 32))  # Assuming input size to be (3, 32, 32)
