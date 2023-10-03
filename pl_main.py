import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch_lr_finder import LRFinder
from models.model import *
from pytorch_lightning.loggers import TensorBoardLogger
from models.resnet import *

def model_init(BATCH_SIZE,best_lr,epochs):
    seed_everything(42)
    model = LitModel(BATCH_SIZE,best_lr,epochs)
    return model

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def max_LR_finder(datamodule,criterion=nn.CrossEntropyLoss()):
    device = get_device()
    model = ResNet18().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.03,
        momentum=0.9,
        weight_decay=1e-4,
    )
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(datamodule, end_lr=100, num_iter=100)
    best_lr = lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state
    return best_lr

def train_model(model,datamodule,epochs=30):
    # Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir="logs/", name="model")
    # Initialize the Lightning Trainer
    trainer = Trainer(precision = 16,max_epochs=epochs,accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)])

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule=datamodule)
    torch.save(model.state_dict(), 'saved_model_lightning.pth')
    return trainer