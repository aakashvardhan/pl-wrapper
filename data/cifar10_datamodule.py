from data.data_transform import *
import albumentations as A
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import os

PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
NUM_WORKERS = int(os.cpu_count() / 2)
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = PATH_DATASETS

    ):
        super().__init__()
        self.dims = (3, 32, 32)
        self.save_hyperparameters(logger=False)

        self.train_transforms = transform(query="train")
        self.test_transforms = transform(query="test")

        self.data_train = None
        self.data_val = None


    @property
    def num_classes(self):
        return 10
    
    @property
    def class_names(self):
        return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    
    def prepare_data(self):
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.data_train = datasets.CIFAR10(self.hparams.data_dir, train=True)

        if stage == 'test' or stage is None:
            self.data_val = datasets.CIFAR10(self.hparams.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(
            PyTorchDataset(root=self.hparams.data_dir, train=True, transform=self.train_transforms),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            PyTorchDataset(root=self.hparams.data_dir, train=False, transform=self.test_transforms),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )