import os

import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.core.display import display
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import albumentations as A
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


means = [0.4914, 0.4822, 0.4465]
stds = [0.2470, 0.2435, 0.2616]

import albumentations as A

# Create custom dataset with Albumentations transformations
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)  # Convert image to ndarray
        transformed = self.transform(image=image)
        transformed_image = transformed['image']
        return transformed_image, label

    def __len__(self):
        return len(self.dataset)

class MyDataModule(LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self):
        # Download dataset if needed
        CIFAR10('.', train=True, download=True)

    def setup(self, stage=None):
        # Define train and validation datasets
        transform_train = A.Compose([
            A.Normalize(mean=means, std=stds, always_apply=True),
            A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.HorizontalFlip(always_apply=True, p = 1),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), always_apply=True, p=0.5),
            ToTensorV2(),
        ])
        transform_val = A.Compose([A.Normalize(mean=means, std=stds, always_apply=True),
        ToTensorV2(),])


        train_dataset = CIFAR10('.', train=True, download=True)
        val_dataset = CIFAR10('.', train=False, download=True)

        self.train_dataset = AlbumentationsDataset(train_dataset, transform_train)
        self.val_dataset = AlbumentationsDataset(val_dataset, transform_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)



def train(cifar10_dm, new_lr):
    model = LitResnet(lr = new_lr)

    trainer = Trainer(
        max_epochs=24,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )


    trainer.fit(model, cifar10_dm)
    trainer.test(model, cifar10_dm)
