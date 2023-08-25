from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import albumentations as A
from PIL import Image
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from dataset import YOLODataset, get_train_transform, get_test_transforms
import config
import torch.optim as optim
import os
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer, seed_everything, LightningDataModule
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from model import YOLOv3
from loss import YoloLoss
loss_fn = YoloLoss()
scaler = torch.cuda.amp.GradScaler()

scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to(config.DEVICE)


class MyDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # Download dataset if needed
        pass

    def setup(self, stage=None):
        # Define train and validation datasets

        IMAGE_SIZE = config.IMAGE_SIZE
        self.train_dataset = YOLODataset(
            config.DATASET + "/train.csv",
            transform=get_train_transform,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            stage="train"
        )
        self.val_dataset = YOLODataset(
            config.DATASET + "/test.csv",
            transform=get_test_transforms,
            S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
            img_dir=config.IMG_DIR,
            label_dir=config.LABEL_DIR,
            anchors=config.ANCHORS,
            stage = "test"
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=config.BATCH_SIZE, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=config.BATCH_SIZE, num_workers=4)
    

class LitYoloV3(LightningModule):
    def __init__(self, loader = None):
        super().__init__()
        self.validation_step_outputs = []
        self.train_step_outputs = []
        self.validation_step_preds = []
        self.train_step_preds = []
        self.save_hyperparameters()
        self.model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
        self.optimizer = optim.Adam(
    self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
)

    def forward(self, x):
        out = self.model.forward(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        self.train_step_outputs.append(y)
        
        with torch.cuda.amp.autocast():
            out = self.model.forward(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            self.train_step_preds.append(out)
        return loss

    def evaluate(self, batch, stage=None):
        losses = []
        x, y = batch
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )
        self.validation_step_outputs.append(y)
        with torch.cuda.amp.autocast():
            out = self.model.forward(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )
            self.validation_step_preds.append(out)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")
        
    def on_train_epoch_end(self):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        
        for y, out in zip(self.train_step_outputs, self.train_step_preds):
            for i in range(3):
                y[i] = y[i].to(config.DEVICE)
                obj = y[i][..., 0] == 1 # in paper this is Iobj_i
                noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

                correct_class += torch.sum(
                    torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
                )
                tot_class_preds += torch.sum(obj)

                obj_preds = torch.sigmoid(out[i][..., 0]) > config.CONF_THRESHOLD
                correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
                tot_obj += torch.sum(obj)
                correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
                tot_noobj += torch.sum(noobj)

        self.log("Train Class accuracy is: ", torch.round((correct_class/(tot_class_preds+1e-16))*100, decimals=2), prog_bar=True)
        self.log("Train No obj accuracy is: ", torch.round((correct_noobj/(tot_noobj+1e-16))*100, decimals=2),prog_bar=True)
        self.log("Train Obj accuracy is: ", torch.round((correct_obj/(tot_obj+1e-16))*100, decimals=2),prog_bar=True)
        self.train_step_outputs.clear()
        self.train_step_preds.clear()
        
    def on_val_epoch_end(self):
        tot_class_preds, correct_class = 0, 0
        tot_noobj, correct_noobj = 0, 0
        tot_obj, correct_obj = 0, 0
        
        for y, out in zip(self.validation_step_outputs, self.validation_step_preds):
            for i in range(3):
                y[i] = y[i].to(config.DEVICE)
                obj = y[i][..., 0] == 1 # in paper this is Iobj_i
                noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

                correct_class += torch.sum(
                    torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
                )
                tot_class_preds += torch.sum(obj)

                obj_preds = torch.sigmoid(out[i][..., 0]) > config.CONF_THRESHOLD
                correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
                tot_obj += torch.sum(obj)
                correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
                tot_noobj += torch.sum(noobj)

        self.log("Val Class accuracy is: ", torch.round((correct_class/(tot_class_preds+1e-16))*100, decimals=2), prog_bar=True)
        self.log("Val No obj accuracy is: ", torch.round((correct_noobj/(tot_noobj+1e-16))*100, decimals=2), prog_bar=True)
        self.log("Val Obj accuracy is: ", torch.round((correct_obj/(tot_obj+1e-16))*100, decimals=2), prog_bar=True)
        self.validation_step_outputs.clear()
        self.validation_step_preds.clear()

    def configure_optimizers(self):
        steps_per_epoch = 1035
        scheduler_dict = {
            "scheduler": OneCycleLR(
                self.optimizer,
                config.LEARNING_RATE,
                epochs=self.trainer.max_epochs,
                pct_start=5/self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
                div_factor=100,
                three_phase=False,
                final_div_factor=100,
                anneal_strategy='linear'
            ),
            "interval": "step",
        }
        return {"optimizer": self.optimizer, "lr_scheduler": scheduler_dict}