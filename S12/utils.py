# Import necessary packages.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.resnet import ResNet18
from torch.optim.lr_scheduler import OneCycleLR
from torch_lr_finder import LRFinder
from gradcam.utils import visualize_cam
from gradcam import GradCAM
import matplotlib.pyplot as plt
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


classes_dict = {
    0: "Airplane", 1: "Automobile", 2: "Bird", 3: "Cat", 4:"Deer", 5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9:"Truck"
}


def get_train_parameters():
    """Function to get the training cretarians.

    Returns:
        Objects: All training parameters.
    """
    use_cuda = torch.cuda.is_available() # Boolean to get whether cuda is there in device or not.
    device = torch.device("cuda" if use_cuda else "cpu") # If cuda is present set the device to cuda, otherwise set to cpu.
    model = ResNet18().to(device) # Set the model to same device.
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4) # Initialize the optimizer to Adam with learning rate 0.001.
    
    criterion = nn.CrossEntropyLoss() # Define the entropy loss function
    
    return model, device, optimizer, criterion

def get_schedular(optimizer, train_loader, epochs, max_lr):
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=5/epochs,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
    return scheduler

def GetCorrectPredCount(pPrediction, pLabels):
    """Function to get the correct prediction count.

    Args:
        pPrediction (Object): Predicted tensors
        pLabels (Object): Actual labels of the images.

    Returns:
        Object(Tensor): If the predicted lables are actual labels then those items will be counted and returned.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()



def suggest_lr(cifar10_dm):
    model = LitResnet(lr = 0.05)

    trainer = Trainer(
        max_epochs=24,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        logger=CSVLogger(save_dir="logs/"),
        callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
    )
    cifar10_dm.setup("train")
    train_loader = cifar10_dm.train_dataloader()
    lr_finder = trainer.tuner.lr_find(model, train_loader, num_training = 200, max_lr = 10)
    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    return new_lr



def plot_loss_accuracy(trainer):
    """Plot train and test losses and accuracy."""
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    del metrics["step"]
    metrics.set_index("epoch", inplace=True)
    display(metrics.dropna(axis=1, how="all").head())
    sn.relplot(data=metrics, kind="line")


def get_misclassified(model, device, test_loader, n = 10):
    """Get mis clasified outputs and images"""
    model.eval()
    wr = 0
    images = []
    tr = []
    pr = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            for k in range(0, len(data)):
              if pred[k].item()!=target[k].item():
                  images.append(data[k])
                  tr.append(target[k])
                  pr.append(pred[k])
                  wr+=1
                  if wr==10:
                    break
            if wr==10:
                break
    return images, tr, pr


def show_misclassified(img, trs, prs):
    """Plot misclassified images."""
    f, axarr = plt.subplots(2, 5, figsize=(15, 15))
    for i in range(0, 5):
        axarr[0,i].imshow(img[i].cpu().T)
        axarr[0,i].set_title(classes_dict[int(trs[i].item())] + "/"+classes_dict[prs[i].item()])
        axarr[1,i].imshow(img[i+5].cpu().T)
        axarr[1,i].set_title(classes_dict[int(trs[i+5].item())] + "/"+classes_dict[prs[i+5].item()])
