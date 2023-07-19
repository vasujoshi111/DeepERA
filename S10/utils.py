# Import necessary packages.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Net
from torch.optim.lr_scheduler import OneCycleLR


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
    model = Net().to(device) # Set the model to same device.
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
        # verbose = True
        # div_factor=100,
        # three_phase=False,
        # final_div_factor=100,
        # anneal_strategy='linear'
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
