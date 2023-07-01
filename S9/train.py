# Import necessary packages.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount
from tqdm import tqdm

# Globally define losses and accuracy
# Data to plot accuracy and loss graphs
train_losses = []
train_acc = []

def train(model, device, train_loader, optimizer, criterion):
    """Function to train the model.

    Args:
        model (Object): Torch.nn module network
        device (Object): Device name(CPU/Cuda)
        train_loader (Object): Torch train loader object
        optimizer (Object): Torch optimizer function
        criterion (Object): Torch Loss function
    """
    model.train() # Set the model to train mode.
    pbar = tqdm(train_loader) # Set one bar for visualization

    train_loss = 0 # Initialize train_loss to 0.
    correct = 0 # Initialize Correctly predicted labels to 0.
    processed = 0 # Initialize processed images to 0.

    # Enumerate through each data.
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) # Load the data and change the device to cpu/cuda.
        optimizer.zero_grad() # Set all gradients to zero initially

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item()

        # Backpropagation
        loss.backward() # Calculate gradient.
        optimizer.step() # Update weights.

        correct += GetCorrectPredCount(pred, target) # Get the correct prediction count.
        processed += len(data)

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Append the accuracy and losses to lists
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))
