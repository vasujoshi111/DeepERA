# Import necessary packages.
import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount
from tqdm import tqdm

# Globally define losses and accuracy
# Data to plot accuracy and loss graphs
test_losses = []
test_acc = []
train_losses = []
train_acc = []
lr_lists = []

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
    
    
def get_data_loader(batch_size):
    
    # Define the transformations
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2470, 0.2435, 0.2616]

    transform_train = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            A.PadIfNeeded(min_height=36, min_width=36, always_apply=True),
            A.RandomCrop(height=32, width=32, always_apply=True),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=means, always_apply=True, p=0.5),
            ToTensorV2(),
        ]
    )

    transform_test = A.Compose(
        [
            A.Normalize(mean=means, std=stds, always_apply=True),
            ToTensorV2(),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Create transformed datasets
    transformed_trainset = AlbumentationsDataset(trainset, transform_train)
    transformed_testset = AlbumentationsDataset(testset, transform_test)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(transformed_trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(transformed_testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


def train(model, device, train_loader, optimizer, criterion, epoch, scheduler):
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
        scheduler.step() #  Onle cycle policy
        
        correct += GetCorrectPredCount(pred, target) # Get the correct prediction count.
        processed += len(data)
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_description(desc= f'Epoch: {epoch} Train: LR = {current_lr} Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        lr_lists.append(current_lr)

    # Append the accuracy and losses to lists
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))



def test(model, device, test_loader, criterion):
    """Function to train the model.

    Args:
        model (Object): Torch.nn module network
        device (Object): Device name(CPU/Cuda)
        train_loader (Object): Torch train loader object
        optimizer (Object): Torch optimizer function
        criterion (Object): Torch Loss function
    """
    model.eval() # Set the model to test mode.

    test_loss = 0
    correct = 0

    # As in test mode, no gradients are required to calculate.
    with torch.no_grad():
        # Enumerate through each data.
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) # Load the data and change the device to cpu/cuda.

            output = model(data) # Pass the data to the model.
            test_loss += criterion(output, target, reduction = "sum").item() # Get the loss

            correct += GetCorrectPredCount(output, target) # Get the correct prediction count.

    # Append the accuracy and losses to lists
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def train_validate(model, device, train_loader, optimizer, criterion, epochs, scheduler, test_loader):
    for epoch in range(epochs):
        train(model, device, train_loader, optimizer, criterion, epoch, scheduler)
        test(model, device, test_loader, criterion)
    