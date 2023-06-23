# Import necessary packages.
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils import GetCorrectPredCount
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        #Block 1: To get 5*5 receptive feild, as most of the images edegs cover with the 5 pixels.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.025)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Dropout(0.025)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 10, 3),
            nn.BatchNorm2d(10),
            nn.Dropout(0.025)
        )     
        self.gap = nn.AvgPool2d(2) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gap(x)
        x = x.flatten().view(-1, 10)
        x = F.log_softmax(x, dim=1)
        return x


def get_train_parameters():
    """Function to get the training cretarians.

    Returns:
        Objects: All training parameters.
    """
    use_cuda = torch.cuda.is_available() # Boolean to get whether cuda is there in device or not.
    device = torch.device("cuda" if use_cuda else "cpu") # If cuda is present set the device to cuda, otherwise set to cpu.
    model = Net().to(device) # Set the model to same device.
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) # Initialize the optimizer to SGD with learning rate 0.01 and momentum 0.9.
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True) # Initialize the scheduler learning rate.
    # New Line
    criterion = F.nll_loss # Define the entropy loss function
    
    return model, device, optimizer, scheduler, criterion


# Globally define losses and accuracy
# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

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