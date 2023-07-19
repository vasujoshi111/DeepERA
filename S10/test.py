# Import necessary packages.
import torch
from utils import GetCorrectPredCount

# Globally define losses and accuracy
# Data to plot accuracy and loss graphs
test_losses = []
test_acc = []


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
    