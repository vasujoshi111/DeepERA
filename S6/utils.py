# Import necessary packages.
import torch
from torchvision import datasets, transforms


def get_data_loader(batch_size):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1), # Crop center of image with patch size of 22
        transforms.Resize((28, 28)), # Resize the image
        transforms.RandomRotation((-15., 15.), fill=0), # Rotate teh image from -15 to 15 degrees randomly.
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,)), # Apply mean 0.1307 and std of 0.3081 to data
        ])
    
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,)) # Apply mean 0.1307 and std of 0.3081 to data which are same from train data.
        ])
    
    # Download the train data and apply train transforms
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    
    # Download the test data and apply test transforms
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    # Apply all the keyword argumenets like batching the dataset, shuffling the data etc.
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader, test_loader


def GetCorrectPredCount(pPrediction, pLabels):
    """Function to get the correct prediction count.

    Args:
        pPrediction (Object): Predicted tensors
        pLabels (Object): Actual labels of the images.

    Returns:
        Object(Tensor): If the predicted lables are actual labels then those items will be counted and returned.
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
