# Import necessary packages.
import torch
from torchvision import datasets, transforms


def get_data_loader(batch_size):
    kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

    # Train data transformations
    train_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505, 0.26158768)), # Apply mean 0.1307 and std of 0.3081 to data
        ])
    
    # Test data transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(), # Convert image to tensor
        transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505, 0.26158768)) # Apply mean 0.1307 and std of 0.3081 to data which are same from train data.
        ])
    
    # Download the train data and apply train transforms
    train_data = datasets.CIFAR10('../data', train=True, download=True, transform=train_transforms)
    
    # Download the test data and apply test transforms
    test_data = datasets.CIFAR10('../data', train=False, download=True, transform=test_transforms)

    # Apply all the keyword argumenets like batching the dataset, shuffling the data etc.
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader, test_loader

