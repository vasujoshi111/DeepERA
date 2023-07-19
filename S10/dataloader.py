# Import necessary packages.
import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

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
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
            A.RandomCrop(height=32, width=32, always_apply=True),
            # A.Flip()
            A.HorizontalFlip(always_apply=True, p = 1),
            # A.Resize(32, 32),
            A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=(0.4914, 0.4822, 0.4465), always_apply=True, p=0.5),
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

