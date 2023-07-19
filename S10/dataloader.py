# Import necessary packages.
import torch
import torchvision
import torchvision.transforms as transforms
import albumentations as A
import numpy as np


class CoarseDropout(A.CoarseDropout):
    def get_params_dependent_on_targets(self, params):
        if np.any(params['image'].shape[:2] < np.array([self.max_height, self.max_width])):
            return {'holes': []}
        return super().get_params_dependent_on_targets(params)


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
    transform = A.Compose([
        A.HorizontalFlip(p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=tuple(np.array([(0.49139968, 0.48215827 ,0.44653124)])), mask_fill_value = None),
    ])

    # Load CIFAR-10 dataset
    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505, 0.26158768)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505, 0.26158768)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


    # Create transformed datasets
    transformed_trainset = AlbumentationsDataset(trainset, transform)
    # transformed_testset = AlbumentationsDataset(testset)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(transformed_trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    return train_loader, test_loader

