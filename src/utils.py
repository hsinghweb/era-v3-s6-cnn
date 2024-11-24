import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

def get_data_loaders(batch_size):
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load full training dataset
    full_train_dataset = datasets.MNIST('../data', train=True, download=True,
                                      transform=train_transform)
    
    # Split training dataset into train and validation
    train_size = 50000  # 50k for training
    val_size = 10000    # 10k for validation/test
    
    train_dataset, _ = random_split(full_train_dataset, 
                                  [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))
    
    # Use the test set as our validation/test set
    val_dataset = datasets.MNIST('../data', train=False,
                               transform=test_transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    
    return train_loader, val_loader 