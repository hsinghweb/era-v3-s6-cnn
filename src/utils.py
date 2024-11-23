import torch
from torchvision import datasets, transforms

def get_data_loaders(batch_size):
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

    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                 transform=train_transform)
    test_dataset = datasets.MNIST('../data', train=False,
                                transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)
    
    return train_loader, test_loader 