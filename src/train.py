import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import Net
from config import *
from utils import get_data_loaders

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'Epoch {epoch}: Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return test_loss, accuracy

def main():
    torch.manual_seed(RANDOM_SEED)
    
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    # Calculate dataset sizes
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    
    model = Net().to(DEVICE)
    total_params = count_parameters(model)
    print(f"\nTotal Model Parameters: {total_params:,}")
    print("\nDataset Split:")
    print(f"Training samples: {train_size:,}")
    print(f"Validation/Test samples: {test_size:,}")
    print(f"Split ratio: {train_size}/{test_size}")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_accuracy = 0.0
    train_losses = []
    test_losses = []
    accuracies = []
    target_accuracy = 99.4
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, DEVICE, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            
        scheduler.step(accuracy)
        
        # Early stopping if accuracy reaches target
        if accuracy >= target_accuracy:
            print(f"\nReached target accuracy of {target_accuracy}% at epoch {epoch}")
            break
    
    print("\nTraining Complete!")
    print("=" * 50)
    print(f"Dataset Split Summary:")
    print(f"Training Set: {train_size:,} samples")
    print(f"Validation/Test Set: {test_size:,} samples")
    print(f"Split Ratio: {train_size}/{test_size}")
    print("-" * 50)
    print(f"Total Model Parameters: {total_params:,}")
    print(f"Best Validation/Test Accuracy: {best_accuracy:.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation/Test Loss: {test_losses[-1]:.4f}")
    print(f"Training stopped at epoch: {len(accuracies)}/{EPOCHS}")
    print("=" * 50)

if __name__ == '__main__':
    main() 