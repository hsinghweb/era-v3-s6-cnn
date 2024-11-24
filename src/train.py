import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import Net
from config import *
from utils import get_data_loaders

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_description(f'Epoch {epoch} Loss: {loss.item():.4f}')
    return running_loss / len(train_loader)

def test(model, device, test_loader):
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
    
    print(f'Test set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy

def main():
    torch.manual_seed(RANDOM_SEED)
    
    train_loader, test_loader = get_data_loaders(BATCH_SIZE)
    
    model = Net().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_accuracy = 0.0
    train_losses = []
    test_losses = []
    accuracies = []
    
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, DEVICE, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, DEVICE, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            
        scheduler.step(accuracy)
    
    print("\nTraining Complete!")
    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    print(f"Average Training Loss: {sum(train_losses)/len(train_losses):.4f}")
    print(f"Average Test Loss: {sum(test_losses)/len(test_losses):.4f}")
    print(f"Average Test Accuracy: {sum(accuracies)/len(accuracies):.2f}%")

if __name__ == '__main__':
    main() 