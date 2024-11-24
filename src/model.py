import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # reduced to 8 channels
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second Block
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # reduced to 16 channels
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.15)
        
        # Third Block
        self.conv3 = nn.Conv2d(16, 20, 3, padding=1)  # reduced to 20 channels
        self.bn3 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.15)
        
        # Fourth Block
        self.conv4 = nn.Conv2d(20, 24, 3, padding=1)  # reduced to 24 channels
        self.bn4 = nn.BatchNorm2d(24)
        self.dropout3 = nn.Dropout(0.15)
        
        # Fifth Block
        self.conv5 = nn.Conv2d(24, 32, 3, padding=1)  # reduced to 32 channels
        self.bn5 = nn.BatchNorm2d(32)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC Layer
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # First Block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Third Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(self.pool2(x))
        
        # Fourth Block
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout3(x)
        
        # Fifth Block
        x = F.relu(self.bn5(self.conv5(x)))
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(-1, 32)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 