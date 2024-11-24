import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # Reduced to 8 channels
        self.bn1 = nn.BatchNorm2d(8)
        
        # Second Block
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # Reduced to 16 channels
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.1)  # Increased dropout
        
        # Third Block
        self.conv3 = nn.Conv2d(16, 20, 3, padding=1)  # Reduced to 20 channels
        self.bn3 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.1)  # Increased dropout
        
        # Fourth Block with parallel paths
        self.conv4_1x1 = nn.Conv2d(20, 20, 1)  # Reduced to 20 channels
        self.conv4_main = nn.Conv2d(20, 20, 3, padding=1)  # Reduced to 20 channels
        self.bn4 = nn.BatchNorm2d(20)
        
        # Fifth Block
        self.conv5 = nn.Conv2d(20, 20, 3, padding=1)  # Reduced to 20 channels
        self.bn5 = nn.BatchNorm2d(20)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC Layer
        self.fc = nn.Linear(20, 10)

    def forward(self, x):
        # First Block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Third Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Fourth Block with skip connection
        x_main = F.relu(self.bn4(self.conv4_main(x)))
        x_1x1 = self.conv4_1x1(x)
        x = x_main + F.relu(x_1x1)
        
        # Fifth Block
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Global Average Pooling and Final FC
        x = self.gap(x)
        x = x.view(-1, 20)
        x = self.fc(x)
        return F.log_softmax(x, dim=1) 