import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First Block
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)  # Changed output to 16
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second Block
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # Changed input to 16 to match previous layer
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.05)
        
        # Third Block
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)  # Changed input to 32 to match previous layer
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.05)
        
        # Fourth Block with parallel paths
        self.conv4_1x1 = nn.Conv2d(32, 32, 1)  # Changed input and output to 32
        self.conv4_main = nn.Conv2d(32, 32, 3, padding=1)  # Changed input to 32
        self.bn4 = nn.BatchNorm2d(32)
        
        # Fifth Block with squeeze-excitation
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)  # Changed input and output to 32
        self.bn5 = nn.BatchNorm2d(32)
        self.se_fc1 = nn.Linear(32, 24)  # squeeze
        self.se_fc2 = nn.Linear(24, 32)  # excitation
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Final FC Layers
        self.fc = nn.Linear(32, 10)  # Changed input to 32

    def forward(self, x):
        # First Block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second Block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))
        
        # Third Block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(self.pool2(x))
        
        # Fourth Block with parallel paths
        x_1x1 = self.conv4_1x1(x)
        x_main = F.relu(self.bn4(self.conv4_main(x)))
        x = x_main + F.relu(x_1x1)  # residual-like connection
        
        # Fifth Block with squeeze-excitation
        x_se = self.gap(x)
        x_se = x_se.view(-1, 32)
        x_se = F.relu(self.se_fc1(x_se))
        x_se = torch.sigmoid(self.se_fc2(x_se))
        x_se = x_se.view(-1, 32, 1, 1)
        x = F.relu(self.bn5(self.conv5(x))) * x_se
        
        # GAP and FC
        x = self.gap(x)
        x = x.view(-1, 32)
        x = F.relu(self.fc(x))
        return F.log_softmax(x, dim=1) 