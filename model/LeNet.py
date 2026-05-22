import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet 网络定义
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))    # x=[8, 6, 110
        x = self.pool2(torch.relu(self.conv2(x)))   # x=[8, 16, 53, 53]
        x = x.view(-1, 16 * 53 * 53)    # x=[8, 16*53*53]
        x = torch.relu(self.fc1(x)) # x=[8, 120]
        x = torch.relu(self.fc2(x)) # x=[8, 84]
        x = self.fc3(x) # x=[8, 10]
        return x

