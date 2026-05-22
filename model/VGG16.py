import torch
import torch.nn as nn
import torch.nn.functional as F
class VGG16(nn.Module):
    def __init__(self, num_classes=5):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
