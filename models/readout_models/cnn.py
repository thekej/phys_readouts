import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

#Convolutional Neural Network Model
class CNN(nn.Module):
    def __init__(self, input_shape, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[1], 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
