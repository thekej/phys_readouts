import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Multi-layer Perceptron Model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=1):
        super().__init__()
        input_dim = 1
        for dim in input_size[1:]:
            input_dim *= dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.sigmoid(x)
        return x
