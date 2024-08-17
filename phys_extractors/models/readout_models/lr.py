import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Logistic Regression Model
class BinaryLogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_dim = 1
        for dim in input_size[1:]:
            self.input_dim *= dim
        self.fc = nn.Linear(self.input_dim, 1)
        nn.init.uniform_(self.fc.weight, -0.01, 0.01) 
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
