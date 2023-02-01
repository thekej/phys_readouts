import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
    
# Attention-based classifier model
class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionModel, self).__init__()
        input_dim = 1
        for dim in input_size[1:]:
            input_dim *= dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc4 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        attention = torch.softmax(self.fc3(x), dim=1)
        x = torch.sum(attention * x, dim=1)
        x = self.fc4(x)
        return x
    

class BinaryAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_heads=8):
        super(BinaryAttentionModel, self).__init__()
        input_dim = 1
        for dim in input_size[1:]:
            input_dim *= dim
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x, _ = self.attention(x, x, x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


