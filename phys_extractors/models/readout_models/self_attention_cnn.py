import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F

# Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self, in_features, attention_dim):
        super().__init__()
        self.query = nn.Conv2d(in_features, attention_dim, kernel_size=1)
        self.key = nn.Conv2d(in_features, attention_dim, kernel_size=1)
        self.value = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H*W)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value(x).view(batch_size, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, H, W)
        return out

    
# CNN with Self Attention Module
class CNNWithAttention(nn.Module):
    def __init__(self, input_shape, num_classes=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_shape[1], 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.attention = SelfAttention(32, 16)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = self.attention(x)
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
