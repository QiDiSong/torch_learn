import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * x


data = torch.tensor([1, 2, 3, 4])
model = Model()
print(model(data))
