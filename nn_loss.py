import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

loss = L1Loss(reduction='sum')
result = loss(inputs, targets)
print(result)

loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
result_mse2 = loss_mse(targets, inputs)
print(result_mse2)
print(result_mse2)


x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)