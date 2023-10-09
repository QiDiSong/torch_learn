import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)


simple_model = SimpleModel()
data_x = torch.tensor([[1.0], [2.0], [3.0]])
data_y = torch.tensor([[2.0], [4.0], [6.0]])

criterion = nn.MSELoss()
optimizer = optim.SGD(simple_model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = simple_model(data_x)
    # print(y_pred.dtype)
    loss = criterion(y_pred, data_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar("Loss", loss.item(), epoch)

writer.close()
print('hello')

# 1. python tensor_board.py
# 2. tensorboard --logdir=logs
# 3. click the http://localhost:6006/ button.
