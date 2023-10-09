import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input_data = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])
input_data = torch.reshape(input_data, (-1, 1, 5, 5))
print(input_data.shape)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        output = self.maxpool(x)
        return output


model = Model()
output = model(input_data)
print(output)


writer = SummaryWriter("logs_maxpool")
dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
print('hello')