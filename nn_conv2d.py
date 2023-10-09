import torch
from torch import nn
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)


class ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        out = self.conv2d(x)
        return out

convmodel = ConvModel()
writer = SummaryWriter("Conv")
step = 0
for data in dataloader:
    imgs, targets = data
    output = convmodel(imgs)
    writer.add_images("input", imgs, step)
    output = torch.reshape(output, [-1, 3, 30, 30])
    writer.add_images("output", output, step)
    step += 1