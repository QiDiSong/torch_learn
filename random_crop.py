from torchvision import transforms
import torch
from torch import nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

trans_random = transforms.RandomCrop(50)
trans_totensor = transforms.ToTensor()
trans_compose = transforms.Compose([trans_random, trans_totensor])

for i in range(10):
    img_crop = trans_compose(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
