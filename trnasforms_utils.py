from torchvision import transforms
from PIL import Image

image_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(image_path)
print(img.size)
transforms_totensor = transforms.ToTensor()
img_totensor = transforms_totensor(img)
print(img_totensor.shape)


transforms.Resize((2,2))