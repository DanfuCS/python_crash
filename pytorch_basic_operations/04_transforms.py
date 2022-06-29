from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# python的用法 ———> tensor数据类型
#   1. transform如何使用
#   2. tensor数据类型和普通数据类型的区别

img_path = "../dataset/hymenoptera_data/hymenoptera_data/train/ants/7759525_1363d24e88.jpg"
img = Image.open(img_path)
print(img)

writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
print(type(img_tensor))
print(img_tensor.shape)

writer.add_image(tag="image_trans", img_tensor=img_tensor, global_step=1)
writer.close()
