# 1. tensorboard的使用
import torch.utils.tensorboard
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img_path = "../dataset/hymenoptera_data/hymenoptera_data/train/ants/6240338_93729615ec.jpg"
img = Image.open(img_path)
img_array = np.array(img)
print(type(img_array))
print(img_array.shape)

writer.add_image(tag="Image show", img_tensor=img_array, global_step=1, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x", scalar_value=2*i, global_step=i)

writer.close()
