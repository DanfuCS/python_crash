# Dataset: 提供一种方式去获取数据(及其label)
#   1. 如何获取每一个数据及其label
#   2. 返回总共存在多少数据
# Dataloader: 为网络模型提供不同的数据形式

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_path[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "../dataset/hymenoptera_data/hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

# 展示图片
img, label = ants_dataset[0]
img.show()

# 拼接数据集
train_dataset = ants_dataset + bees_dataset
print(len(ants_dataset))
print(len(bees_dataset))
print(len(train_dataset))
