# loss:
#   1. 计算实际输出与目标之间的差距
#   2. 为更新输出提供一定的依据(反向传播)
import torch
import torchvision.datasets
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

# inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# targets = torch.tensor([1, 2, 5], dtype=torch.float32)
#
# inputs = torch.reshape(inputs, (1, 1, 1, 3))
# targets = torch.reshape(targets, (1, 1, 1, 3))
#
# loss_l1 = L1Loss(reduction='mean')
# result_l1 = loss_l1(inputs, targets)
# print(result_l1)
#
# loss_mse = MSELoss()
# result_mse = loss_mse(inputs, targets)
# print(result_mse)
#
# x = torch.tensor([0.1, 0.2, 0.3])
# y = torch.tensor([1])
# x = torch.reshape(x, (1, 3))
# loss_cross = CrossEntropyLoss()
# result_cross = loss_cross(x, y)
# print(result_cross)


dataset = torchvision.datasets.CIFAR10("../dataset", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, stride=1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, stride=1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


net = Model()
loss = nn.CrossEntropyLoss()

for data in dataloader:
    imgs, targets = data
    outputs = net(imgs)
    print(outputs.shape)
    print(targets.shape)

    result_loss = loss(outputs, targets)
    print(result_loss)
    result_loss.backward()
    print("ok")
