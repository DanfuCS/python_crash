# 通过在44-46行上进行单步调试，查看data和grad的变化
import torch
import torchvision.datasets
from torch import nn
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

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
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
num_epochs = 100

for epoch in range(num_epochs):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = net(imgs)

        optimizer.zero_grad()
        result_loss = loss(outputs, targets)
        result_loss.backward()
        optimizer.step()

        running_loss = running_loss + result_loss
    print(running_loss)
