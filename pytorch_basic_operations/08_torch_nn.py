import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=3, ceil_mode=True)
        self.relu1 = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.relu1(x)
        return x


net = Model()
print(net)
