import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.01
num_epochs = 100
transform = transforms.Compose([
    transforms.ToTensor(),  # convert the PIL image to tensor
    transforms.Normalize((0.1307, ), (0.3081, ))  # the parameters are mean and std
])

# 准备数据集
train_dataset = datasets.MNIST(root="../dataset/mnist",
                               train=True,
                               download=False,
                               transform=transform)
test_dataset = datasets.MNIST(root="../dataset/mnist",
                              train=False,
                              download=False,
                              transform=transform)
# 加载数据集
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size,
                          drop_last=True)
test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         batch_size=batch_size)


# design model
class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

        self.branch_1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch_5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch_3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch_3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch_3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, x):
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        branch_1x1 = self.branch_1x1(x)

        branch_5x5 = self.branch_5x5_1(x)
        branch_5x5 = self.branch_5x5_2(branch_5x5)

        branch_3x3 = self.branch_3x3_1(x)
        branch_3x3 = self.branch_3x3_2(branch_3x3)
        branch_3x3 = self.branch_3x3_3(branch_3x3)

        outputs = [branch_pool, branch_1x1, branch_3x3, branch_5x5]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = Inception(in_channels=10)
        self.incep2 = Inception(in_channels=20)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.maxpool(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.maxpool(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)


def train(epoch):
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % 937 == 936:
            print(f"epoch {epoch}\t loss {train_loss/937}")
            train_loss = 0


def test():
    correct = 0
    total = 0
    test_loss_epoch = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"accuracy on test dataset: {(100 * correct / total)}%%")


if __name__ == '__main__':
    for epoch in range(1, num_epochs+1):
        train(epoch)
        if epoch % 5 == 0:
            test()
