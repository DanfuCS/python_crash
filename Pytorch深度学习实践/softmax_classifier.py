# import torch
#
# criterion = torch.nn.CrossEntropyLoss()
# Y = torch.LongTensor([2, 0, 1])
#
# Y_pred1 = torch.tensor([[0.1, 0.2, 0.9],
#                         [1.1, 0.1, 0.2],
#                         [0.2, 2.1, 0.1]])
# Y_pred2 = torch.tensor([[0.8, 0.2, 0.3],
#                         [0.2, 0.3, 0.5],
#                         [0.2, 0.2, 0.5]])
#
# l1, l2 = criterion(Y_pred1, Y), criterion(Y_pred2, Y)
# print(l1.data, l2.data)
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = self.l5(x)  # 最后一层不做激活，直接采用CrossEntropyLoss计算loss
        return x


model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)


def train(epoch):
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader, 0):
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
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"accuracy on test dataset: {(100 * correct / total)}%%")


if __name__ == '__main__':
    for epoch in range(1, num_epochs):
        train(epoch)
        if epoch % 10 == 0:
            test()


