import torch.optim
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import *

# 指定GPU训练:
# 方式1: 在以下三个位置加.cuda()
#   1. 网络模型
#   2. 数据(输入, 标注)
#   3. 损失函数
# 方式2: 指定device, 然后to(device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.CIFAR10("../dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=False)
test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=False)
# 得到数据集大小
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的图片数: {train_data_size}\t 测试数据集的图片数: {test_data_size}")

# 利用dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=128)
test_dataloader = DataLoader(test_data, batch_size=128)

# 创建网络模型
model = Model().to(device)

# 指定损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 指定优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 设置训练网络的参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
num_epochs = 100  # 训练的轮数

# 添加tensorboard
writer = SummaryWriter("../logs_train")

for epoch in range(num_epochs):
    print(f"-------第{epoch + 1}轮训练开始-------")

    # 训练步骤开始
    model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    model.eval()
    total_test_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            test_loss = loss_fn(outputs, targets)
            total_test_loss += test_loss
            accuracy = ((outputs.argmax(1) == targets).sum())
            total_accuracy += accuracy

    # 测试结果显示
    print(f"整体测试集上的Loss: {total_test_loss}")
    print(f"整体测试集上的Acc: {total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(model.state_dict(), "model_{}.pth".format(epoch))
    print("模型已保存")

writer.close()
