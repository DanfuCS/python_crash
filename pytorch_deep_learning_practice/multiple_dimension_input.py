import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000
learning_rate = 0.01

# 1. 建立数据集
data_info = np.loadtxt('../dataset/diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(data_info[:, :-1])  # numpy到tensor的转换
y_data = torch.from_numpy(data_info[:, [-1]])


# 2. 设计模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(8, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)
        self.activate1 = nn.ReLU()
        self.activate2 = nn.Sigmoid()

    def forward(self, x):
        x = self.activate1(self.linear1(x))
        x = self.activate1(self.linear2(x))
        x = self.activate2(self.linear3(x))
        return x


model = Model()

# 3. 建立损失函数和优化器
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 4. 训练循环
for epoch in range(1, num_epochs+1):
    # forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f"epoch {epoch}: {loss.item()}")

    # backward
    optimizer.zero_grad()
    loss.backward()

    # update
    optimizer.step()

