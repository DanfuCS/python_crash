import torch

num_epochs = 100
learning_rate = 0.01

# 1. Prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


# 2. Design model
class LR(torch.nn.Module):  # 模型继承自nn.Module类，是所有网络模块的基类
    def __init__(self):  # 构造函数：用于初始化对象
        super(LR, self).__init__()  # LR调用父类的构造
        self.linear = torch.nn.Linear(1, 1)  # 构造Linear对象，包含weight和bias

    def forward(self, x):  # 定义在前馈过程中执行的计算
        y_pred = self.linear(x)  # nn.Linear已经实现了__call__()
        return y_pred


model = LR()  # 创建一个LR类的实例

# 3. Construct loss and optimizer
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# 4. Training cycle (前馈+反馈+更新)
for epoch in range(1, num_epochs+1):
    # 前馈计算：预测+计算loss
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f"epoch {epoch}: {loss}")

    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 反向传播
    optimizer.step()  # 更新模型参数


print(f"w={model.linear.weight.item()}")
print(f"b={model.linear.bias.item()}")

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print(f"y_pred={y_test.data}")



