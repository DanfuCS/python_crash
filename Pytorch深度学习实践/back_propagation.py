import torch

x_data = [1.0, 2.0, 3.0]
y_data = [6.0, 15.0, 28.0]
num_epochs = 100
alpha = 0.005

w1 = torch.Tensor([1.0])
w1.requires_grad = True

w2 = torch.Tensor([1.0])
w2.requires_grad = True

b = torch.Tensor([1.0])
b.requires_grad = True


def forward(x):
    return x ** 2 * w1 + x * w2 + b


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


for epoch in range(num_epochs):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # 前馈计算，构建计算图，同时计算得到loss
        l.backward()  # 反馈计算，计算requires_grad=True的tensor关于loss的梯度
        print('\t grad', w1.grad.item(), w2.grad.item(), b.grad.item(),)
        w1.data = w1.data - alpha * w1.grad.data  # w.grad用于更新权重
        w2.data = w2.data - alpha * w2.grad.data
        b.data = b.data - alpha * b.grad.data

        w1.grad.data.zero_()  # 更新后，为避免梯度累积，需将grad置0
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print("progress: ", epoch, l.item())

print("predict (after training)", 4, forward(4).item())
