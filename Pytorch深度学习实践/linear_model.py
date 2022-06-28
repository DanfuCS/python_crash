# 引入必要的包或者库
import numpy as np
import matplotlib.pyplot as plt

# 准备训练集
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# 定义模型
# w是全局变量，python可以直接在函数中使用全局变量
def model(x):
    return x * w


# 定义损失函数
def loss(x, y):
    y_pred = model(x)
    return (y_pred - y) ** 2


# 列举权重和对应的MSE
w_list = []
mse_list = []

for w in np.arange(0, 4.1, 0.1):
    print("w=", w)
    loss_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = model(x_val)
        loss_val = loss(x_val, y_val)
        loss_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    loss_mean = loss_sum / len(x_data)
    print("MSE=", loss_mean)
    w_list.append(w)
    mse_list.append(loss_mean)


# 作图可视化
plt.plot(w_list, mse_list)
plt.xlabel("w")
plt.ylabel("Loss")
plt.show()