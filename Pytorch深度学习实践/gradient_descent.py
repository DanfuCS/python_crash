import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# 设置初始w
w = 1.0
alpha = 0.01


# 定义模型
def model(x):
    return x * w


# 定义损失函数
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):  # 打包为元组的列表：如zipped = zip(xs, ys)，则zipped = [(1, 2), (2, 4), (4, 6)]
        y_pred = model(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# 定义梯度下降的计算
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)


epoch_list = []
cost_list = []
print(f"Predict (before training) 4 {model(4)}")

for epoch in range(100):
    cost_val = cost(x_data, y_data)  # 计算第epoch次的所有样本的平均loss
    gradient_val = gradient(x_data, y_data)  # 计算第epoch次的所有样本的平均梯度值
    w -= alpha * gradient_val  # 梯度更新
    print(f"Epoch {epoch}, w={w}, loss={cost_val}")
    epoch_list.append(epoch)
    cost_list.append(cost_val)

print(f"Predict (after training) 4 {model(4)}")

plt.plot(epoch_list, cost_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
