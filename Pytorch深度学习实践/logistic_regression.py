import torch.nn as nn
import torch.optim
import numpy as np
import matplotlib.pyplot as plt

num_epochs = 1000
learning_rate = 0.01

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[0], [0], [1]])


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegression()

criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(1, num_epochs+1):
    y_pred = model(x_data).to(torch.float32)
    y_data = y_data.to(torch.float32)
    loss = criterion(y_pred, y_data)
    print(f"epoch {epoch}: {loss.item()}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0, 100, 2000)
x_t = torch.tensor(x).view((2000, 1))
y_t = model(x_t.to(torch.float32))
y = y_t.data.numpy()

plt.plot(x, y)
plt.plot([0, 100], [0.5, 0.5], c='r')
plt.xlabel("Hours")
plt.ylabel("Probability of Pass")
plt.grid()
plt.show()
