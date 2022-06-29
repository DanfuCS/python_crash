import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

batch_size = 32
num_workers = 8
num_epochs = 200
learning_rate = 0.01


class DiabetesDataset(Dataset):  # 继承自Dataset类
    def __init__(self, file_path):
        all_data = np.loadtxt(file_path, delimiter=',', dtype=np.float32)
        self.len = all_data.shape[0]
        self.x_data = torch.from_numpy(all_data[:, :-1])
        self.y_data = torch.from_numpy(all_data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('../dataset/diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)


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
criterion = nn.BCELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

epoch_list = []
loss_list = []

if __name__ == '__main__':
    for epoch in range(1, num_epochs + 1):
        for i, (inputs, labels) in enumerate(train_loader, 0):  # enumerate(seq, [start=0])
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_list.append(epoch)
        loss_list.append(loss.data.item())
        print(f"epoch {epoch} loss {loss.item()}")

    plt.plot(epoch_list, loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()
