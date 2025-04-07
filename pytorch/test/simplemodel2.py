import numpy as np
import random
from matplotlib import pyplot as plt
import torch
from torch import nn

w = 2
b = 3
xlim = [-10, 10]
x_train = np.random.randint(low=xlim[0], high=xlim[1], size=30)

y_train = [w * x + b + random.randint(0,2) for x in x_train]

plt.plot(x_train, y_train, 'bo')


class CustomLayer(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.conv1_1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=3, padding='same')
        self.conv1_2 = nn.Conv2d(in_channels=3, out_channels=output_channels, kernel_size=2, padding='same')

    def forward(self, input):
        x = self.conv1_1(input)
        x = self.conv1_2(x)
        return x


class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = CustomLayer(1,1)
        self.layerm = CustomLayer(1,1)
        self.layern = CustomLayer(1,1)

    def forward(self, input):
        x = self.aa(input)
        x = self.layerm(x)
        x = self.layern(x)
        return x


model = CustomModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)
y_train = torch.tensor(y_train, dtype=torch.float32)

for epoch in range(1000):
    inputs = torch.Tensor(x_train)
    # model(x) 就相当于调用 LinearModel 中的 forward 方法
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, y_train)
    model.zero_grad()
    loss.backward()
    optimizer.step()

for param in model.named_parameters():
    print(param)