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


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # 作为 nn.Module 中可训练的参数使用
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
    # 必须重写，前向传播
    # model(x) 就相当于调用 LinearModel 中的 forward 方法
    def forward(self, x):
        return self.weight * x + self.bias


model = LinearModel()
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)
y_train = torch.tensor(y_train, dtype=torch.float32)

for epoch in range(1000):
    inputs = torch.Tensor(x_train)
    outputs = model(inputs)
    # 损失函数
    loss = nn.MSELoss()(outputs, y_train)
    model.zero_grad()
    loss.backward()
    optimizer.step()

print("----------")
print(model.state_dict())
print("----------")

for param in model.named_parameters():
    print(param)

print(model.forward(2))

print("----------")

# 保存模型
torch.save(model.state_dict(), "./model.pth")

liner_model2 = LinearModel()
# 加载模型
liner_model2.load_state_dict(torch.load("./model.pth"))
liner_model2.eval()


for param in model.named_parameters():
    print(param)

print(liner_model2.forward(2))
