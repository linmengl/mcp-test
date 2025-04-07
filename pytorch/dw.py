import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# 创建计算图对象
plt.figure(figsize=(10, 6))

# 定义输入和卷积核
input_data = np.array([[1, 2, 3, 4, 5],
                       [6, 7, 8, 9, 10],
                       [11, 12, 13, 14, 15],
                       [16, 17, 18, 19, 20],
                       [21, 22, 23, 24, 25]])
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

# 定义卷积层和激活函数
conv_layer = signal.convolve2d(input_data, kernel, mode='valid')
activation = np.maximum(conv_layer, 0)

# 绘制卷积计算图
plt.subplot(131)
plt.imshow(input_data, cmap='gray')
plt.title('Input')

plt.subplot(132)
plt.imshow(kernel, cmap='gray')
plt.title('Kernel')

plt.subplot(133)
plt.imshow(activation, cmap='gray')
plt.title('Activation')

plt.tight_layout()
plt.show()