import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# x = np.linspace(0, 10, 10)
# print(x)

im = Image.open('aa.jpg')
print(im.size)
# 输出: 318, 116

im_pillow = np.asarray(im)
print(im_pillow.shape)

print("--------------")

# 取第三个维度索引为 0 的全部数据
im_pillow_c1 = im_pillow[:,:,0]
# 取第三个维度索引为 1 的全部数据
im_pillow_c2 = im_pillow[:,:,1]
# 取第三个维度索引为 2 的全部数据
im_pillow_c3 = im_pillow[:,:,2]

print(im_pillow_c1.shape)

zeros = np.zeros((im_pillow.shape[0], im_pillow.shape[1], 1), dtype=np.uint8)
print(zeros.shape)

# 数组增加一个维度  np.newaxis 用来 扩展维度
im_pillow_c1 = im_pillow_c1[:,:,np.newaxis]
im_pillow_c2 = im_pillow_c2[:,:,np.newaxis]
im_pillow_c3 = im_pillow_c3[:,:,np.newaxis]
print("im_pillow_c1--")
print(im_pillow_c1)

# axis=2（通道维度） 在 NumPy 中，axis 代表 在哪个维度上进行拼接
red_image = np.concatenate((im_pillow_c1, zeros, zeros), axis=2)
green_image = np.concatenate((zeros, im_pillow_c2, zeros), axis=2)
blue_image = np.concatenate((zeros, zeros, im_pillow_c3), axis=2)
print("红色图片")
print(red_image)

# 画图
plt.figure(figsize=(8, 6))

plt.subplot(2,2,1)
plt.title("原图")
plt.imshow(im_pillow)
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Red Channel")
plt.imshow(red_image.astype(np.uint8))
plt.axis('off')

plt.subplot(2,2,3)
plt.title("绿色通道")
plt.imshow(green_image.astype(np.uint8))
plt.axis('off')

plt.subplot(2,2,4)
plt.title("蓝色通道")
plt.imshow(blue_image.astype(np.uint8))
plt.axis('off')
# plt.savefig('./rgb_pillow.png', dpi=150)

plt.show()
