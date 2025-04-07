from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


image = Image.open("aa.jpg")
im_pillow = np.array(image)

im_copy_red = np.array(im_pillow)
im_copy_red[:,:,1:] = 0

im_copy_green = np.array(im_pillow)
im_copy_green[:,:,[0,2]] = 0

im_copy_blue = np.array(im_pillow)
im_copy_blue[:,:,:2] = 0

plt.figure(figsize=(8, 6))

plt.subplot(2,2,1)
plt.title("原图")
plt.imshow(im_pillow)
plt.axis('off')

plt.subplot(2,2,2)
plt.title("Red Channel-2")
plt.imshow(im_copy_red.astype(np.uint8))
plt.axis('off')

plt.subplot(2,2,3)
plt.title("绿色通道")
plt.imshow(im_copy_green.astype(np.uint8))
plt.axis('off')

plt.subplot(2,2,4)
plt.title("蓝色通道")
plt.imshow(im_copy_blue.astype(np.uint8))
plt.axis('off')
# plt.savefig('./rgb_pillow.png', dpi=150)

plt.show()
