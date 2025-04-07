import torch
import numpy as np

from PIL import Image

img_size = (256, 256)
# 加载模型
unet = torch.load('./ckpts/unet_epoch_51.pth', weights_only=False)
unet.eval()
# 加载并处理输入图片
ori_image = Image.open('data/cat_output/JPEGImages/6.jpg')
im = np.asarray(ori_image.resize(img_size))
im = im / 255.
im = im.transpose(2, 0, 1)
im = im[np.newaxis, :, :]
im = im.astype('float32')
# 模型预测
output = unet(torch.from_numpy(im)).detach().numpy()
# 模型输出转化为Mask图片
output = np.squeeze(output)
output = np.where(output>0.5, 1, 0).astype(np.uint8)
mask = Image.fromarray(output, mode='P')
mask.putpalette([0,0,0, 0,128,0])
mask = mask.resize(ori_image.size)
# mask.save('output.png')

image = ori_image.convert('RGBA')
mask = mask.convert('RGBA')
# 合成
image_mask = Image.blend(image, mask, 0.3)
image_mask.save("output_mask.png")