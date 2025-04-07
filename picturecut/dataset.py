import os
import torch
import numpy as np

from torch.utils.data import Dataset
from PIL import Image


class CatSegmentationDataset(Dataset):
    # 模型输入是3通道数据
    in_channels = 3
    # 模型输出是1通道数据
    out_channels = 1

    def __init__(self, images_dir, image_size=256,):
        print("Reading images...")
        # 原图所在的位置
        # image_root_path = images_dir + os.sep + 'JPEGImages'
        image_root_path = os.path.join(images_dir, 'JPEGImages')
        # Mask所在的位置
        mask_root_path = images_dir + os.sep + 'SegmentationClass'
        # 将图片与Mask读入后，分别存在image_slices与mask_slices中
        self.image_slices = []
        self.mask_slices = []
        for im_name in os.listdir(image_root_path):
            # 原图与mask的名字是相同的，只不过是后缀不一样
            mask_name = im_name.split('.')[0] + '.png'

            image_path = os.path.join(image_root_path, im_name)
            mask_path = os.path.join(mask_root_path, mask_name)

            im = np.asarray(Image.open(image_path).resize((image_size, image_size)))
            mask = np.asarray(Image.open(mask_path).resize((image_size, image_size)))
            self.image_slices.append(im / 255.)
            self.mask_slices.append(mask)

    def __len__(self):
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]
        mask = self.mask_slices[idx]

        # tensor的顺序是（Batch_size, 通道，高，宽）而numpy读入后的顺序是(高，宽，通道)
        image = image.transpose(2, 0, 1)
        # Mask是单通道数据，所以要再加一个维度
        mask = mask[np.newaxis, :, :]

        image = image.astype(np.float32)
        mask = mask.astype(np.float32)

        return image, mask