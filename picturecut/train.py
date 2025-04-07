import torch

from unet import UNet
from torch.utils.data import DataLoader
from dataset import CatSegmentationDataset as Dataset
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch.nn as nn
"""数据加载"""
def data_loaders(args):
    dataset_train = Dataset(
        images_dir=args.images,
        image_size=args.image_size,
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )

    return loader_train

"""损失函数"""
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc

# args是传入的参数
def main(args):
    makedirs(args)
    # 根据cuda可用情况选择使用cpu或gpu
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    # 加载训练数据
    loader_train = data_loaders(args)
    # 实例化UNet网络模型
    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    # 将模型送入gpu或cpu中
    unet.to(device)
    # 损失函数
    dsc_loss = DiceLoss()
    # 优化方法
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    loss_train = []
    step = 0
    # 训练n个Epoch
    for epoch in tqdm(range(args.epochs), total=args.epochs):
        unet.train()
        for i, data in enumerate(loader_train):
            step += 1
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            y_pred = unet(x)
            optimizer.zero_grad()
            loss = dsc_loss(y_pred, y_true)
            loss_train.append(loss.item())
            loss.backward()
            optimizer.step()
            if (step + 1) % 10 == 0:
                print('Step ', step, 'Loss', np.mean(loss_train))
                loss_train = []
        torch.save(unet, args.ckpts + '/unet_epoch_{}.pth'.format(epoch))

def makedirs(args):
    os.makedirs(args.ckpts, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training U-Net model for segmentation of Cat")
    parser.add_argument("--batch-size",type=int,default=16,help="Batch Size (default: 16)",)
    parser.add_argument("--epochs",type=int,default=100,help="Epoch number (default: 100)",)
    parser.add_argument("--lr",type=float,default=0.0001,help="Learning rate",)
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for training (default: cuda:0)",)
    parser.add_argument("--workers", type=int, default=4, help="Workers' count (default: 4)",)
    parser.add_argument("--ckpts", type=str, default="./ckpts", help="folder to save weights")
    parser.add_argument("--logs", type=str, default="./logs", help="folder to save logs")
    parser.add_argument("--images", type=str, default="./data/cat_output", help="root folder with images")
    parser.add_argument("--image-size", type=int, default=256, help="target input image size (default: 256)",)
    args = parser.parse_args()
    main(args)