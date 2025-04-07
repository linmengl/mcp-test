# 省略了一些模块的引入
from efficientnet_pytorch import EfficientNet
from dataset import build_data_set
import torch
import argparse
import torch.nn as nn
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        print('Epoch ', epoch, ' Batch ', batch_idx, ' Loss ', loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main(args):

    args.classes_num = 2
    # part1: 模型加载
    if args.pretrained:
        # 只使用全连接层以外的参数
        model = EfficientNet.from_pretrained(args.arch, num_classes=args.classes_num, advprop=args.advprop)
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
        model = EfficientNet.from_name(args.arch, override_params={'num_classes': args.classes_num})

    # part2: 损失函数、优化方法
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    train_dataset = build_data_set(args.image_size, args.train_data)

    # 数据集
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        )

    # 假设验证集和训练集在同一个路径，你可以改成 val 的路径
    val_dataset = build_data_set(args.image_size, "./data/val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for epoch in range(args.epochs):
        # 调用train函数进行训练，稍后补充
        train(train_loader, model, criterion, optimizer, epoch)
        # 模型评估
        validate(val_loader, model)
        # 模型保存
        if epoch % args.save_interval == 0:
            if not os.path.exists(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir,
                    'checkpoint.pth.tar.epoch_%s' % epoch))

def validate(val_loader, model):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            pred = torch.argmax(output, dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 转成 numpy 数组
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 评估指标
    cm = confusion_matrix(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='binary')
    recall = recall_score(all_targets, all_preds, average='binary')
    f1 = f1_score(all_targets, all_preds, average='binary')

    print("\n[Validation Result]")
    print("Confusion Matrix:\n", cm)
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=u"[+]----------------- 图像分类Sample -----------------[+]")
    parser.add_argument('--train_data', default='./data/train', help='location of train data')
    parser.add_argument('--image-size', default=224, dest='image_size', type=int, help='size of input image')
    parser.add_argument('--batch-size', default=10, dest='batch_size', type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, dest='num_workers', type=int, help='worders number of Dataloader')
    parser.add_argument('--epochs', default=10, type=int, help='epochs')
    parser.add_argument('--lr',  default=0.001,type=float, help='learning rate')
    parser.add_argument('--checkpoint-dir', default='./ckpts/', dest='checkpoint_dir', help='location of checkpoint')
    parser.add_argument('--save-interval', default=1, dest='save_interval', type=int, help='save interval')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay', help='weight decay')
    parser.add_argument('--arch', default='efficientnet-b0', help='arch type of EfficientNet')
    parser.add_argument('--pretrained', default=True, help='learning rate')
    parser.add_argument('--advprop', default=False, help='advprop')
    args = parser.parse_args()
    main(args)



