import torchvision
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from PIL import Image
import argparse
from dataset import build_transform

if __name__ == '__main__':
    model = EfficientNet.from_name('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load('ckpts/checkpoint.pth.tar.epoch_8'))
    model.eval()
    # image = Image.open("./data/val/others/1.jpeg").convert('RGB')
    image = Image.open("data/val/logo/06.jpeg").convert('RGB')
    transform = build_transform(224)
    input_tensor = transform(image).unsqueeze(0)
    pred = model(input_tensor)
    # pred = model(input_tensor).argmax()
    print("prediction:", pred)