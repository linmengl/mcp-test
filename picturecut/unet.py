import torch.nn as nn
import torch

"""基本卷积模块"""
class Block(nn.Module):

    # 基础卷积单元
    def __init__(self, in_channels, features):
        super(Block, self).__init__()

        self.features = features
        self.conv1 = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )
        self.conv2 = nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )

    def forward(self, input):
        x = self.conv1(input)
        # 标准化，稳定训练，加快收敛。对每个 mini-batch 的数据做标准化处理（变成均值为 0，方差为 1），加快训练、提高稳定性
        x = nn.BatchNorm2d(num_features=self.features)(x)
        # 非线性激活，增加表达力
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(num_features=self.features)(x)
        x = nn.ReLU(inplace=True)(x)

        return x


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        # 编码器负责提取图像的低级和高级特征，通过一系列的卷积层来实现。每一层都会通过卷积操作来提取不同层次的特征。
        self.conv_encoder_1 = Block(in_channels, features)
        self.conv_encoder_2 = Block(features, features * 2)
        self.conv_encoder_3 = Block(features * 2, features * 4)
        self.conv_encoder_4 = Block(features * 4, features * 8)

        # 瓶颈层是网络最深的部分，它的作用是通过较深的特征提取来捕捉高级语义信息。
        # 这里，瓶颈层接收来自编码器第四层（features * 8 通道）的特征图，并输出 features * 16 通道的特征图
        self.bottleneck = Block(features * 8, features * 16)

        # 解码器的作用是通过反卷积（ConvTranspose2d）操作逐步恢复图像的空间分辨率，同时将高层特征映射到更低层的空间。
        # 解码器还会将对应层的编码器特征图与解码器的输出进行拼接（跳跃连接）。
        # upconv4通过反卷积（上采样）操作，将 features * 16 通道的特征图上采样到 features * 8 通道
        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        # conv_decoder_4: 对上采样后的特征图和来自编码器第四层的特征图进行拼接，然后通过卷积操作得到新的特征图
        self.conv_decoder_4 = Block((features * 8) * 2, features * 8)

        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.conv_decoder_3 = Block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.conv_decoder_2 = Block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Block(features * 2, features)
        # 网络通过一个 1x1 的卷积层，将解码器输出的特征图（通道数为 features）转换为最终的输出通道数 out_channels（通常是 1，用于二分类任务）
        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        conv_encoder_1_1 = self.conv_encoder_1(x)
        conv_encoder_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_1_1)

        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        conv_encoder_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_2_1)

        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        conv_encoder_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_3_1)

        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        conv_encoder_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_4_1)

        bottleneck = self.bottleneck(conv_encoder_4_2)

        conv_decoder_4_1 = self.upconv4(bottleneck)
        conv_decoder_4_2 = torch.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2)

        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        conv_decoder_3_2 = torch.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        conv_decoder_2_2 = torch.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)

        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)
        conv_decoder_1_2 = torch.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)
        conv_decoder_1_3 = self.decoder1(conv_decoder_1_2)

        return torch.sigmoid(self.conv(conv_decoder_1_3))