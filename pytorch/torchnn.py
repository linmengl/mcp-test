import torch
import torch.nn as nn

input_feat = torch.tensor([[4,1,7,5],[4,4,2,5],[7,7,2,4],[1,0,2,4]],dtype=torch.float32).unsqueeze(0).unsqueeze(0)
print(input_feat)

conv2d = nn.Conv2d(1,1,(2,2),stride=1,padding='same',bias=False)
print(conv2d.weight)
print(conv2d.bias)


kernels = torch.tensor([[[[1,0],[2,1]]]],dtype=torch.float32,requires_grad=False)
conv2d.weight = nn.Parameter(kernels,requires_grad=False)

print(conv2d.weight)
print(conv2d.bias)

print("-----------")

out_put = conv2d(input_feat)
print(out_put)

print("==============")

# 生成一个三通道的5x5特征图
x = torch.rand((3, 5, 5)).unsqueeze(0)
print(x.shape)
# 输出：
# torch.Size([1, 3, 5, 5])
# 请注意DW中，输入特征通道数与输出通道数是一样的
in_channels_dw = x.shape[1]
out_channels_dw = x.shape[1]
print(in_channels_dw, out_channels_dw)
# 一般来讲DW卷积的kernel size为3
kernel_size = 3
stride = 1
# DW卷积groups参数与输入通道数一样
dw = nn.Conv2d(in_channels_dw, out_channels_dw, kernel_size, stride, groups=in_channels_dw)

print(dw(x).shape)

print("++++++++++++++")
in_channels_pw = out_channels_dw
out_channels_pw = 4
kernel_size_pw = 1
pw = nn.Conv2d(in_channels_pw, out_channels_pw, kernel_size_pw, stride)
out = pw(dw(x))
print(out.shape)