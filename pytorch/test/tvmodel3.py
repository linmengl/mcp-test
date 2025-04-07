import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision
import PIL.Image as Image
import multiprocessing

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        # conv1输出的特征图为222x222大小
        self.fc = nn.Linear(16 * 222 * 222, 10)

    def forward(self, input):
        x = self.conv1(input)
        # 进去全连接层之前，先将特征图铺平
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

# if __name__ == '__main__' : # 多进程配置
#     multiprocessing.set_start_method('spawn' ,force = True)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = MyCNN().to(device)
#
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#
#     cifar10_dataset = torchvision.datasets.CIFAR10(root = "./data", train = False, download=True, transform=transform, target_transform=None)
#
#     dataloader = torch.utils.data.DataLoader(cifar10_dataset, batch_size=32, shuffle=True, num_workers=2)
#
#     # 优化器
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay = 1e-2, momentum=0.9)
#     steps = 0
#     for epoch in range(16):
#         for images, labels in dataloader:
#             outputs = model(images)
#             # 交叉熵损失函数
#             loss = nn.CrossEntropyLoss()(outputs, labels)
#             steps += 1
#             if steps % 100 == 0:
#                 print('Epoch {}, Loss {}'.format(epoch + 1, loss))
#             # 清空梯度
#             optimizer.zero_grad()
#             # 反向传播 计算梯度
#             loss.backward()
#             # 应用梯度更新参数  仅更新优化器初始化时传入的参数
#             optimizer.step()
#
#     print(model.state_dict())
#     torch.save(model, "./model/myCNN.pth")
#
#     im = Image.open('dog.jpg')
#     input_tensor = transform(im).unsqueeze(0)
#     result = model(input_tensor.to(device)).argmax()
#     print(result)

if __name__ == '__main__' :
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load('./model/myCNN.pth', weights_only = False)
    model.eval()

    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


    im = Image.open('dog.jpg').convert('RGB')
    input_tensor = transform(im).unsqueeze(0)
    result = model(input_tensor.to(device)).argmax()
    print(result)