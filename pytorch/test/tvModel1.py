import torchvision.models as models
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载预训练的 AlexNet 模型
alexnet = models.alexnet(pretrained=True).to(device)
# torch.save(alexnet.state_dict(), "./model/alexnet-owt-4df8aa71.pth")
# 加载模型权重
alexnet.load_state_dict(torch.load('./model/alexnet-owt-4df8aa71.pth'))

# 冻结特征提取层（前5层卷积），只训练最后的全连接层
for param in alexnet.parameters():
    param.requires_grad = False

# 定义图像预处理步骤
# transform = transforms.Compose([
#     transforms.RandomResizedCrop((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

# 拆分训练/测试预处理逻辑
# 训练集需数据增强防止过拟合，测试集需确定性变换保证评估一致性
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),  # 增强数据多样性
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.RandomRotation(degrees=15),  # 随机旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),          # 保持与训练相同尺度
    transforms.CenterCrop(224),      # 确定性裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载并预处理输入图像
# im = Image.open('dog.jpg').convert('RGB')
# input_tensor = transform(im).unsqueeze(0)
# print(alexnet(input_tensor).argmax())
# 输出：263

# print(alexnet)

# 修改模型的全连接层以适应新的分类任务
fc_in_features = alexnet.classifier[6].in_features
alexnet.classifier[6] = torch.nn.Linear(in_features=fc_in_features, out_features=10)

# print(alexnet)

# 加载 CIFAR-10 数据集
cifar10_dataset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       transform=train_transform,
                                       target_transform=None,
                                       download=True)

# 创建数据加载器
dataloader =  torch.utils.data.DataLoader(dataset=cifar10_dataset, # 传入的数据集, 必须参数
                               batch_size=32,       # 输出的batch大小
                               shuffle=True,       # 数据是否打乱
                               num_workers=2)      # 进程数, 0表示只有主进程

# 定义优化器
optimizer = torch.optim.SGD(alexnet.parameters(), lr=1e-4, weight_decay=1e-2, momentum=0.9)

if __name__ == '__main__' :
    # 训练模型
    writer = SummaryWriter()
    steps = 0
    for epoch in range(3):
        for item in dataloader:
            # 前向传播
            output = alexnet(item[0])
            target = item[1]
            # 交叉熵损失函数 计算损失
            # 使用交叉熵损失函数计算模型输出与真实标签之间的差异。nn.CrossEntropyLoss() 是 PyTorch 中常用的分类损失函数，适用于多类别分类任务
            loss = nn.CrossEntropyLoss()(output, target)
            # 梯度清零  PyTorch 中，梯度是累积的，如果不清零，之前计算的梯度会影响当前的梯度计算
            alexnet.zero_grad()
            # 反向传播 计算损失函数相对于模型参数的梯度，这些梯度将用于后续的参数更新
            loss.backward()
            # 参数更新 模型的参数在每个批次上都会根据当前的梯度进行更新，从而逐步优化模型的性能
            optimizer.step()
            writer.add_scalar('Loss/train', loss, item[0])
            if steps % 100 == 0:
                print('Epoch {}, Loss {}'.format(epoch + 1, loss))

    # 通过 测试集 评估模型的性能，通常的方法包括：
    # 	1.	计算准确率（Accuracy）：测试数据输入模型，比较预测结果与真实标签，计算正确率。
    # 	2.	计算损失（Loss）：在测试集上计算交叉熵损失，看看是否比训练初期更低。
    # 	3.	混淆矩阵（Confusion Matrix）：观察模型在哪些类别上表现较好或较差。

    # 评估模式
    alexnet.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # 评估时不需要计算梯度
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = alexnet(images)
            _, predicted = torch.max(outputs.data, 1)  # 取最大概率的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 评估准确率
    accuracy = 100 * correct / total
    print(f'测试集准确率: {accuracy:.2f}%')

    # 保存模型
    torch.save(alexnet.state_dict(), "./model/alexnet_cifar10.pth")

# 保存最佳模型参数
# torch.save({
#     'epoch': epoch,
#     'model_state_dict': alexnet.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,
# }, "./model/alexnet_cifar10_best.pth")