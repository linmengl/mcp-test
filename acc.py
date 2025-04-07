import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
# 使用GPU 如果无使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 加载小程序数据集
data = pd.read_csv('mini_program_data.csv')
# 提取特征和标签
X = data[['completeness', 'error_rate']].values  # 特征列
y = data['label'].values  # 标签列（优质与否）
# 特征标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1).to(device)
# 创建 TensorDataset 和 DataLoader
batch_size = 32  # 设置批次大小
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 定义逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        # 使用sigmoid
        return torch.sigmoid(self.linear(x))
# 初始化模型、损失函数和优化器
input_dim = X_train.shape[1]
model = LogisticRegressionModel(input_dim).to(device)
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器，学习率 0.01
# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()  # 设定为训练模式
    for batch_X, batch_y in train_loader:  # 使用 DataLoader 进行批次训练
        optimizer.zero_grad()  # 清零梯度
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        # 反向传播和优化
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:  # 每10个epoch输出一次损失
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
model.eval()  # 设定为评估模式
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    predicted = (val_outputs > 0.5).float()  # 使用0.5作为阈值进行分类
    # 计算准确率和F1分数
    accuracy = accuracy_score(y_val_tensor.cpu(), predicted.cpu())  
    f1 = f1_score(y_val_tensor.cpu(), predicted.cpu())
    print(f'Validation Accuracy: {accuracy:.2f}')
    print(f'Validation F1 Score: {f1:.2f}')