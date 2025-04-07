import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 官方文档 https://lightgbm.readthedocs.io/en/latest/Parameters.html?utm_source=chatgpt.com#core-parameters

# 读取 CSV 数据
df = pd.read_csv("data.csv")

# 假设特征列是 'feature1', 'feature2', ..., 'featureN'
feature_columns = ['feature1', 'feature2', 'feature3']
target_column = 'label'  # 目标变量

# 选取特征和目标变量
X = df[feature_columns]
y = df[target_column]

# 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 LightGBM，创建数据集，lgb.Dataset() 是 LightGBM 专用的数据格式，用于训练和验证。
lgb_train = lgb.Dataset(X_train, y_train)
# reference=lgb_train 表示 lgb_eval 作为评估集，但不会用于训练，只用于监控模型表现。
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)


params = {
    'objective': 'binary',     # 二分类问题
    'metric': 'accuracy',      # 评估指标为准确率
    'boosting_type': 'gbdt',   # 使用 GBDT（梯度提升决策树）
    'learning_rate': 0.1,      # 学习率，控制更新步长
    'num_leaves': 31           # 叶子节点数，控制模型复杂度
}

# 训练LightGBM模型
""" lgb.train() 用于训练模型。
 valid_sets=[lgb_eval] → 使用 lgb_eval 作为验证集，监控过拟合。
 early_stopping_rounds=10 → 如果验证集的性能在 10 轮内没有提升，则提前停止训练，防止过拟合。"""
model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], early_stopping_rounds=10)

# 预测
""" model.predict(X_test) 计算测试集的 预测概率（范围 0~1）。
    > 0.5 → 概率大于 0.5 预测为 1，否则为 0。
	astype(int) → 将布尔值转换为整数（0 或 1）。"""
y_pred = (model.predict(X_test) > 0.5).astype(int)

# 评估。计算模型的准确率（Accuracy），即正确预测的样本数占总样本数的比例
print("LightGBM 准确率:", accuracy_score(y_test, y_pred))

# 存储模型
model.save_model("lightgbm_model.txt")

# 加载模型
# loaded_model = lgb.Booster(model_file="lightgbm_model.txt")