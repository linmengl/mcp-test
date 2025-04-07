from flask import Flask, request, jsonify
from torchvision import models
from PIL import Image
from torchvision import transforms
import torch
import io
import json
import requests

app = Flask(__name__)

# 加载预训练模型
model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# 定义 preprocess 函数
# 作用 ：确保输入图像与模型训练时的数据格式一致： 标准化参数 ：基于 ImageNet 数据集的均值和标准差
preprocess = transforms.Compose([
    transforms.Resize(256),             # 调整图像大小
    transforms.CenterCrop(224),          # 中心裁剪为 224x224
    transforms.ToTensor(),               # 转换为张量格式
    transforms.Normalize(               # 归一化处理
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    ),
])

# 定义分类接口
@app.route('/classify', methods=['POST'])
def classify():
    # 接收上传的图片
    file = request.files['image']
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes))

    # 预处理并推理  将图像转换为符合模型输入的张量（添加 unsqueeze(0) 是为了增加 batch 维度）
    input_tensor = preprocess(img).unsqueeze(0)
    # 禁用梯度计算（ torch.no_grad() ）以提升性能
    with torch.no_grad():
        output = model(input_tensor)
    
    # 计算概率并返回结果  通过 Softmax 获取概率分布，读取 ImageNet 类别标签文件，返回概率最高的前 3 个类别
    probs = torch.nn.functional.softmax(output[0], dim=0)

    # with 语句：自动管理文件资源（类似Java的try-with-resources）
    # with open("imagenet_classes.txt") as f:
        # line.strip() ：去除每行首尾的空白字符（类似Java的 String.trim() ）
        # 列表推导式： [x for x in iterable] 是Python特有的简洁语法，相当于Java的循环+ArrayList.add()
        # labels = [line.strip() for line in f] 
    
    # 下载类别标签文件（1000个类别） 
    labels_url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    response = requests.get(labels_url) 
    labels_data = response.json()
    labels = [labels_data[str(i)][1] for i in range (1000)]

    # 创建包含前三名预测类别及其概率的字典 
    # torch.topk(probs, 3) 获取概率张量中最大的3个值
    top3 = torch.topk(probs, 3);
    # probs[i].item() 将PyTorch张量转为Python浮点数
    top3Result = {labels[i]: probs[i].item() for i in top3.indices}
    return jsonify(top3Result)

# 启动 Flask 服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8089)