# ai_demo_mac.py
import torch
from torchvision import models, transforms
from PIL import Image

# 设备设置（关键修改）
device = torch.device("mps")  # 使用Apple Metal加速
model = models.resnet18(pretrained=True).to(device)
model.eval()

# 预处理管道
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict(image_path):
    # 加载图像
    img = Image.open(image_path)
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)  # 数据转移至MPS
    
    # 推理
    with torch.no_grad():
        output = model(input_batch)
    
    # 解析结果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f]
    top3 = torch.topk(probabilities, 3)
    return {labels[idx]: prob.item() for prob, idx in zip(top3.values, top3.indices)}

if __name__ == "__main__":
    results = predict("test_image.jpg")  # 替换你的图片路径
    for cls, prob in results.items():
        print(f"{cls}: {prob:.2%}")