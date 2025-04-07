import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 作者给出的标准化方法
def _norm_advprop(img):
    return img * 2.0 - 1.0

# 图片转换
def build_transform(dest_image_size):
    normalize = transforms.Lambda(_norm_advprop)

    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)
    else:
        dest_image_size = dest_image_size

    transform = transforms.Compose([
        transforms.RandomResizedCrop(dest_image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return transform

# 数据加载
def build_data_set(dest_image_size, data):
    transform = build_transform(dest_image_size)
    dataset=datasets.ImageFolder(data, transform=transform, target_transform=None)
    return dataset
