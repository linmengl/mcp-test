from torchvision import transforms
from PIL import Image

image = Image.open("aa.jpg")

print(type(image))




image1 = transforms.ToTensor()(image)
print(type(image1))

image2 = transforms.ToPILImage()(image1)
print(type(image2))


resize_img_oper = transforms.Resize((200,200), interpolation=2)
img = resize_img_oper(image)
img.save("aa3.jpg")

