import torch

x = torch.tensor([[1.0,2.0],[3.0,2.0]])
print(x)

zeros = torch.zeros(2,3)
print(zeros)

ones = torch.ones(4,5)
print(ones)

rands = torch.rand(2,2)
print("rand-----"+str(rands))

randns = torch.randn(2,2)
print("rand-----"+str(randns))

import numpy as np

np_array = np.array([[1,2],[3,4]])
torch_tensor = torch.from_numpy(np_array)
print(torch_tensor)

a = torch.tensor([3,5,7])
b = torch.tensor([2,4,5])
print(a+b)

x = torch.tensor([[1,2],[3,4]])
y = torch.tensor([[5,6],[7,8]])
print(torch.matmul(x,y))

print("-----------------")

m = torch.randn(2,3,4)
print("torch.randn m="+str(m))
n = torch.randn(2,4,5)
print("torch.randn n="+str(n))
c = torch.bmm(m,n)
print(c)


print("-----------------------------------")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mm = torch.tensor([1,2,3], device=device)
print(mm)

print(torch.backends.mps.is_available())  # 是否支持 MPS
print(torch.backends.mps.is_built())      # PyTorch 是否编译了 MPS 支持


