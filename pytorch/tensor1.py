import torch

a = torch.zeros(2,3,5)

print(a.size())
print(a.numel())

print(torch.Size([2,3,5]))

print("---------------------------")

x = torch.rand(2,1,3)
print(x.shape)

y = x.squeeze(-2)
print(y.shape)


print("---------------------------2")

A=torch.tensor([[4,5,7], [3,9,8],[2,3,4]])
print(A)
mask = torch.tensor([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
B = torch.masked_select(A, mask>0)
print(B)
