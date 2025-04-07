import torch;

x = torch.arange(12)
print(x)
print(x.shape)
print(x.view(3,4))
print(x.reshape(4,3))

for i in range ( 5 ) : print ( i )