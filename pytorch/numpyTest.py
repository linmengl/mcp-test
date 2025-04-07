import numpy as np
import matplotlib.pyplot as plt

# X = np.arange(-50, 51, 2)
# Y = X ** 2
#
# plt.plot(X, Y, color='blue')
# plt.legend()
# plt.show()

scores=np.random.rand(5,5,2)
scores[:,:,0]=1-scores[:,:,1]
mask=np.argmax(scores,axis=2)
print(mask)

print(np.array([2,2]))