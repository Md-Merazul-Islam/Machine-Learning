import torch
import numpy as np
x= torch.tensor(3.)
y = torch.tensor(4.,requires_grad=True)
z= torch.tensor(5.,requires_grad=True)

ans = x*y +z 
print(ans)

ans.backward()

print('dx/dy =',x.grad)
print('dy/dz =',y.grad)
print('dz/dx =',z.grad )

# numpy to tensor 
num = np.array([[2,4],[3,5]])
print('num =',num)
num1 = torch.from_numpy(num)
print('num =',num1)
