import torch

torch.manual_seed(1729)
rd1 = torch.rand(5, 2)
print("print the random tensor : \n", rd1)


rd2 = torch.rand(5, 2)
print("print another  the random tensor : \n", rd2)

torch.manual_seed(1729)
rd3 = torch.rand(5, 2)

print("print the same random tensor after reseeding: \n", rd3)