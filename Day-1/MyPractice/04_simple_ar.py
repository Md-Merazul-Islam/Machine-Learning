import torch

r = torch.rand(2,2) - 0.5 * 2
print("a random matrix : \n", r)



print("\n absolute value of  r : \n", torch.abs(r))
# print("\n absolute value of  r : \n", torch.absolute(r))

print("\n inverse of r : \n", torch.asin(r))

print("\n determinant of r : \n", torch.det(r))


print("\n singular value of r : \n", torch.svd(r))

print("\n average value of r : \n", torch.std_mean(r))

print("\n maximum value of r : \n", torch.max(r))


