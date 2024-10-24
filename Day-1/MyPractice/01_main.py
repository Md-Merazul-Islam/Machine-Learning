import torch

# # Creating a tensor of zeros
zero = torch.zeros(5, 4)
print("The shape is : \n", zero)

# print("this data type is : \n", zero.dtype)

# # create a tensor of ones
one = torch.ones(5, 3)
print("The shape is : \n", one)
print("this data type is : \n", one.dtype)

# # create a tensor of random value
random = torch.rand(3, 4)
print("The shape is : \n", random)
print("this data type : \n", random.dtype)

# so here is see that all data types are 32 bits.


t1 = torch.tensor([[1, 2], [2, 4]])
t2 = torch.tensor([[5, 6], [7, 8]])

ans = t1 + t2

print(ans)
