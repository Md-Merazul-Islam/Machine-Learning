import torch 

one = torch.ones(2,3)
# print(one)

two = torch.ones(2,3) *2
# print(two)

three = one + two
# print(three)

print(three.shape) #for show size 




# this code will give runtime error because tensor arithmetic operation must be similar type shape
r1 = torch.rand(2,3)

r2 = torch.rand(3,2)

r3 = r1 + r2
print(r3)



