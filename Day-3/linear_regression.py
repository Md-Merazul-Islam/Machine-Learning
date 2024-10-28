import torch
import numpy as np
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')


targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# convert in tensor
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
# print(inputs)
# print(targets)

# Weights and biases
# temp , rainfall, humidity for apple
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)  # temp , rainfall, humidity for orange
# print(w)
# print(b)


def model(x):
    return x @ w.t()+b


preds = model(inputs)
# print(targets)
# print(preds)

# MSE loos


def mse(t1, t2):
    diff = t1-t2
    return torch.sum(diff * diff) / diff.numel()


loss = mse(preds, targets)
# print(loss)

# compute gradinet
loss.backward()

# geadinets for weights

w.grad

# reset weights
w.grad.zero_()
b.grad.zero_()


# generate predictions
preds = model(inputs)
print(preds)

# calculate the loss
loss = mse(preds, targets)
print("fist los : ", loss)


loss.backward()
# print("\nafter backward : \n")
# print(w.grad)
# print(b.grad)

# adjust weights & rest gradient
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

print(w)
print(b)

# calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)

# train for the 100 epoch
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()


preds = model(inputs)
loss = mse(preds, targets)
print(preds)
print(targets)
print(loss)
