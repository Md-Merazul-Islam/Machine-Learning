import torch
import numpy as np
import jovian
# Define inputs and targets
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

# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Initialize weights and biases with gradient tracking
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

# Define model
def model(x):
    return x @ w.t() + b

# Define Mean Squared Error (MSE) loss function
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Generate predictions and calculate initial loss
preds = model(inputs)
loss = mse(preds, targets)

# Compute gradients
loss.backward()

# Check gradients for weights
print("Initial gradients for weights:", w.grad)
print("Initial gradients for biases:", b.grad)

# Reset gradients for next calculations
w.grad.zero_()
b.grad.zero_()

# Generate predictions after resetting gradients
preds = model(inputs)
print("\nPredictions after resetting gradients:\n", preds)

# Calculate the loss
loss = mse(preds, targets)
print("\nInitial loss:", loss)

# Compute gradients for the loss
loss.backward()

# Adjust weights and reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

print("\nUpdated weights:\n", w)
print("\nUpdated biases:\n", b)

# Calculate loss after weights update
preds = model(inputs)
loss = mse(preds, targets)
print("\nLoss after initial weight update:", loss)

# Train model for 100 epochs
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()

    # Update weights and reset gradients
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

# Final predictions and loss after training
preds = model(inputs)
loss = mse(preds, targets)
print("\nFinal predictions:\n", preds)
print("\nTargets:\n", targets)
print("\nFinal loss:", loss)
