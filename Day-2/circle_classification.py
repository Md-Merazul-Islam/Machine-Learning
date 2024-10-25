import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate synthetic data for a circular boundary classification
torch.manual_seed(0)
n_samples = 1000
radius = 1.0

# Random points in the 2D plane
# Points between -1 and 1 for both x and y
x = torch.rand(n_samples, 2) * 2 - 1
# Labels: 1 if outside the circle, 0 if inside the circle
y = (x[:, 0]**2 + x[:, 1]**2 > radius**2).float().reshape(-1, 1)

# Visualize the data
plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), cmap='coolwarm', s=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Points Inside and Outside Circle")
plt.show()

# 2. Define the Neural Network Model


class CircleClassifier(nn.Module):
    def __init__(self):
        super(CircleClassifier, self).__init__()
        self.hidden = nn.Linear(2, 8)   # 2 inputs, 8 hidden neurons
        # 1 output neuron for binary classification
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        # Sigmoid to produce probability between 0 and 1
        x = torch.sigmoid(self.output(x))
        return x


model = CircleClassifier()
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 3. Training the Model
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

# 4. Testing and Visualizing the Results
x1, x2 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
grid = torch.tensor(np.c_[x1.ravel(), x2.ravel()], dtype=torch.float32)
with torch.no_grad():
    pred = model(grid).reshape(100, 100)

plt.contourf(x1, x2, pred, levels=[0, 0.5, 1], alpha=0.3, cmap='coolwarm')
plt.scatter(x[:, 0], x[:, 1], c=y.reshape(-1), cmap='coolwarm', s=10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Decision Boundary")
plt.show()


