import torch
import torch.nn as nn
import torch.optim as optim

class simpleNNet(nn.Module):
  def __init__(self):
    super(simpleNNet, self).__init__()
    self.fc1 = nn.Linear(2,4) #input 
    self.fc2= nn.Linear(4,1) # hidden
    
  def forward(self, x):
    x= torch.relu(self.fc1(x)) # to hidden layer activations
    x = torch.sigmoid(self.fc2(x))   # apply sigmoid activations to ouput layer
    return x

net = simpleNNet()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

inputs = torch.tensor([[0.5,0.8],[0.2,0.4],[0.1,0.9],[0.6,0.3]])
labels = torch.tensor([[1.0],[0.0],[1.0],[0.0]])    

for e in range(100):
  outputs = net(inputs)
  loss = criterion(outputs, labels)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  if e %10 ==0:
    print("epoch {e} , Loss : {loss.item()}")
    