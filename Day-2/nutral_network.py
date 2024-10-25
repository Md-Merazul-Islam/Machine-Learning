import torch
import torch.nn as nn # for creating class
import torch.optim as optim # for optimization


x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y= torch.tensor([[0],[1],[1],[0]],dtype=torch.float32)

# print(x)

class SimpleNN(nn.Module):
  def __init__(self):
    super (SimpleNN, self).__init__()
    self.hidden=nn.Linear(2,4) # 2 input and 4 hider neuron
    self.output = nn.Linear(4,1) # 1 output neuron
  
  def forward(self,x):
    x = torch.relu(self.hidden(x)) # when neuron activate
    x = torch.sigmoid(self.output(x)) # when neuron output 
    return x
  
  
    
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)

epochs = 1000

for epoch in range(epochs):
  optimizer.zero_grad() # reset gradients
  outputs=model(x)
  loss = criterion(outputs,y)
  loss.backward() # following loss update model
  optimizer.step()
  if (epoch+1)% 100==0:
    print(f'epoch: {epoch+1}/{epochs}, loss: {loss.item():.4f}')
    
    
print("Final Outputs:", model(x).detach().numpy())