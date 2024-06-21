
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as F
import torch.nn.init as init



data = np.loadtxt('measurements.csv', delimiter=',')



# Converting input data to PyTorch tensor
x_data = torch.tensor(data[:, 0], dtype=torch.float32).unsqueeze(1)  # Assuming input is in the first column
y_data = torch.tensor(data[:, 1], dtype=torch.float32).unsqueeze(1)  # Assuming target is in the second column



# Splitting data
train_split = int(0.8 * len(x_data))
test_x_split = int(0.2 * len(x_data)) + train_split


train_x,  test_x = x_data[:train_split], x_data[train_split:test_x_split]
train_y,  test_y = y_data[:train_split], y_data[train_split:test_x_split]



# Scatterplotting
plt.figure(figsize=(12, 6))
plt.scatter(train_x, train_y, label='Target')
plt.xlabel('Input')
plt.ylabel('Value')
plt.legend()
plt.show()


# Scatterplotting
plt.figure(figsize=(12, 6))
plt.scatter(test_x, test_y, label='Target')
plt.xlabel('Input')
plt.ylabel('Value')
plt.legend()
plt.show()




# Define Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 100)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)  # use LeakyReLU replace tanh
        x = self.dropout1(x)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)  # useLeakyReLU replace  ReLU
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# Create model
model = NeuralNetwork()


# Define loss function and optimizer
criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# optimizer = optim.SGD(model.parameters(),lr=0.0001, weight_decay=0.001)  # Stochastic Gradient Descent Optimiser

# optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=0.001)  # Adam
optimizer = optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.001)  # RMSprop


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=400, gamma=0.5)

best_validation_loss = float('inf')
best_model_state = None

num_epochs = 2000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_x)
    loss = criterion(outputs, train_y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #
    scheduler.step()

    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}, Learning Rate: {:.6f}'.format(epoch + 1, num_epochs, loss.item(), optimizer.param_groups[0]['lr']))

# Test the model
with torch.no_grad():
    test_outputs = model(test_x)


# Test the best model
with torch.no_grad():
    test_outputs = model(test_x)
    test_outputs_train = model(train_x)




# Scatterplotting
plt.figure(figsize=(12, 6))
plt.scatter(train_x.numpy(), train_y.numpy(), label='Target')
plt.scatter(train_x.numpy(), test_outputs_train.numpy(), label='Predicted (Sorted by Time)')
plt.xlabel('Input')
plt.ylabel('Value')
plt.legend()
plt.show()


# sklearn to split data\]

# Scatterplotting
plt.figure(figsize=(12, 6))
plt.scatter(test_x.numpy(), test_y.numpy(), label='Target')
plt.scatter(test_x.numpy(), test_outputs.numpy(), label='Predicted (Sorted by Time)')
plt.xlabel('Input')
plt.ylabel('Value')
plt.legend()
plt.show()
