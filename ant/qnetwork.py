import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np

# ! A Machine-learning algorithm
# Backpropagation = https://en.wikipedia.org/wiki/Backpropagation

# Q Learning is kinda like linear regression 
# - tries to fit a line to the delta between the target value and the output of the neural network 

class DeepQNetwork(nn.Module):
  # lr = learning rate
  # input_dims = input dimensions
  # n_actions = number of actions
  #
  def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, action_range):
    super(DeepQNetwork, self).__init__()
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_fims = fc2_dims
    self.action_range = action_range

    self.device = T.device("cpu")
    
    # First layer of deep q network
    # Input: self.input_dims - Output: self.fc1_dims
    self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
    # Second layer of deep q network
    # Input: self.fc1_dims - Output: self.fc2_dims
    self.fc2 = nn.Linear(*self.fc1_dims, self.fc2_fims)
    # Third layer of deep q network
    # Input: self.fc2_dims - Output: number of actions
    self.fc3 = nn.Linear(*self.fc2_dims, 1)

    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.loss = nn.MSELoss()
  
  def forward(self, state):
    x = T.relu(self.fc1(state))
    x = T.relu(self.fc2(x))
    action_value = self.fc3(x)
        
    # Ensure the output action value is within the specified range
    action_value = T.sigmoid(action_value)  # Transform to [0, 1]
    action_value = action_value * (self.action_range[1] - self.action_range[0]) + self.action_range[0]
        
    return action_value
