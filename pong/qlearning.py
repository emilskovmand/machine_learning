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
  def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, num_actions):
    super(DeepQNetwork, self).__init__()
    self.learning_rate = learning_rate
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.num_actions = num_actions

    self.device = T.device("cpu")
    
    # First layer of deep q network
    # Input: self.input_dims - Output: self.fc1_dims
    self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
    # Second layer of deep q network
    # Input: self.fc1_dims - Output: self.fc2_dims
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    # Third layer of deep q network
    # Input: self.fc2_dims - Output: number of actions
    self.fc3 = nn.Linear(self.fc2_dims, self.num_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.loss = nn.MSELoss()
  
  def forward(self, state):
    x = T.relu(self.fc1(state))
    x = T.relu(self.fc2(x))
    action_value = self.fc3(x)
        
    return action_value