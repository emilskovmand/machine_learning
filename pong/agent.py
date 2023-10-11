import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
from collections import namedtuple
from qlearning import DeepQNetwork  # Assuming DeepQNetwork is implemented in qlearning module

class DQNAgent:
    def __init__(self, input_dims, num_actions, learning_rate=0.001, gamma=0.99, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.995):
        self.input_dims = input_dims
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0
        
        self.q_network = DeepQNetwork(learning_rate, input_dims, 8, num_actions)
        self.target_network = DeepQNetwork(learning_rate, input_dims, 8, num_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def select_action(self, state):
        sample = np.random.rand()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if sample < eps_threshold:
            return random.randint(0, self.num_actions - 1)
        else:
            state = T.tensor([state], dtype=T.float32).to(self.q_network.device)
            q_values = self.q_network(state)
            return T.argmax(q_values).item()

    def train(self, replay_buffer, batch_size=32):
        if len(replay_buffer) < batch_size:
            return

        transitions = replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Compute the expected Q-values using the target network
        non_final_mask = T.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=T.bool)
        non_final_next_states = T.tensor([s for s in batch.next_state if s is not None], dtype=T.float32)
        state_batch = T.tensor(batch.state, dtype=T.float32)
        action_batch = T.tensor(batch.action, dtype=T.long).unsqueeze(1)
        reward_batch = T.tensor(batch.reward, dtype=T.float32)


        q_values = self.q_network(state_batch)
        next_q_values = T.zeros(batch_size)
        next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values

        # Compute the Q-values for the selected actions
        q_values = q_values.gather(1, action_batch)

        # Compute the loss
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))

        # Update the Q-network
        self.q_network.optimizer.zero_grad()
        loss.backward()
        self.q_network.optimizer.step()


    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define a Transition named tuple to store experiences
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
