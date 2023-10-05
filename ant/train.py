from game import Game
from agent import DQNAgent, ReplayBuffer
import math

def train():
    # Initialize the game and agent
    game = Game(FPS=60)
    game.init()

    number_of_actions = [0, 2 * math.pi]

    states = game.get_state()

    agent = DQNAgent(input_dims=states, action_range=number_of_actions)

    # Set up other training parameters
    epsilon = 1.0  # Initial exploration rate
    epsilon_min = 0.1  # Minimum exploration rate
    epsilon_decay = 0.995  # Exploration rate decay
    replay_buffer = ReplayBuffer(capacity=10000)
    batch_size = 32
    update_target_network_freq = 100

    num_episodes = 1000  # Number of episodes
    max_steps_per_episode = 1000  # Maximum steps per episode

    for episode in range(num_episodes):
        state = game.get_state()

        for step in range(max_steps_per_episode):
            
            action = agent.select_action(state)

            # Execute the action in the game and observe the next state, reward, and done flag
            next_state, reward, done = game.tick(action)

            # Store the experience in the replay buffer
            replay_buffer.push(state, action, reward, next_state, done)

            # Train the agent
            agent.train(replay_buffer, batch_size)

            # Update the target network periodically
            if step % update_target_network_freq == 0:
                agent.update_target_network()

            # Update the current state
            state = next_state

            # Decay exploration rate
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            if done:
                break

    # Save the trained model?
    # torch.save(agent.q_network.state_dict(), 'trained_agent.pth')

if __name__ == '__main__':
    train()