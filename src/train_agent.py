import numpy as np
from multi_robot_env import MultiRobotWarehouseEnv
from dql_agent import DQLAgent
import logging
import os
import re
import threading

logging.basicConfig(level=logging.INFO)

# Disable GPU if necessary (optional)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Uncomment to disable GPU

# # Limit GPU memory usage (optional)
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if physical_devices:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             physical_devices[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])  # Set limit (e.g., 1024 MB)
#     except RuntimeError as e:
#         print(e)

def find_latest_checkpoint():
    # List all files in the current directory and filter for weights files
    weights_files = [f for f in os.listdir() if re.match(r"dqn_model_\d+\.weights\.h5", f)]
    if not weights_files:
        return None  # No checkpoint files available, start fresh

    # Extract episode numbers from file names and find the highest one
    latest_checkpoint = max(weights_files, key=lambda x: int(re.findall(r'\d+', x)[0]))
    return latest_checkpoint


def train_dqn_agent(num_robots=2, episodes=100, batch_size=32):
    logging.info("Starting training...")
    env = MultiRobotWarehouseEnv(num_robots=num_robots)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    # action_size = env.action_space.n
    action_size = 8

    agent = DQLAgent(state_size, action_size)

    for e in range(episodes):
        logging.info(f"Episode {e + 1}/{episodes} starting...")
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(500):  # Set max steps per episode
             # Generate an action for each robot
            actions = [agent.act(state) for _ in range(num_robots)]
            
             # Step the environment with the list of actions
            next_state, reward, done, _ = env.step(actions)
            
            next_state = np.reshape(next_state, [1, state_size])

             # Remember the experience
            agent.remember(state, actions, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            if done:
                agent.update_target_model()  # Update target model periodically
                # print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                logging.info(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
            
            # Perform experience replay once enough samples are collected
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Save the model every 50 episodes
        if e % 50 == 0:
            agent.save(f"dqn_model_{e}.weights.h5")

        # print(f"Episode {e + 1}/{episodes} finished with total reward: {total_reward}")
        logging.info(f"Episode {e + 1}/{episodes} finished with total reward: {total_reward}")


def run_training():
    train_dqn_agent()

def test_dqn_agent(checkpoint_path, num_robots=2, test_episodes=10):
    # Initialize environment and agent
    env = MultiRobotWarehouseEnv(num_robots=num_robots)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = 8
    agent = DQLAgent(state_size, action_size)
    
    # Load the specified checkpoint
    agent.load(checkpoint_path)
    logging.info(f"Testing agent loaded from {checkpoint_path}")
    
    # Run multiple test episodes and calculate the average reward
    total_rewards = []
    for episode in range(test_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0

        for time in range(500):  # Set max steps per episode
            # Act without exploration (set epsilon to 0 for testing)
            agent.epsilon = 0
            actions = [agent.act(state) for _ in range(num_robots)]
            next_state, reward, done, _ = env.step(actions)
            next_state = np.reshape(next_state, [1, state_size])
            
            state = next_state
            episode_reward += reward
            
            if done:
                break

        total_rewards.append(episode_reward)
        logging.info(f"Test Episode: {episode + 1}/{test_episodes}, Reward: {episode_reward}")

    avg_reward = np.mean(total_rewards)
    logging.info(f"Average Reward over {test_episodes} test episodes: {avg_reward}")


if __name__ == "__main__":
    training_thread = threading.Thread(target=run_training)
    training_thread.start()
    training_thread.join()