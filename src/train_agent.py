import numpy as np
import matplotlib.pyplot as plt
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

def find_latest_checkpoint(checkpoint_path=None):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path
    
    # List all files in the current directory and filter for weights files
    weights_files = [f for f in os.listdir() if re.match(r"dqn_model_\d+\.weights\.h5", f)]
    return max(weights_files, key=lambda x: int(re.findall(r'\d+', x)[0])) if weights_files else None


def train_dqn_agent(num_robots=5, episodes=25, batch_size=32):
    logging.info("Starting training...")
    env = MultiRobotWarehouseEnv(num_robots=num_robots)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    # action_size = env.action_space.n
    action_size = 8

    agent = DQLAgent(state_size, action_size)
    
    # Initialize lists to store metrics
    episode_rewards = []
    episode_epsilons = []

    for e in range(episodes):
        logging.info(f"Episode {e + 1}/{episodes} starting...")
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        total_reward = 0

        for time in range(300):  # Set max steps per episode
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
        
        
        # Append metrics to the lists
        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.epsilon)

        # Save the model every 50 episodes
        if e % 5 == 0:
            agent.save(f"dqn_model_{e}.weights.h5")

        # print(f"Episode {e + 1}/{episodes} finished with total reward: {total_reward}")
        logging.info(f"Episode {e + 1}/{episodes} finished with total reward: {total_reward}")
        
        # Save metrics to a file for later plotting
        np.save("episode_rewards.npy", episode_rewards)
        np.save("episode_epsilons.npy", episode_epsilons)


def plot_training_history():
    # Load saved metrics
    episode_rewards = np.load("episode_rewards.npy")
    episode_epsilons = np.load("episode_epsilons.npy")

    # Plot Episode Rewards
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward per Episode")
    plt.legend()

    # Plot Epsilon Decay
    plt.subplot(1, 2, 2)
    plt.plot(episode_epsilons, label="Epsilon Decay", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay over Episodes")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()

def run_training():
    train_dqn_agent()
    plot_training_history()

def test_dqn_agent(checkpoint_path, num_robots=2, test_episodes=10):
    # Initialize environment and agent
    env = MultiRobotWarehouseEnv(num_robots=num_robots)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = 8
    agent = DQLAgent(state_size, action_size)
    
    # Load the latest or specified checkpoint
    checkpoint_path = find_latest_checkpoint(checkpoint_path)
    if checkpoint_path:
        agent.load(checkpoint_path)
        logging.info(f"Testing agent loaded from {checkpoint_path}")
    else:
        logging.warning("No checkpoint found to load for testing.")
        return
    
     # Metrics storage
    total_rewards, total_steps, success_count = [], [], 0

    # Run multiple test episodes and calculate the average reward
    for episode in range(test_episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        episode_reward = 0
        steps = 0

        for time in range(500):  # Set max steps per episode
            agent.epsilon = 0   # Act without exploration (set epsilon to 0 for testing)
            actions = [agent.act(state) for _ in range(num_robots)]
            next_state, reward, done, _ = env.step(actions)
            next_state = np.reshape(next_state, [1, state_size])
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            if done:
                success_count += 1
                break
        
        # Log episode stats
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        logging.info(f"Test Episode: {episode + 1}/{test_episodes}, Reward: {episode_reward}, Steps: {steps}")

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = success_count / test_episodes
    logging.info(f"Average Reward over {test_episodes} test episodes: {avg_reward}")

    # Save metrics for future analysis
    np.save("test_rewards.npy", total_rewards)
    np.save("test_steps.npy", total_steps)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(total_rewards, label="Reward per Test Episode")
    plt.xlabel("Test Episode")
    plt.ylabel("Total Reward")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(total_steps, label="Steps per Test Episode")
    plt.xlabel("Test Episode")
    plt.ylabel("Steps")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.bar(["Success Rate"], [success_rate * 100], color='g')
    plt.ylabel("Success Rate (%)")

    plt.tight_layout()
    plt.savefig("test_results.png")
    # plt.show()


if __name__ == "__main__":
    # training_thread = threading.Thread(target=run_training)
    # training_thread.start()
    # training_thread.join()
    # plot_training_history()
    test_dqn_agent(checkpoint_path="dqn_model_10.weights.h5", num_robots=5, test_episodes=5)