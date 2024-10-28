import numpy as np
from multi_robot_env import MultiRobotWarehouseEnv
from dql_agent import DQLAgent
# import tensorflow as tf
import logging
#import os
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

if __name__ == "__main__":
    training_thread = threading.Thread(target=run_training)
    training_thread.start()
    training_thread.join()