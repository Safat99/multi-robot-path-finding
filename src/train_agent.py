import numpy as np
from multi_robot_env import MultiRobotWarehouseEnv
from dql_agent import DQLAgent

def train_dqn_agent(num_robots=2, episodes=1000, batch_size=10):
    env = MultiRobotWarehouseEnv(num_robots=num_robots)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    # action_size = env.action_space.n
    action_size = 8

    agent = DQLAgent(state_size, action_size)

    for e in range(episodes):
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
                print(f"Episode: {e}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
                break
            
            # Perform experience replay once enough samples are collected
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Save the model every 50 episodes
        if e % 50 == 0:
            agent.save(f"dqn_model_{e}.weights.h5")

        print(f"Episode {e + 1}/{episodes} finished with total reward: {total_reward}")

if __name__ == "__main__":
    train_dqn_agent()