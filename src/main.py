import numpy as np
from multi_robot_env import MultiRobotEnv
from dql_agent import DQLAgent

if __name__ == "__main__":
    num_robots = 2
    env = MultiRobotEnv(num_robots=num_robots)
    state_size = env.observation_space.shape[0] * env.observation_space.shape[1]
    action_size = env.action_space.n

    agent = DQLAgent(state_size, action_size)
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if len(agent.memory) > 32:
                agent.replay(32)
        print(f"Episode {e + 1}/{episodes} finished.")
