from multi_robot_env import MultiRobotWarehouseEnv

env = MultiRobotWarehouseEnv(num_robots=3, grid_size=10)

state = env.reset()
# env.render()

# Take some random actions to see how it behaves
for _ in range(10):  # Simulate 10 steps
    actions = env.action_space.sample()  # Random actions
    next_state, reward, done, _ = env.step([actions] * env.num_robots)  # Same action for all robots
    print("Reward:", reward)
    env.render()
    if done:
        break