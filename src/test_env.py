from multi_robot_env import MultiRobotWarehouseEnv
import time
import pygame

env = MultiRobotWarehouseEnv(num_robots=3, grid_size=10)

# state = env.reset()
# env.render_with_pyplot()

# Take some random actions to see how it behaves
for _ in range(20):  # Simulate 10 steps
    actions = env.action_space.sample()  # Random actions --> returns an integer
    next_state, reward, done, _ = env.step([actions] * env.num_robots)  # Same action for all robots
    print("State:", next_state)
    print("Reward:", reward)
    print()
    # env.render_with_pyplot()
    env.render_with_pygame()
    
    pygame.time.delay(500)
    
    if done:
        break