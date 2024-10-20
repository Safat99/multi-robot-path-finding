import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MultiRobotWarehouseEnv(gym.Env):
    def __init__(self, num_robots=3, grid_size=10, max_steps=100):
        super(MultiRobotWarehouseEnv, self).__init__()
        self.num_robots = num_robots
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action and observation space
        self.action_space = spaces.Discrete(8)  # 0-3: Cardinal directions, 4-7: Diagonal moves
        self.observation_space = spaces.Box(low=0, high=grid_size, shape=(num_robots, 2), dtype=np.float32)
        
        # Initialize warehouse and robots
        self.warehouse = np.zeros((grid_size, grid_size))  # Grid representing warehouse
        self.state = np.random.randint(0, grid_size, size=(self.num_robots, 2))  # Robot positions
        self.goal_positions = []  # Store task locations
        self.spawn_tasks()  # Create initial tasks
        
    def spawn_tasks(self):
        """Randomly place tasks in the warehouse."""
        for _ in range(5):  # Create 5 initial tasks
            self.goal_positions.append(np.random.randint(0, self.grid_size, size=(2)))

    def reset(self):
        """Reset the environment."""
        self.state = np.random.randint(0, self.grid_size, size=(self.num_robots, 2))
        self.current_step = 0
        self.goal_positions = []
        self.spawn_tasks()
        return self.state

    def step(self, actions):
        # Implement the logic for each step in the environment
        rewards = []
        done = False

        for i, action in enumerate(actions):
            current_position = self.state[i]
            
            # Update the robot positions based on the action
            if action == 0:  # Move up
                new_position = [current_position[0], min(current_position[1] + 1, self.grid_size - 1)]
            elif action == 1:  # Move down
                new_position = [current_position[0], max(current_position[1] - 1, 0)]
            elif action == 2:  # Move left
                new_position = [max(current_position[0] - 1, 0), current_position[1]]
            elif action == 3:  # Move right
                new_position = [min(current_position[0] + 1, self.grid_size - 1), current_position[1]]
            elif action == 4:  # Move up-right (diagonal)
                new_position = [min(current_position[0] + 1, self.grid_size - 1), min(current_position[1] + 1, self.grid_size - 1)]
            elif action == 5:  # Move up-left (diagonal)
                new_position = [max(current_position[0] - 1, 0), min(current_position[1] + 1, self.grid_size - 1)]
            elif action == 6:  # Move down-right (diagonal)
                new_position = [min(current_position[0] + 1, self.grid_size - 1), max(current_position[1] - 1, 0)]
            elif action == 7:  # Move down-left (diagonal)
                new_position = [max(current_position[0] - 1, 0), max(current_position[1] - 1, 0)]
            
            # Update robot position
            self.state[i] = new_position

            # Check for task pickup
            for task in self.goal_positions:
                if np.array_equal(self.state[i], task):
                    rewards.append(10)  # Reward for picking up an item
                    self.goal_positions.remove(task)
                    self.spawn_new_task()  # Add new task
                else:
                    rewards.append(-1)  # Small negative reward for each step
            
            # Collision avoidance (penalize robots in same location)
            for j in range(i+1, self.num_robots):
                if np.array_equal(self.state[i], self.state[j]):
                    rewards[i] -= 5  # Penalize collisions

        # Check if done (if all tasks are completed or max steps reached)
        self.current_step += 1
        if len(self.goal_positions) == 0 or self.current_step >= self.max_steps:
            done = True
        
        # Calculate reward, next state, and check for done
        return np.array(self.state), np.sum(rewards), done, {}  

    def spawn_new_task(self):
        """Spawn a new task at a random free location in the warehouse."""
        empty_spaces = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.state and (x, y) not in self.goal_positions]
        new_task_location = random.choice(empty_spaces)
        self.goal_positions.append(new_task_location)            
        
    def render_with_pyplot(self, mode='human'):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.grid_size)
        ax.set_ylim(0, self.grid_size)

        robots, = ax.plot([], [], 'bo', ms=10)  # Robots as blue dots
        tasks, = ax.plot([], [], 'rs', ms=10)  # Tasks as red squares

        def init():
            robots.set_data([], [])
            tasks.set_data([], [])
            return robots, tasks

        def update(frame):
            robot_positions = np.array(self.state)
            task_positions = np.array(self.goal_positions)
            
            robots.set_data(robot_positions[:, 0], robot_positions[:, 1])
            tasks.set_data(task_positions[:, 0], task_positions[:, 1])
            
            return robots, tasks

        ani = animation.FuncAnimation(fig, update, init_func=init, frames=self.max_steps, interval=200, blit=True)
        plt.show()