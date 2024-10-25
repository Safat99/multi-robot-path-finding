import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pygame
import random


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
        
        # Task Management
        self.goal_positions = []  # Store task locations
        self.delivery_stations = self._initialize_delivery_stations()  # Delivery locations (stations at the grid edges)
        self.tasks = [] # List to track task as pairs (pick-up, drop-off)
        
        self.spawn_tasks()  # Create initial tasks
        
        pygame.init()
        self.screen_size = 600
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Multi-Robot Warehouse")
        self.clock = pygame.time.Clock()  # To control the frame rate

    def _initialize_delivery_stations(self):
        """Create fixed delivery stations at the edges of the grid."""
        stations = [(0, 0), (0, self.grid_size-1), (self.grid_size-1, 0), (self.grid_size-1, self.grid_size-1)]
        return stations
        
    def spawn_tasks(self):
        """Randomly place tasks in the warehouse."""
        for _ in range(5):  # Create 5 initial tasks
            pick_up = tuple(np.random.randint(0, self.grid_size, size=2))  # Random pick-up location
            drop_off = random.choice(self.delivery_stations)  # Random delivery station
            self.tasks.append((pick_up, drop_off))
            self.goal_positions.append(pick_up)  # Pick-up points are the goals for now

    def reset(self):
        """Reset the environment for a new episode"""
        self.current_step = 0
        self.state = np.random.randint(0, self.grid_size, size=(self.num_robots, 2))
        self.tasks = [] # Clear old tasks
        self.spawn_tasks() # Spawn new tasks
        return self.state

    def step(self, actions):
        """Execute one step in the environment, move robots, and handle task completion."""
        # rewards = []
        self.current_step += 1
        done = False
        
        
        # Move the robots based on actions (same as before, with updated movement)
        for i in range(self.num_robots):
            self._move_robot(i, actions[i])  # Action should be a list of actions for each robot

        # Check if any robot has reached a pick-up location
        for i, robot_pos in enumerate(self.state):
            for task in self.tasks:
                pick_up, drop_off = task
                if tuple(robot_pos) == pick_up:  # Robot reached the pick-up
                    # Now assign this robot the task to deliver the item to the drop-off
                    self.goal_positions[i] = drop_off  # Update goal to the drop-off location
        
        # Check if tasks are completed (robots reached drop-off points)
        for i, robot_pos in enumerate(self.state):
            if tuple(robot_pos) in self.delivery_stations:
                # Task completed for this robot
                done = True  # Assuming one task per episode for simplicity

        # Reward based on task completion and other criteria
        reward = self._calculate_reward()

        # Check if episode should end
        if self.current_step >= self.max_steps or done:
            done = True


        # Check if done (if all tasks are completed or max steps reached)
        if self.current_step >= self.max_steps or done:
            done = True
        
        # Calculate reward, next state, and check for done
        # return self.state, np.sum(rewards), done, {}
        return self.state, reward, done, {}  

    def _move_robot(self, robot_index, action):
        """Move the robot based on the action"""
        current_position = self.state[robot_index]
            
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
        self.state[robot_index] = new_position

    def _calculate_reward(self):
        reward = 0
    
        # Task-related rewards
        for i, robot_pos in enumerate(self.state):
            if tuple(robot_pos) in self.delivery_stations:  # Robot delivered the item
                reward += 10  # Large positive reward for completing the task
            else:
                # Small negative reward for each step taken (time penalty)
                reward -= 1
                
                # Calculate Chebyshev distance to the goal
                goal_pos = self.goal_positions[i]  # Assuming goal position is set for each robot
                chebyshev_distance = max(abs(robot_pos[0] - goal_pos[0]), abs(robot_pos[1] - goal_pos[1]))
            
                # Reward or penalty based on distance to the goal
                reward += 1 / (chebyshev_distance + 1)  # Smaller distances yield higher rewards (closer to the goal)
        
        # Penalize collisions (two robots on the same position)
        for i in range(self.num_robots):
            for j in range(i+1, self.num_robots):
                if np.array_equal(self.state[i], self.state[j]):
                    reward -= 5  # Collision penalty
        
        # Additional conditions can be added here
        return reward
        

    def spawn_new_task(self):
        """Spawn a new task at a random free location in the warehouse."""
        empty_spaces = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if (x, y) not in self.state.tolist() and (x, y) not in self.goal_positions]
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
    
    
    def render_with_pygame(self, mode='human'):
        """Render the robots and tasks using pygame."""
        
        
        cell_size = self.screen_size // self.grid_size
        
        # Colors
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)
        # GREEN = (0, 255, 0)   
        BLUE = (0, 0, 255)

        self.screen.fill(WHITE)
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                pygame.draw.rect(self.screen, BLACK, (x * cell_size, y * cell_size, cell_size, cell_size), 1)  # Draw grid cells

        # Draw robots
        for i, robot_pos in enumerate(self.state):
            pygame.draw.circle(self.screen, BLUE, (robot_pos[0] * cell_size + cell_size // 2, robot_pos[1] * cell_size + cell_size // 2), cell_size // 3)

        # Draw tasks
        for task in self.goal_positions:
            pygame.draw.rect(self.screen, RED, (task[0] * cell_size + 5, task[1] * cell_size + 5, cell_size - 10, cell_size - 10))

        pygame.display.flip()
        
        # Frame rate control
        self.clock.tick(30)  # Limit to 30 frames per second

        # Close the Pygame window when done
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()