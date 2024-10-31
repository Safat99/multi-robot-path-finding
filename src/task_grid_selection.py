import pygame
import sys

class TaskGridSelection:
    def __init__(self, grid_size, num_tasks, num_robots, cell_size=50):
        pygame.init()
        self.grid_size = grid_size
        self.num_tasks = num_tasks
        self.num_robots = num_robots
        self.cell_size = cell_size

        self.width = grid_size * cell_size
        self.height = grid_size * cell_size + 2 * cell_size
        self.sidebar_width = 200
        self.screen = pygame.display.set_mode((self.width + self.sidebar_width, self.height))
        
        pygame.display.set_caption("Select Task and Robot Locations")
        self.selected_cells = []
        self.robot_positions = []
        self.font = pygame.font.Font(None, 24)
        
        # Track the current selection phase ('tasks' or 'robots')
        self.selection_phase = 'tasks'

    def run(self):
        running = True
        proceed_button_rect = None
        finish_button_rect = None

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    
                    # Check if "Proceed" or "Finish" button is clicked
                    if self.selection_phase == 'tasks':
                        if proceed_button_rect and proceed_button_rect.collidepoint(pos) and len(self.selected_cells) == self.num_tasks:
                            self.selection_phase = 'robots'
                    elif self.selection_phase == 'robots':
                        if finish_button_rect and finish_button_rect.collidepoint(pos) and len(self.robot_positions) == self.num_robots:
                            running = False
                    
                    # Convert mouse position to grid coordinates
                    grid_x, grid_y = pos[0] // self.cell_size, pos[1] // self.cell_size

                    # Toggle task or robot selection based on the current phase
                    if pos[0] < self.width:  # Inside the grid area
                        if self.selection_phase == 'tasks':
                            if (grid_x, grid_y) in self.selected_cells:
                                self.selected_cells.remove((grid_x, grid_y))
                            elif len(self.selected_cells) < self.num_tasks:
                                self.selected_cells.append((grid_x, grid_y))
                        elif self.selection_phase == 'robots':
                            if (grid_x, grid_y) in self.robot_positions:
                                self.robot_positions.remove((grid_x, grid_y))
                            elif len(self.robot_positions) < self.num_robots and (grid_x, grid_y) not in self.selected_cells:
                                self.robot_positions.append((grid_x, grid_y))

            # Draw grid, sidebar, and buttons
            self._draw_grid()
            if self.selection_phase == 'tasks':
                proceed_button_rect = self._draw_sidebar("Tasks remaining: ", self.num_tasks - len(self.selected_cells), "Proceed", (0, 128, 0))
            elif self.selection_phase == 'robots':
                finish_button_rect = self._draw_sidebar("Robots remaining: ", self.num_robots - len(self.robot_positions), "Finish", (128, 0, 0))
            pygame.display.flip()

        pygame.quit()
        
        # Return positions if setup was completed
        if self.selection_phase == 'robots':
            return {"tasks": self.selected_cells, "robots": self.robot_positions}
        else:
            return None
    
    def _draw_grid(self):
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid cells
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                if (x, y) in self.selected_cells:
                    color = (0, 255, 0)  # Task cells in green
                    pygame.draw.rect(self.screen, color, rect)
                elif (x, y) in self.robot_positions:
                    color = (0, 0, 255)  # Robot cells in blue
                    # Draw a circle at the center of the cell for the robot
                    robot_center = (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2)
                    pygame.draw.circle(self.screen, color, robot_center, self.cell_size // 3)
                else:
                    color = (200, 200, 200)  # Regular cells in light gray
                    pygame.draw.rect(self.screen, color, rect, 1)  # Draw outline for empty cells

    def _draw_sidebar(self, text, count, button_label, button_color):
        # Sidebar background
        pygame.draw.rect(self.screen, (240, 240, 240), (self.width, 0, self.sidebar_width, self.height))

        # Task or robot count display
        text_surface = self.font.render(f"{text} {count}", True, (0, 0, 0))
        self.screen.blit(text_surface, (self.width + 10, 10))

        # Draw button
        button_rect = pygame.Rect(self.width + 50, self.height - 80, 100, 40)
        pygame.draw.rect(self.screen, button_color, button_rect)
        button_text = self.font.render(button_label, True, (255, 255, 255))
        self.screen.blit(button_text, (self.width + 65, self.height - 70))

        return button_rect
    


# To be called from PyQt setup function
def launch_task_grid(grid_size, num_tasks, num_robots):
    grid = TaskGridSelection(grid_size, num_tasks, num_robots) # initialize for user setup
    selected_tasks = grid.run()
    print(f"Selected Tasks: {selected_tasks}")
    return selected_tasks
