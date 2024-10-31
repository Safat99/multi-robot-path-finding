from PyQt6 import QtWidgets, QtCore
from task_grid_selection import launch_task_grid
from multi_robot_env import MultiRobotWarehouseEnv

class RobotTestingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Multi-Robot Warehouse Task Management")

        # Main layout
        main_layout = QtWidgets.QVBoxLayout()

        # Task selection: Random vs Custom
        task_selection_layout = QtWidgets.QHBoxLayout()
        self.random_task_radio = QtWidgets.QRadioButton("Random Tasks")
        self.custom_task_radio = QtWidgets.QRadioButton("Custom Tasks")
        self.custom_task_radio.setChecked(True)  # Default selection
        task_selection_layout.addWidget(self.random_task_radio)
        task_selection_layout.addWidget(self.custom_task_radio)

        # Input fields for the configuration parameters
        config_layout = QtWidgets.QFormLayout()
        self.num_robots_input = QtWidgets.QSpinBox()
        self.num_robots_input.setRange(1, 10)
        self.grid_size_input = QtWidgets.QSpinBox()
        self.grid_size_input.setRange(5, 100)  # Assumes grid between 5x5 and 100x100
        self.num_tasks_input = QtWidgets.QSpinBox()
        self.num_tasks_input.setRange(1, 500)

        config_layout.addRow("Number of Robots:", self.num_robots_input)
        config_layout.addRow("Grid Size (n x n):", self.grid_size_input)
        config_layout.addRow("Number of Tasks:", self.num_tasks_input)

        # Button to start setup based on selected task type
        self.setup_button = QtWidgets.QPushButton("Setup Tasks")
        self.setup_button.clicked.connect(self.setup_tasks) # program starts here

        # Adding widgets to main layout
        main_layout.addLayout(task_selection_layout)
        main_layout.addLayout(config_layout)
        main_layout.addWidget(self.setup_button)

        self.setLayout(main_layout)

    def setup_tasks(self):
        # Retrieve values from input fields
        num_robots = self.num_robots_input.value()
        grid_size = self.grid_size_input.value()
        num_tasks = self.num_tasks_input.value()

        # Determine task setup type
        if self.custom_task_radio.isChecked():
            task_positions, robot_positions = self.start_custom_task_setup(num_robots, grid_size, num_tasks)
            return task_positions, robot_positions
        else:
            self.start_random_task_setup(num_robots, grid_size, num_tasks)
            return None, None

    def start_custom_task_setup(self, num_robots, grid_size, num_tasks):
        print("Custom task setup with PyGame grid...")
        selected_tasks = launch_task_grid(grid_size=grid_size, num_tasks=num_tasks, num_robots=num_robots) # a dictionary with {'tasks': [], 'robots': []}
        
        task_positions = selected_tasks["tasks"]
        robot_positions = selected_tasks["robots"]

        # env = MultiRobotWarehouseEnv(num_robots=num_robots, grid_size=grid_size, task_positions=task_positions, robot_positions=robot_positions)
        # env.render_with_pygame()

        return task_positions, robot_positions

    def start_random_task_setup(self, num_robots, grid_size, num_tasks):
        # Start random task setup here
        print("Random task setup for testing simulation.")
        # Here you can call the simulation logic to generate random tasks
    
    
    def show_report(self, task_data):
    # `task_data` will contain results after task completion (e.g., steps taken, tasks completed)
        report_text = ""
        for robot_id, stats in task_data.items():
            report_text += f"Robot {robot_id}: Tasks Completed: {stats['tasks_completed']}, Steps: {stats['steps']}\n"
        report_msg = QtWidgets.QMessageBox()
        report_msg.setWindowTitle("Simulation Report")
        report_msg.setText(report_text)
        report_msg.exec()
        
        # Save option for storing the report
        save_report = QtWidgets.QMessageBox.question(self, "Save Report", "Do you want to save this report?", 
                                                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if save_report == QtWidgets.QMessageBox.Yes:
            self.save_report_to_file(task_data)
    
    def save_report_to_file(self, task_data):
        with open("simulation_report.txt", "w") as f:
            for robot_id, stats in task_data.items():
                f.write(f"Robot {robot_id}: Tasks Completed: {stats['tasks_completed']}, Steps: {stats['steps']}\n")
        print("Report saved successfully.")

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = RobotTestingApp()
    window.show()
    app.exec()
