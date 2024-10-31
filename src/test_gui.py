from PyQt6 import QtWidgets, QtCore

class RobotTestingApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Multi-Robot Dynamic Testing")

        # Layout and widgets for input
        layout = QtWidgets.QVBoxLayout()

        # Agent count input
        self.num_agents_label = QtWidgets.QLabel("Number of Agents:")
        self.num_agents_input = QtWidgets.QSpinBox()
        self.num_agents_input.setRange(1, 10)
        
        # Task location input
        self.task_loc_label = QtWidgets.QLabel("Task Location (x, y):")
        self.task_x_input = QtWidgets.QSpinBox()
        self.task_y_input = QtWidgets.QSpinBox()
        
        # Drop-off location input
        self.drop_loc_label = QtWidgets.QLabel("Drop-Off Location (x, y):")
        self.drop_x_input = QtWidgets.QSpinBox()
        self.drop_y_input = QtWidgets.QSpinBox()
        
        # Button to start the test
        self.start_button = QtWidgets.QPushButton("Start Test")
        self.start_button.clicked.connect(self.start_test)

        # Add widgets to layout
        layout.addWidget(self.num_agents_label)
        layout.addWidget(self.num_agents_input)
        layout.addWidget(self.task_loc_label)
        layout.addWidget(self.task_x_input)
        layout.addWidget(self.task_y_input)
        layout.addWidget(self.drop_loc_label)
        layout.addWidget(self.drop_x_input)
        layout.addWidget(self.drop_y_input)
        layout.addWidget(self.start_button)
        
        self.setLayout(layout)
    
    def start_test(self):
        num_agents = self.num_agents_input.value()
        task_location = (self.task_x_input.value(), self.task_y_input.value())
        drop_off_location = (self.drop_x_input.value(), self.drop_y_input.value())
        
        # Call the function to run the test with these parameters
        avg_reward, avg_steps = test_dqn_agent(
            checkpoint_path="dqn_model_40.weights.h5",
            num_robots=num_agents,
            task_location=task_location,
            drop_off_location=drop_off_location
        )
        
        # Display the result in a message box or label
        result_message = f"Average Reward: {avg_reward}\nAverage Steps: {avg_steps}"
        QtWidgets.QMessageBox.information(self, "Test Results", result_message)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = RobotTestingApp()
    window.show()
    app.exec()
