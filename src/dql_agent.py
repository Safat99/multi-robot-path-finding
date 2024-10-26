import numpy as np
import tensorflow as tf
from collections import deque

class DQLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = 8 # 8 types movement
        self.memory = deque(maxlen=2000)  # Experience replay memory with a fixed size # a list or a queue
        self.gamma = 0.95  # Discount rate
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()
        
        self.target_model = self._build_model()  # Target model -->
        # Target Model (optional but recommended): Use a separate target model to stabilize learning by updating it 
        # less frequently than the main model.
        self.update_target_model()
        

    def _build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(24, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        """This method stores experiences (called transitions) in the agent's memory for replay later."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """This method determines the next action the agent should take. It uses an epsilon-greedy policy:"""
        if np.random.rand() <= self.epsilon: # Exploration
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state) # Exploitation
        return np.argmax(act_values[0]) # rselect the action with the highest Q-value

    def replay(self, batch_size):
        """This is where the actual learning happens. 
        The agent randomly samples a minibatch of experiences from memory 
        and trains the neural network to update the Q-values."""
        
        if len(self.memory) < batch_size:
            return  # Not enough samples to replay
        
        minibatch = np.random.choice(len(self.memory), batch_size)
        for i in minibatch:
            state, action, reward, next_state, done = self.memory[i]
            target = reward
            if not done:
                target += self.gamma * np.amax(self.target_model.predict(next_state)[0]) # target Q-value
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target model weights with current model weights."""
        self.target_model.set_weights(self.model.get_weights())

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        # Ensure the filename ends with .weights.h5
        if not name.endswith('.weights.h5'):
            name += '.weights.h5'
        self.model.save_weights(name)
