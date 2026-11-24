### why we couldn't test with dynamic number of robots?
>>
if your trained DQN model cannot generalize to a dynamic number of robots, 
the limitation likely lies in how the state space and action space **are defined**.

1. State Space Dependency:<br>
The state space is often defined based on the number of robots (e.g., positions, goals, or observations). 
If the training assumes a fixed number of robots, the model will struggle to adapt to more or fewer robots during testing.

2. Action Space Dependency:<br>
The action space might also be fixed for a set number of robots (e.g., each robot having its own set of actions).
This creates a hard dependency on the number of robots during both training and testing.

### what to do?
>> 
Making the Model Dynamic <br>
To allow your model to handle a **variable number of robots**, we need to decouple the state-action representation from the number of robots. Here's how:

#### 1. Use a Multi-Agent Framework<br>
* Treat each robot as an independent agent with its own policy. Instead of training a single monolithic DQN, you train separate policies that can scale across different numbers of agents.

* Alternatively, use a shared policy (parameter-sharing approach), where one policy is trained and shared across all robots.

#### 2. Modify the State Representation
