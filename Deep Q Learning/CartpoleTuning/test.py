import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
TARGET_NETWORK_REFRESH = 5
TAU = 0.001
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 16

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.target_network_pass = 0
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()
        self.model.add(Dense(4, input_shape=(observation_space,), activation="relu"))
        self.model.add(Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self, dqn_solver_target):

        if len(self.memory) < BATCH_SIZE:
            return

        self.target_network_pass += 1
        if self.target_network_pass > TARGET_NETWORK_REFRESH:
            self.target_network_pass = 0
            self.update_target_network(dqn_solver_target)

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            # we use the target network to do the predictions
            if not terminal:
                q_update = (reward + GAMMA * np.amax(dqn_solver_target.model.predict(state_next)[0]))
            q_values = dqn_solver_target.model.predict(state)
            ''' action is the index of the selected action from the two possible ones, its value is either 0 or 1
            q_values is an array containing the values of the two possible actions
            we set the q target value of the selected action equals to q_update, according to the bellman equation
            and we will leave the value of the other action as it was predicted on the first pass of the neural network
            this way there will be no difference, therefore no loss, because we only want to rectify the weights fur the action
            that we have chosen'''
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def update_target_network(self, target):
        target.model.set_weights(self.model.get_weights()) 

    def soft_update_target_network(self, target):
        t = target.model.get_weights()
        n = self.model.get_weights()
        print(t)
        for i in range(10000):
            for idx in range(len(t)):
                for idxB in range(len(t[idx])):
                    if(isinstance(t[idx][idxB], (list,np.ndarray))):
                        for idxC in range(len(t[idx][idxB])): # the weights
                            t[idx][idxB][idxC] =  TAU * n[idx][idxB][idxC] + (1 - TAU) * t[idx][idxB][idxC]
                    else: # the biases
                        t[idx][idxB] = TAU * n[idx][idxB] + (1 - TAU) * t[idx][idxB]
        print(t)
        print(n)


def cartpole():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    a = DQNSolver(observation_space, action_space)
    b = DQNSolver(observation_space, action_space)



    a.soft_update_target_network(b)

if __name__ == "__main__":
    cartpole()