import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import initializers 

from scores.score_logger import ScoreLogger

ENV_NAME = "CartPole-v1"

TARGET_NETWORK_REFRESH = 25

GAMMA = 0.99
LEARNING_RATE = 0.0005

MEMORY_SIZE = 10000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999


class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.target_network_pass = 0
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = Sequential()   
        self.model.add(Dense(200, input_shape=(observation_space,), activation="tanh", kernel_initializer=initializers.random_normal()))
        self.model.add(Dense(200, activation="tanh", kernel_initializer=initializers.random_normal()))
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
            #print("update target network")
            self.update_target_network(dqn_solver_target)

        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            # get the best action according to the basic q network

            q_index = (self.model.predict(state_next)[0])
            '''print("qindex")
            print(q_index)
            print(q_index.argmax())'''
            q_index = q_index.argmax()
            # we use the target network to do the predictions
            if not terminal:
                '''print("q target values")
                print(dqn_solver_target.model.predict(state_next)[0])
                print(dqn_solver_target.model.predict(state_next)[0][q_index])'''
                q_update = (reward + GAMMA * dqn_solver_target.model.predict(state_next)[0][q_index])
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


def cartpole():
    env = gym.make(ENV_NAME)
    #score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    dqn_solver_target = DQNSolver(observation_space, action_space)
    dqn_solver_target.model.set_weights(dqn_solver.model.get_weights())

    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(step))
                #score_logger.add_score(step, run)
                break
            dqn_solver.experience_replay(dqn_solver_target)

if __name__ == "__main__":
    cartpole()