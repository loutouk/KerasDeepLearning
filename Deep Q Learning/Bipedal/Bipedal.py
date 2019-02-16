import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from scores.score_logger import ScoreLogger

ENV_NAME = "BipedalWalker-v2"

def bipedal():
    env = gym.make(ENV_NAME)
    score_logger = ScoreLogger(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.shape[0]
    print(str(observation_space) + " " + str(action_space))

    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        while True:
            step += 1
            # comment to accelerate the learning
            env.render()
            state_next, reward, terminal, info = env.step(env.action_space.sample())#take random action
            reward = reward if not terminal else -reward
            if terminal:
                score_logger.add_score(step, run)
                break


if __name__ == "__main__":
    bipedal()
