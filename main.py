import os

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from gym_runner import GymRunner
from q_learning_agent import QLearningAgent

import highway_env
from highway_env.envs.merge_env1 import *

class MergeHighwayAgent(QLearningAgent):
    def __init__(self):
        super().__init__(4, 2)

    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=90))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

        # load the weights of the model if reusing previous training session
        # model.load_weights("models/highway_env_merge-v1.h5")
        return model


if __name__ == "__main__":
    #gym = GymRunner('CartPole-v1', 'gymresults/cartpole-v1')
    gym = GymRunner('merge-v1', 'gymresults/highway_env_merge-v1')
    agent = MergeHighwayAgent()

    gym.train(agent, 5000)
    gym.run(agent, 500)

    agent.model.save_weights("models/highway_env_merge-v1.h5", overwrite=True)
    gym.close()
