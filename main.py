import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

import numpy as np

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


GAME = "CartPole-v0"
EPOCHS = 20
STEPS = 100


def get_model(shape, num_actions):
    model = Sequential()

    model.add(Flatten(input_shape=(1,) + shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))
    model.add(Activation('linear'))

    print(model.summary())


def init(input_shape, num_actions):
    model = get_model(input_shape, num_actions)

    dqn = DQNAgent(
        model=model,
        nb_actions=num_actions,
        memory=SequentialMemory(limit=50000, window_length=1),
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=BoltzmannQPolicy(),
    )

    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    return dqn


def main():
    environment = gym.make(GAME)

    model = init(environment.observation_space.shape, environment.action_space.n)

    model.fit(environment, nb_steps=STEPS, visualize=True, verbose=2)
    model.save_weights('dqn_{}_weights.h5f'.format(GAME), overwrite=True)
    model.test(environment, nb_episodes=EPOCHS, visualize=True)

    environment.close()
