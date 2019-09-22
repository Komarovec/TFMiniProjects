# objective is to get the cart to the flag.
# for now, let's just move randomly:

import gym
import numpy as np
import random
import keras
from collections import deque
from keras.layers import Dense
from keras import Sequential
from keras.optimizers import Adam

env = gym.make("MountainCar-v0")

#Q learning
LEARNING_RATE = 0.9
DISCOUNT = 0.95

#AI
BATCH_SIZE = 20
EXPLORATION_RATE_MAX = 1.0
EXPLORATION_RATE_MIN = 0.1
EXPLORATION_RATE_DECAY = 0.999

def createModel(inputs, outputs):
    model = keras.Sequential()
    model.add(Dense(24, input_dim=inputs, activation="relu"))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(outputs, activation="linear"))
    model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    return model

def think(observation, model, exploration):
    if(np.random.random() < exploration):
        return env.action_space.sample()
    else:
        return np.argmax(model.predict(observation)[0])

observation_space = env.observation_space.shape[0]
action_space = env.action_space.n

model = createModel(observation_space, action_space)
exploration_rate = EXPLORATION_RATE_MAX
memory = deque()

run = 0

while True:
    run += 1
    steps = 0
    print("Run: {}".format(run))
    obs = env.reset()
    obs = np.array([obs])

    while True:
        env.render()
        steps += 1
        
        action = think(obs, model, exploration_rate)

        obs1, reward, done, _ = env.step(action)
        obs1 = np.array([obs1])

        reward = reward if not done else -reward
        deltaPos = abs(obs[0][0]-obs1[0][0])
        deltaPos = deltaPos - 0.2

        if(deltaPos < 0):
            deltaPos = -1

        reward = deltaPos

        print(reward)

        memory.append((obs, action, reward, obs1, done))

        obs = obs1


        if(done):
            print("Exploration rate: {}, Steps: {}".format(exploration_rate, steps))
            print("-------------")
            break

        #Experience replay
        if(len(memory) > BATCH_SIZE):
            batch = random.sample(memory, BATCH_SIZE)

            for obs, action, reward, obs1, done in batch:
                q_update = reward
                if not done:
                    q_update = reward + DISCOUNT * np.amax(model.predict(obs1)[0])

                q_values = model.predict(obs)
                q_values[0][action] = q_update

                model.fit(obs, q_values, verbose=0)

            exploration_rate *= EXPLORATION_RATE_DECAY
            exploration_rate = max(EXPLORATION_RATE_MIN, exploration_rate)
                






env.close()