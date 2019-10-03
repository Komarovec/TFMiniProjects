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
from keras.models import model_from_json
import numpy
import os

env = gym.make("MountainCar-v0")

#Q learning
LEARNING_RATE = 0.001
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
    allReward = 0
    print("Run: {}".format(run))
    obs = env.reset()
    obs = np.array([obs])

    while True:
        env.render()
        steps += 1
        
        action = think(obs, model, exploration_rate)

        obs1, completed, done, _ = env.step(action)
        obs1 = np.array([obs1])

        if(abs(obs1[0][1]) >= 0.01 or (obs1[0][0] > -0.2 or obs[0][0] < -0.7)):
            reward = 1
        else:
            reward = -1

        if(completed == 1):
            reward = 100

            # serialize model to JSON
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            
            print("SOLVED!")
            env.close()

        allReward += reward

        #print(reward)
        #print(obs1)
        #print("---------------")

        memory.append((obs, action, reward, obs1, done))

        obs = obs1


        if(done):
            print("Exploration rate: {}, Steps: {}, Reward: {}".format(exploration_rate, steps, allReward))
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