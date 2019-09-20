import gym
import keras
import tensorflow as tf
import numpy as np
import random
from statistics import median, mean
from collections import Counter

env = gym.make('CartPole-v0')
initial_games = 10_000
goal_steps = 300
hm_episodes = 5000

print(env.action_space.n)

#Q learning table
Q = np.zeros([20,env.action_space.n])
lr = .95
y = .8

#Rewards per step
rList = []

def convertToSegment(angle):
    return int(round(angle*10))

for i in range(hm_episodes):
    o = env.reset()

    rAll = 0

    for j in range(goal_steps):
        env.render()
        s = convertToSegment(o[2])
        angle = o[2]
        
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

        o1, r, d, _ = env.step(a)
        s1 = convertToSegment(o1[2])

        r = 1 if abs(angle) < (3.1415/8) else -1

        #print(r)
        Q[s,a] = Q[s,a] + lr*(r + y*np.max(Q[s1,:]) - Q[s,a])

        rAll += r

        o = o1

        if(abs(angle) > (3.1415/4)):
            break

    rList.append(rAll)
    print("Score: {}".format(rAll))

print("Score over time: " + str(sum(rList)/hm_episodes))
