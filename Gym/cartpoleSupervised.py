import gym
import keras
import tensorflow as tf
import numpy as np
import random
from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make('CartPole-v0')
initial_games = 10_000
goal_steps = 500
score_requirement = 50
env.reset()

def initial_population():
    training_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        # moves specifically from this environment:
        game_memory = []
        # previous observation that we saw
        prev_observation = []
        # for each frame in 200
        for _ in range(goal_steps):
            # choose random action (0 or 1)
            action = random.randrange(0,2)
            # do it!
            observation, reward, done, _ = env.step(action)
            
            if len(prev_observation) > 0 :
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score+=reward
            if done: break

        if(score >= score_requirement):
            accepted_scores.append(score)
            for data in game_memory:
                # convert to one-hot (this is the output layer for our neural network)
                if data[1] == 1:
                    output = np.array([0,1])
                elif data[1] == 0:
                    output = np.array([1,0])
                    
                # saving our training data
                training_data.append([data[0], output])

        # reset env to play again
        env.reset()
        # save overall scores
        scores.append(score)
    
    # some stats here, to further illustrate the neural network magic!
    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))
    
    return training_data

def createFlattenModel(input_size):
    #Create model
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(input_size,1)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    #Compile model
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    return model

def createNormalModel(input_size):
    #Create model
    model = keras.Sequential([
        keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(input_size,1)),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(256, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.8),
        keras.layers.Dense(2, activation=tf.nn.softmax)
    ])

    #Compile model
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    model.summary()

    return model

def train_model(training_data, model=False):
    
    X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
    y = np.array([i[1] for i in training_data])

    if not model:
        model = createNormalModel(input_size = len(X[0]))

    model.fit(X, y, epochs=2)
    return model

def testModel(model):
    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                output = model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0]
                action = np.argmax(output)

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)

    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(score_requirement)
    return sum(scores)/len(scores)

def newDataset():
    data = initial_population()
    np.save('dataset.npy', data)    # .npy extension is added if not given

newDataset()
data = np.load('dataset.npy', allow_pickle=True)

flattenModel = train_model(data, createFlattenModel(4))
flattenScore = testModel(flattenModel)
