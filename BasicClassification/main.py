
import random
import numpy as np
import turtle as tt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#Generace dat
posX = []
posY = []
label = []

for i in range(0, 100):
    posX.append(random.randint(-500,500));
    posY.append(random.randint(-500,500));
    if(posY[i] < 0):
        label.append("down")
    else:
        label.append("up")


data = np.array(posX,posY)

print(data.shape)

point1 = (-500, 0)
point2 = (500, 0)


tt.hideturtle()
tt.pencolor("blue")
tt.speed(0)
tt.penup()
tt.goto(point1)
tt.pendown()
tt.goto(point2)
tt.penup()

for i in range(0, 100):
    if(posY[i] < 0):
        tt.pencolor("black")
    else:
        tt.pencolor("red")
    tt.penup()
    tt.setpos([posX[i],posY[i]]);
    tt.pendown()
    tt.circle(2)

tt.exitonclick()

#AI
model = keras.Sequential([ 
    keras.layers.Dense(1, input_shape=(2,), activation=tf.nn.relu),
    keras.layers.Dense(4,activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, label, epochs=5)