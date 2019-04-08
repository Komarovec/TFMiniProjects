
import random
import numpy as np
import turtle as tt
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

#Generace dat
def generateData(val):
    data = np.empty((0,2), int)
    for i in range(0, val):
        posX = random.randint(-500,500)
        posY = random.randint(-500,500)
        data = np.append(data, [[posX,posY]], axis=0)
    return data

#Testovani dat
def testData(data):
    label = np.empty((0,1), int)
    for i in data:
        if(i[1] < 0):
            label = np.append(label, 0)
        else:
            label = np.append(label, 1)
    return label

#Plotting
def plotData(data, label):
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
        if(label[i] == 1):
            tt.pencolor("black")
        else:
            tt.pencolor("red")
        tt.penup()
        tt.setpos(data[i]);
        tt.pendown()
        tt.circle(2)


#Main
train_data = generateData(50000)
train_label = testData(train_data)

test_data = generateData(10000)
test_label = testData(test_data)

print("Data shape:")
print("Data shape: "+str(train_data.shape))
print("Data: "+str(train_data))
print("Index: "+str(train_data[0,0]))
print("----------------")

plotData(test_data, test_label)

#AI
model = keras.Sequential([ 
    keras.layers.Dense(32, input_shape=(2,)),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, train_label, epochs=5)

test_loss, test_acc = model.evaluate(test_data, test_label)
print('Test accuracy:', test_acc)

predictions = model.predict(test_data)
print("Prediction: "+str(predictions[0].shape))

predictionsTest = []
for i in predictions:
    predictionsTest.append(i[0])

tt.clear()
plotData(test_data, predictionsTest)
tt.exitonclick()