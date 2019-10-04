# TensorFlow and tf.keras
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import Sequence
import multiprocessing as mp
import numpy as np

import time

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



class SeqGen(Sequence):
    
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        #print("Fetching batch {}".format(idx))
        time.sleep(0.5)

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x, batch_y


#Ploting functions
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array)
 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

def main():
    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print("Data Info:")
    print("train img shape: "+str(train_images.shape))
    print("Len: "+str(len(train_labels)))
    print("Train Labels: "+str(train_labels))
    print("Test img shape: "+str(test_images.shape))
    print("Len: "+str(len(test_labels)))
    print("--------------------")


    #Normalize data
    train_images = train_images / 255.0

    test_images = test_images / 255.0

    #Create model
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(1024, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(512, activation=tf.nn.sigmoid),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    #Compile model
    model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    workers=mp.cpu_count()
    batch_size = 512

    model.fit_generator(
        generator=SeqGen(train_images, train_labels, batch_size=batch_size),
        epochs=10,
        verbose=1,
        workers=workers
    )

    #Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    #Make predictions
    predictions = model.predict(test_images)
    print("Prediction: "+str(predictions[0]))

    #Plot results
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, test_labels, test_images)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, test_labels)
        plt.show()

if __name__ == "__main__":
    main()