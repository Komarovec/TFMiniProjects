
import tkinter as tk
from tkinter import filedialog
import kivy

kivy.require('1.0.6')
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout

from PIL import Image as img
#AI
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

class Agent():
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __init__(self, **kwargs):
        self.initializeModel()
        self.downloadDataset()

    def initializeModel(self):
        #Create model
        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.sigmoid),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation=tf.nn.sigmoid),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation=tf.nn.sigmoid)
        ])

        #Compile model
        self.model.compile(optimizer='adam', 
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    def downloadDataset(self):
        fashion_mnist = keras.datasets.fashion_mnist

        (self.train_images, self.train_labels), (self.test_images, self.test_labels) = fashion_mnist.load_data()
        
        self.train_images = self.train_images / 255.0
        self.test_images = self.test_images / 255.0

    def printImages(self):
        print(self.test_images[0].shape)

    def train(self):
        self.model.fit(self.train_images, self.train_labels, epochs=10)

    def evaulate(self, img):
        prediction = self.model.predict(img)
        return prediction[0]

class Canvas(GridLayout):
    def __init__(self, **kwargs):
        super(Canvas, self).__init__(**kwargs)
        self.ag = Agent()
        self.ag.printImages()

        self.image = Image(source='image.jpg')
        self.btn = Button(text="Change image")
        self.btn.bind(on_press=self.btnPressed)

        self.btnTF = Button(text="Train")
        self.btnTF.bind(on_press=self.btnTFpressed)

        self.btnEvaulate = Button(text="Evaulate")
        self.btnEvaulate.bind(on_press=self.btnEvaulatePressed)

        self.cols = 2
        self.add_widget(self.btn)
        self.add_widget(self.btnTF)
        self.add_widget(self.btnEvaulate)
        self.add_widget(self.image)

    def btnPressed(self, touch):
        tk.Tk().withdraw()
        in_path = filedialog.askopenfilename()
        if(in_path != ''):
            self.image.source = in_path

    def btnTFpressed(self, touch):
        self.ag.train()

    def btnEvaulatePressed(self, touch):
        imgTest = img.open(self.image.source)
        imgTest = imgTest.resize((28,28), img.ANTIALIAS)
        imgTest = imgTest.convert('L')
        im_array = np.array(imgTest)
        im_array = im_array / 255
        test_array = np.expand_dims(im_array, axis=0)
        print(test_array.shape)
        prediction = self.ag.evaulate(test_array)
        predicted_label = np.argmax(prediction)
        print(self.ag.class_names[predicted_label])

class PicturesApp(App):
    def build(self):
        return Canvas()

if __name__ == '__main__':
    PicturesApp().run()