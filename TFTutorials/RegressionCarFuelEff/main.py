from __future__ import absolute_import, division, print_function

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Verze TF
print(tf.__version__)

#Nastaveni cesty k datasetu
dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print(dataset_path)

#Nastaveni sloupcu a lokalniho datasetu
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

#Stazeni datasetu
dataset = raw_dataset.copy()
print(str(dataset.tail()))
print(str(dataset.isna().sum()))

#Smazani zaznamu ktere obasuji necitelne data
dataset = dataset.dropna()

#Oddeleni mista puvodu v ciselna podobe
origin = dataset.pop('Origin')

#Dosazeni za misto puvodu --> novy sloupec --> je ze zeme 1 --> neni 0
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
print(str(dataset.tail()))

#Rozdeleni datasetu 80% Train; 20% Test
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Vykresleni --> zatim nefunguje );
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

train_stats = train_dataset.describe()
print(str(train_stats.pop("MPG")))
train_stats = train_stats.transpose()
print(str(train_stats))

#Oddeleni ucicích hodnot (MPG/Ucinost) od parametru / vlastnosti auta
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#Funkce na normalizaci dat --> Zlepsuje efektivitu modelu a dela je standartni
def norm(x):
      return (x - train_stats['mean']) / train_stats['std']

#Normalizace
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


#Funkce pro vytvoreni modelu
def build_model():
    #Sekvenční model
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    #Optimizer --> Zatim nemam tuseni.. ZATIM
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    #Vytvoreni a return modelu
    model.compile(loss='mean_squared_error',
        optimizer=optimizer,
        metrics=['mean_absolute_error', 'mean_squared_error'])
    return model

#Vytvoreni modelu
model = build_model()

#Vypis specifikaci modelu
model.summary()

#Test pro overeni vstupu/vystupu modelu
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(str(example_result))

#Kazda epocha jedna tecka
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

#Pocet ucicích epoch
EPOCHS = 1000

#Ukládej pri uceni vse do history objektu
history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

#Vypsani history objektu
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#Funkce pro vykresleni history objektu
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
            label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
            label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
            label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()


plot_history(history)

#Vytvoreni dalsiho modelu
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)