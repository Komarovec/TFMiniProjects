from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
from keras.models import model_from_json
from time import time
import numpy as np
import base64
import tensorflow as tf
import re
import os
import sys 

app = Flask(__name__)

def convertImage(imgData1):
    imgstr = re.search(r'base64,(.*)',str(imgData1)).group(1)
    decodedstring = base64.b64decode(imgstr)

    with open('output.png','wb') as output:
        output.write(decodedstring)

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    imgData = request.get_data()
    convertImage(imgData)
    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x,(28,28))
    x = x.reshape(1,28,28,1)

    with graph.as_default():
        out = model.predict(x)
        print(out)
        print(np.argmax(out,axis=1))
        response = np.array_str(np.argmax(out,axis=1))
        return response	

if __name__ == "__main__":
    global model, graph
    graph = tf.compat.v1.get_default_graph()

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("model.h5")
    print("Loaded model from disk")

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)