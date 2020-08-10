from __future__ import division, print_function
# coding=utf-8
import sys
import os
import re
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import urllib.request
import main
from PIL import Image
#from main import getPrediction

#TF
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import load_img


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, flash, send_from_directory
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__, template_folder='template')
UPLOAD_FOLDER = 'C:/Users/Billionaire-AI/uploads'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH= 'C:/Users/Billionaire-AI/Deep-Learning/static/models/chest-xray.h5'


# Load your trained model
model = keras.models.load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
#print('Model loaded. Start serving...')


def makePredictions(path):
  '''
  Method to predict if the image uploaed is healthy or pneumonic
  '''
  img = Image.open(path) # we open the image
  img_d = img.resize((224,224))
  # we resize the image for the model
  rgbimg=None
  #We check if image is RGB or not
  if len(np.array(img_d).shape)<3:
    rgbimg = Image.new("RGB", img_d.size)
    rgbimg.paste(img_d)
  else:
      rgbimg = img_d
  rgbimg = np.array(rgbimg,dtype=np.float64)
  rgbimg = rgbimg.reshape((1,224,224,3))
  predictions = model.predict(rgbimg)
  a = int(np.argmax(predictions))
  if a==1:
    a = "pneumonic"
  else:
    a="healthy"
  return a

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        if 'img' not in request.files:
            return render_template('index.html',filename="unnamed.png",message="Please upload an file")
        f = request.files['img']  
        if f.filename=='':
            return render_template('index.html',filename="unnamed.png",message="No file selected")
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('index.html',filename="unnamed.png",message="please upload an image with .png or .jpg/.jpeg extension")
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        if len(files)==1:
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        else:
            #files.remove("unnamed.png")
            file_ = files[0]
            #os.remove(app.config['UPLOAD_FOLDER']+'/'+file_)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        predictions = makePredictions(os.path.join(app.config['UPLOAD_FOLDER'],f.filename))
        return render_template('index.html',filename=f.filename,message=predictions,show=True)
    return render_template('index.html',filename='unnamed.png')

if __name__ == '__main__':
	port= int(os.environ.get('PORT', 5000))

	if port == 5000:
		app.debug= True

	app.run(host= '0.0.0.0', port=port)
