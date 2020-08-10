#TF
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Sequential
from keras.preprocessing.image import load_img


MODEL= 'C:/Users/Billionaire-AI/Deep-Learning/static/models/chest-xray.h5'

def getPrediction(filename):
    model = keras.models.load_model(MODEL)
    image = load_img('C:/Users/Billionaire-AI/uploads' +filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    print('%s (%.2f%%)' % (label[0], label[1]*100))
    return label[0], label[1]*100
    