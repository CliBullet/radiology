import sys
from PIL import Image
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
#from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import BatchNormalization , SpatialDropout2D, Dropout
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
  

#define cnn model using transfer learning
#def define_model():
  #load the model
  #model= VGG16(include_top= False, input_shape= (128,128,3))
  
  #freeze model layers as not trainable
  #for layer in model.layers:
    #layer.trainable= False

model= Sequential()

model.add(Conv2D(32,(3,3), padding= 'same', activation= 'relu', input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))
model.add(SpatialDropout2D(0.1))  #reduces overfitting

model.add(Conv2D(32,(3,3), padding= 'same', activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3), padding= 'same', activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3), padding= 'same', activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3), padding= 'same', activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3), padding= 'same', activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3), padding= 'same', activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.5))

  #create new model on top
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(output_dim=128, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=128, activation= 'relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=1, activation= 'sigmoid'))

#define new model
#model= Model(inputs= model.inputs, outputs= output)
#compile the model
opt= SGD(lr=0.001, momentum = 0.9)
model.compile(optimizer= opt, loss= 'binary_crossentropy', metrics= ['accuracy'])

model.summary()

#Create my classes
test_data= []
test_labels= []

for cond in ['/NORMAL/', 'PNEUMONIA']:
  for img in (os.listdir(input_path + 'test' + cond)):
    img= plt.imread(input_path+ 'test' + cond + img)
    img

datagen= ImageDataGenerator(featurewise_center= True)

#prepare iterator
train_datagen= ImageDataGenerator(
    rescale= 1./255,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True)
  
test_datagen= ImageDataGenerator(rescale= 1./255)

train_it= datagen.flow_from_directory(
    '/content/chest_xray/chest_xray/train',
    target_size= (128,128),
    batch_size= 16,
    class_mode= 'binary')
  
val_it= datagen.flow_from_directory(
    '/content/chest_xray/chest_xray/val',
    target_size= (128,128),
    batch_size= 16,
    class_mode= 'binary')

  
#fit the model
model.fit_generator(train_it, 
                    steps_per_epoch= 326, 
                    epochs=128,
                    validation_data= val_it,
                    validation_steps= 39,
                    verbose= 1)
  
#save the model weights
model.save_weights('/content/pneumonia-xray.h5')

#save the model to json
model_json= model_to_json()
with open('/content/pneumonia_xray.json', 'w') as json_file:
  json_file.write(model_json)

print('Model saved to disk...within Colab')
