from keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from tensorflow.python.framework import ops
ops.reset_default_graph()
import h5py
from PIL import Image
import PIL
frok vb100_utils import *

print('h5py version is {}' .format(h5py__version__))



#Get the architecture of CNN
json_file= open('/content/pneumoneia_xray.json')
loaded_model_json= json_file.read()
json_file.close()
loaded_model= model_from_json(loaded_model_json)

#Get weights into the model
loaded_model.load_weights('/content/pneumonia-xray.h5')

#re-define our optimizer and run
opt= SGD(lr=0.001, momentum= 0.9)
loaded_model.compile(optimizer= opt, loss= 'binary_crossentropy', metrics= ['accuracy'])

#test on validation dataset
image = Image.open('/content/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1427-0001.jpeg')
print(type(image))
image= image.resize((224,224))   #depends on the size of the image
image= np.array(image)
print('po array = {}'.format(image.shape))
image= np.true_divide(image, 255)
image= image.reshape(1, 224, 224, 1)
print(type(image), image.shape)

predictions= loaded_model.predict(image)

print(loaded_model)
predictions_c= loaded_model.predict_classes(image)
print(predictions, predictions_c)

classes= {'train': ['NORMAL', 'PNEUMONIA'],
          'val': ['NORMAL', 'PNEUMONIA'],
          'test': ['NORMAL', 'PNEUMONIA']}

predicted_class= classes['train'][predictions_c[0]]
print('We think this is {}.' .format(predicted_class.lower()))