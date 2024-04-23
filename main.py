# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# %matplotlib inline
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#preprocess.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam,SGD,RMSprop

# specifically for cnn
from tensorflow.keras.layers import Flatten,Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image

def make_train_data(picture_type,DIR):
    i = 0
    for img in os.listdir(DIR):

        if ".py" not in img:  # don't use file that is not image
            i = i + 1

            label = picture_type
            path = os.path.join(DIR,img)
            img = cv2.imread(path,cv2.IMREAD_COLOR)

            try:
              img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

              X.append(np.array(img))
              Z.append(str(label))
            except:
              break



        if i == 50:
            break

X=[]
Z=[]
IMG_SIZE=300
IN_PICTURE_DIR='./In Picture'
NOT_IN_PICTURE_DIR='./Not In Picture'

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

make_train_data('In Picture', IN_PICTURE_DIR)
print(len(X))

make_train_data('Not In Picture', NOT_IN_PICTURE_DIR)
print(len(X))

#make_train_data('Not In Picture', './pictures')
#print(len(X))

fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Convenience: '+Z[l])

plt.tight_layout()

le=LabelEncoder()
Y=le.fit_transform(Z)

print(X[0])
print(Y[0])

X=np.array(X)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

x_train[0]

# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu', input_shape = (IMG_SIZE,IMG_SIZE,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))

model.add(Activation('relu'))
model.add(Dense(1, activation = "sigmoid"))

batch_size=32
epochs=20

x_train.shape

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

print(epochs)

History = model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)

pred

pred[5]

pred_digits[1]

def predict_with_class(image_number):
  if pred_digits[image_number] == 0:
    print("Not In Picture")
  elif pred_digits[image_number] == 1:
    print("In Picture")
  else:
    print("In Picture")

predict_with_class(13)
