import numpy as np
import os
import matplotlib.pyplot as plt
from imutils import paths
from keras.utils import to_categorical    #cant use label encoder cause it gives hierachial order

import tensorflow as tf
import keras
#facemask detection model using transfer learning method using mobilenetv2 model
from keras.applications import MobileNetV2                     #readymade model
from keras.applications.mobilenet_v2 import preprocess_input    #convert your image to the format the model requires
from keras.models import Model
from keras.layers import Input 
from keras.layers import AveragePooling2D         
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout                  # randomly drop units in layers to prevent overfitting
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator      #process image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model

dataset = r'C:/................/mask_dataset'

imagepaths= list(paths.list_images(dataset))

labels=[]
data=[]
for i in imagepaths:
    label=i.split(os.path.sep)[-2]
    labels.append(label)
    image=load_img(i,target_size=(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    data.append(image)

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

data= np.array(data,dtype='float32')
labels= np.array(labels)
xtrain,xtest,ytrain,ytest= train_test_split(data,labels,test_size=0.2,random_state=10,stratify=labels)

labels.shape
data.shape
xtrain.shape
ytrain.shape

aug=ImageDataGenerator(rotation_range=20,zoom_range=0.2,width_shift_range=0.15,height_shift_range=0.15,shear_range=0.15,horizontal_flip=True,vertical_flip=True,fill_mode='nearest')     

basemodel=MobileNetV2(weights='imagenet',include_top=False,input_tensor=Input(shape=(224,224,3)))
basemodel.summary()

headmodel=basemodel.output
headmodel=AveragePooling2D(pool_size=(7,7))(headmodel)
headmodel=Flatten(name='Flatten')(headmodel)
headmodel=Dense(128,activation='relu')(headmodel)
headmodel=Dropout(0.5)(headmodel)
headmodel=Dense(2,activation='softmax')(headmodel)

model=Model(inputs=basemodel.input,outputs=headmodel)
model.summary()


for layer in basemodel.layers:
    layer.trainable=False   
model.summary()


learning_rate=0.001
Epochs=20
bs=12

opt=Adam(lr=learning_rate,decay=learning_rate/Epochs)
model.compile(loss='binary_crossentropy',optimizer=opt,metrics=['accuracy'])

H=model.fit(aug.flow(xtrain,ytrain,batch_size=bs),steps_per_epoch=len(xtrain)//bs,validation_data=(xtest,ytest),validation_steps=len(xtest)//bs,epochs=Epochs)

model.save(r'C:/..................../mask.model',save_format='h5')

predict=model.predict(xtest,batch_size=bs)
predict=np.argmax(predict,axis=1)
print(classification_report(ytest.argmax(axis=1),predict,target_names=lb.classes_))

from sklearn.metrics import accuracy_score
predict=to_categorical(predict)
accuracy_score(ytest,predict)

# plot the training loss and accuracy

N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="best")
#plt.savefig(r'plot_v2.png')

















































