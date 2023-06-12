#!/usr/bin/env python
# coding: utf-8

# #Importaciones
# 

# In[1]:


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Convolution2D,BatchNormalization
from tensorflow.keras.layers import Flatten,MaxPooling2D,Dropout
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


from sklearn.model_selection import KFold, StratifiedKFold

#import pandas as pd
import numpy as np


# #Hiperparámetros

# In[2]:


img_height = 32
img_width = 32

batch_size = 128
epochs = 150

lr = 0.001

no_classes = 100


# #Optimizadores
# 

# In[3]:


new_adam = Adam(learning_rate=lr)
new_sgd = SGD(learning_rate=lr, momentum=0.9)
old_sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9)


# #Definición de modelo DenseNet 169
# 

# In[4]:


def make_model_densenet169():
    model_d = DenseNet169(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    x = model_d.output

    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)
    x = Dense(1024,activation='relu')(x) 
    x = Dense(512,activation='relu')(x) 
    x = BatchNormalization()(x)
    x = Dropout(0.7)(x)

    preds = Dense(no_classes, activation='softmax')(x) #FC-layer
    
    model = Model(inputs = model_d.input, outputs=preds)
    
    # Compilar el modelo
    model.compile(optimizer=new_adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# #Carga de CIFAR100 y Stratified 10 Fold 

# In[5]:


# import cifar 100 data
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()


# Parse numbers as floats
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#y_train = tf.keras.utils.to_categorical(y_train, num_classes=100)
#y_test = tf.keras.utils.to_categorical(y_test, num_classes=100)

# Normalize data
x_train = x_train / 255
x_test = x_test / 255


# In[6]:


reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, verbose=1, min_lr=0.0001)
# Entrenar el modelo
model = make_model_densenet169()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[reduce_lr])


# #Resultados

# In[7]:


resultados = {'accuracy':[], 'val_accuracy':[], 'loss':[], 'val_loss':[]}
best_model_dir = "chekpoint_densenet_169/model_cifar_100.h5"
model.save(best_model_dir)


# In[ ]:




