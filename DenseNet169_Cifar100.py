#!/usr/bin/env python
# coding: utf-8

# #Importaciones
# 

# In[9]:


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

# In[4]:


img_height = 32
img_width = 32

batch_size = 128
epochs = 150

lr = 0.005

no_classes = 100


# #Optimizadores
# 

# In[6]:


new_adam = Adam(learning_rate=2e-5)
new_sgd = SGD(learning_rate=lr, momentum=0.9)
old_sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum = 0.9)


# #Definición de modelo DenseNet 169
# 

# In[7]:


def make_model_densenet169(weights=None):
    model_d = DenseNet169(weights=weights, include_top=False, input_shape=(32, 32, 3))

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
    model.compile(optimizer=old_sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model


# #Carga de CIFAR100 y Stratified 10 Fold 

# In[8]:


# import cifar 100 data
# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

#print('x_train shape:', x_train.shape)
#print('y_train shape:', y_train.shape)
#print('x_test shape:', x_test.shape)
#print('y_test shape:', y_test.shape)
# Parse numbers as floats
input_train = x_train.astype('float32')
input_test = x_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255


# In[11]:


#Merge inputs and targets

inputs = np.concatenate((x_train,x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis = 0)


# In[12]:


#Define the K-fold cross validator
kfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 123)


# #Resultados

# In[13]:


resultados = {'accuracy':[], 'val_accuracy':[], 'loss':[], 'val_loss':[]}


# #Stratified 10 Fold Cross Validation

# In[14]:


fold_no = 1
for train,test in kfold.split(inputs,targets):
  #Configurar conjunto de datos para rendimiento
  #AUTOTUNE = tf.data.AUTOTUNE
  #train = train.cache().prefetch(buffer_size = AUTOTUNE)
  #test = test.cache().prefetch(buffer_size = AUTOTUNE)

  #Crear modelo
  model = make_model_densenet169()
  model._name = "Model_"+str(fold_no)
  # Generate a print
  print('------------------------------------------------------------------------')
  print(f'Training for fold {fold_no} ...')

  #ReduceLROnPlateau
  anne = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=5, verbose=1, min_lr=1e-3)

  # Save best model Checkpoint
  best_model_dir = "chekpoint_densenet_169/model_cifar_"+str(fold_no)+".h5"
  print("DIR Save Model:", best_model_dir)
  checkpoint = ModelCheckpoint(best_model_dir, verbose=1, save_best_only=True)

  #Entrenar el modelo
  history = model.fit(inputs[train], targets[train], validation_data= (inputs[test], targets[test]),epochs=epochs, batch_size=batch_size, callbacks=[anne])

  # Guardar el modelo / (pesos)
  model.save(best_model_dir)

# Guardar resultados
  resultados['accuracy'].append(history.history['accuracy'])
  resultados['val_accuracy'].append(history.history['val_accuracy'])
  resultados['loss'].append(history.history['loss'])
  resultados['val_loss'].append(history.history['val_loss'])

  model = None


  fold_no = fold_no + 1


# #Visualizar Datos

# In[1]:


from matplotlib import pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[21]:





# In[ ]:




