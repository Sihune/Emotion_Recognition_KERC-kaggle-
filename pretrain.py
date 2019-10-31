import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

from model_vggface2 import Vggface2_ResNet50
from vgg16Model import build_vgg16
from keras.callbacks import ModelCheckpoint,TensorBoard, EarlyStopping
from resnetModel import ResnetBuilder
import os.path

import time
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

nb_classes = 7

trainingset = np.loadtxt('./pretrainingData/fer2013_training_onehot.csv', delimiter=',')
testingset = np.loadtxt('./pretrainingData/fer2013_publictest_onehot.csv', delimiter=',')

n_inputs = 2304
n_classes = 7
img_dim = 48

x_training = trainingset[:, 0:n_inputs]
y_training = trainingset[:, n_inputs:n_inputs + n_classes]

x_testing = testingset[:, 0:n_inputs]
y_testing = testingset[:, n_inputs:n_inputs + n_classes]

x_training = x_training.reshape(x_training.shape[0], 48, 48)
x_training = np.expand_dims(x_training, axis=4)

x_testing = x_testing.reshape(x_testing.shape[0], 48, 48)
x_testing = np.expand_dims(x_testing, axis=4)

model = ResnetBuilder.build_resnet_152((1, 48, 48), nb_classes)
#model = Vggface2_ResNet50(input_dim = (48,48,1), nb_classes = nb_classes, optimizer='adam')

#input_image = Input(shape = (48,48,1), name='input')
#model = build_vgg16(input_image, nb_classes)

model.summary()

checkpointer = ModelCheckpoint(filepath='./pre_trained/weights.h5',
                               monitor='val_loss',
                               verbose = 1,
                               save_best_only=True)

early_stopper = EarlyStopping(patience = 100)

loss = 'categorical_crossentropy'
lr = 0.000001
model.compile(loss = loss, optimizer = Adam(lr = lr, beta_1=0.9, beta_2 = 0.999),metrics=['accuracy'])

batch_size = 128
epochs = 1000

model.fit(x_training, y_training,validation_data=(x_testing,y_testing),
          batch_size = batch_size,
          epochs = epochs,
          verbose=1,
          callbacks = [checkpointer,early_stopper])




