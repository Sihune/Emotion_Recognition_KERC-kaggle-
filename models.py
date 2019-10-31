"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Reshape, Dense, Flatten, Dropout, ZeroPadding3D, Conv3D, MaxPool3D, BatchNormalization, Input,Convolution3D,Activation,GlobalAveragePooling3D
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, RMSprop , SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from keras.losses import categorical_crossentropy
from collections import deque
import sys

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model=None, features_length=25088):
        """
        `model` = one of:
            lstm
            lrcn
            mlp
            conv_3d
            c3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        self.load_model = load_model
        self.saved_model = saved_model
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        #print(self.input_shape.shape)

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']
        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')

        # Get the appropriate model.
        if model == 'lrcn':
            print("Loading CNN-LSTM model.")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.lrcn()
        elif model == 'mlp':
            print("Loading simple MLP.")
            self.input_shape = (seq_length, features_length)
            self.model = self.mlp()
        elif model == 'conv3D':
            print("Loading conv3D model.")
            self.input_shape = saved_model.input_shape#(seq_length, 64, 64, 3)
            self.model = self.conv3D()
        elif model == 'vgg_16':
            print("Loading vgg_16 model.")
            self.input_shape = (seq_length, 80, 80, 3)
            self.model = self.vgg_16()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1e-6, decay=1e-6)
        #optimizer = SGD(lr=1e-3, decay=1e-6)
        self.model.compile(loss= 'categorical_crossentropy', optimizer=optimizer, metrics=metrics)

        print(self.model.summary())




    def mlp(self):
        """Build a simple MLP. It uses extracted features as the input
        because of the otherwise too-high dimensionality."""
        # Model.
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.5))
        #model.add(Dense(512))
        #model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))

        return model
    
    def lrcn(self):

        """Build a CNN into RNN.
        Starting version from:
            https://github.com/udacity/self-driving-car/blob/master/
                steering-models/community-models/chauffeur/models.py

        Heavily influenced by VGG-16:
            https://arxiv.org/abs/1409.1556

        Also known as an LRCN:
            https://arxiv.org/pdf/1411.4389.pdf\\
            """

        print(self.input_shape)

        model = Sequential()

        model.add(TimeDistributed(Conv2D(32, (7, 7), strides=(2, 2),
                                         activation='relu', padding='same'), input_shape=self.input_shape))
        model.add(TimeDistributed(Conv2D(32, (3, 3),
                                         kernel_initializer="he_normal", activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(64, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(128, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(128, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(256, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(256, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Conv2D(512, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(Conv2D(512, (3, 3),
                                         padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2))))

        model.add(TimeDistributed(Flatten()))

        model.add(Dropout(0.5))
        model.add(LSTM(256, return_sequences=False, dropout=0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        return model


    def vgg_16(self):

        fc_dropout = [0.1, 0.0, 0.0]
        fc_finals = [2048, 0]

        #print(self.saved_model)

        model = Sequential()
        model.add(TimeDistributed(self.saved_model, input_shape=self.saved_model.input_shape))  # (batch_size, frames, features)
        model.add(TimeDistributed(Dense(1024)))  # (batch_size, frames, features)
        model.add(TimeDistributed(Reshape((32, 32,1))))

        #print(model)
        model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
        model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
        #model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        #model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal", activation='relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2),strides=(2, 2, 2),padding='valid'))
        model.add(Dropout(rate=0.25))

        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(128, (3, 3, 3), padding='same', activation='relu'))
        #model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2),strides=(2, 2, 2),padding='valid'))
        model.add(Dropout(rate=0.25))

        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(256, (3, 3, 3), padding='same', activation='relu'))
        #model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        #model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2),strides=(2, 2, 2),padding='valid'))
        model.add(Dropout(rate=0.25))


        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        #model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        #model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2),strides=(2, 2, 2),padding='valid'))
        model.add(Dropout(rate=0.25))

        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        model.add(Conv3D(512, (3, 3, 3), padding='same', activation='relu'))
        #model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        #model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling3D(pool_size = (2, 2, 2),strides=(2, 2, 2),padding='valid'))
        model.add(Dropout(rate=0.25))


        if fc_dropout[0] > 0: model.add(Dropout(fc_dropout[0]))
        if fc_finals[0] > 0: model.add(Dense(fc_finals[0], activation='relu', name='cnn3d_fc1'))
        if fc_dropout[1] > 0: model.add(Dropout(fc_dropout[1]))
        if fc_finals[1] > 0: model.add(Dense(fc_finals[1], activation='relu', name='cnn3d_fc2'))
        if fc_dropout[2] > 0: model.add(Dropout(fc_dropout[2]))
        model.add(Dense(self.nb_classes, activation='softmax', name='cnn3d_predictions'))

        return model

    def conv3D(self):

        #self.saved_model.input_shape = self.saved_model.input_shape.reshape(20,64,64,3)
        print(self.saved_model.input_shape)
        #self.saved_model.input_shape = Reshape((self.seq_length, 224,224,3))

        fc_dropout = [0.1, 0.0, 0.0]
        fc_finals = [2048, 0]

        model = Sequential()
        model.add(TimeDistributed(self.saved_model, input_shape=self.saved_model.input_shape))  # (batch_size, frames, features)
        model.add(TimeDistributed(Dense(1024)))  # (batch_size, frames, features)
        model.add(TimeDistributed(Reshape((32, 32, 1))))  # (batch_size, frames, features)

        # 1st layer group
        model.add(Convolution3D(64, (3, 3, 3), padding='same', name='cnn3d_conv1_1', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(64, (3, 3, 3), padding='same', name='cnn3d_conv1_2', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='cnn3d_pool1'))
        # 2nd layer group
        model.add(Convolution3D(128, (3, 3, 3), padding='same', name='cnn3d_conv2_1', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(128, (3, 3, 3), padding='same', name='cnn3d_conv2_2', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='cnn3d_pool2'))

        # 3nd layer group
        model.add(Convolution3D(256, (3, 3, 3), padding='same', name='cnn3d_conv3_1', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(256, (3, 3, 3), padding='same', name='cnn3d_conv3_2', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(256, (3, 3, 3), padding='same', name='cnn3d_conv3_3', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='cnn3d_pool3'))
        #model.add(GlobalAveragePooling3D())

        # 4nd layer group
        model.add(Convolution3D(512, (3, 3, 3), padding='same', name='cnn3d_conv4_1', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(512, (3, 3, 3), padding='same', name='cnn3d_conv4_2', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(512, (3, 3, 3), padding='same', name='cnn3d_conv4_3', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='cnn3d_pool4'))

        # 5nd layer group
        model.add(Convolution3D(512, (3, 3, 3), padding='same', name='cnn3d_conv5_1', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(512, (3, 3, 3), padding='same', name='cnn3d_conv5_2', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution3D(512, (3, 3, 3), padding='same', name='cnn3d_conv5_3', strides=(1, 1, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='cnn3d_pool5'))
        model.add(GlobalAveragePooling3D())

        # model.add(Flatten())
        # FC layers group
        if fc_dropout[0] > 0: model.add(Dropout(fc_dropout[0]))
        if fc_finals[0] > 0: model.add(Dense(fc_finals[0], activation='relu', name='cnn3d_fc1'))
        if fc_dropout[1] > 0: model.add(Dropout(fc_dropout[1]))
        if fc_finals[1] > 0: model.add(Dense(fc_finals[1], activation='relu', name='cnn3d_fc2'))
        if fc_dropout[2] > 0: model.add(Dropout(fc_dropout[2]))
        model.add(Dense(self.nb_classes, activation='softmax', name='cnn3d_predictions'))
        #print(model.input_shape)
        return model

