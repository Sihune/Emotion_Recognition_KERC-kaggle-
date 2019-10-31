import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam


def build_vgg16(input_image, nb_classes):

    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_1')(input_image)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)
    x = Dropout(rate=0.25, name='conv_dropout1')(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)
    x = Dropout(rate=0.25, name='conv_dropout2')(x)

    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_2')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool3')(x)
    x = Dropout(rate=0.25, name='conv_dropout3')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool4')(x)
    x = Dropout(rate=0.25, name='conv_dropout4')(x)

    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_1')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_2')(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_3')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool5')(x)
    x = Dropout(rate=0.25, name='conv_dropout5')(x)

    # similar to the MLP example!
    # matrix ---> vector
    x = Flatten(name='flatten')(x)

    # FC layers + dropout
    x = Dense(units=4096, activation='relu', name='fc1')(x)
    x = Dropout(rate=0.5, name='fc_dropout1')(x)

    x = Dense(units=4096, activation='relu', name='fc2')(x)
    x = Dropout(rate=0.5, name='fc_dropout2')(x)

    x = Dense(units=1000, activation='relu', name='fc3')(x)
    x = Dropout(rate=0.5, name='fc_dropout3')(x)

    output_label = Dense(units=nb_classes, activation = 'softmax', name='fc3_7ways_softmax')(x)

    model = Model(inputs = input_image, outputs=output_label, name = 'emo_cnn')

    return model
