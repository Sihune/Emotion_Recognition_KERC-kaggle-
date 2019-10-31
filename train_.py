"""
Train our RNN on extracted features or images.
"""
#from model_fine_tuning import FineTuning_EmotionRecognition_V0
#from keras.models import Model
from model_fine_tuning import FineTuning_EmotionRecognition_V0
from keras_vggface.vggface import VGGFace
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from keras.layers import Input
from resnetModel import ResnetBuilder
from data import DataSet
import time
import os.path
from vgg16Model import build_vgg16
from model_vggface2 import Vggface2_ResNet50

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def train(data_type, seq_length, model, saved_model=None,
          class_limit=None, image_shape=None,
          load_to_memory=False, batch_size=32, nb_epoch=100):
    # Helper: Save the model.

    checkpointer = ModelCheckpoint(filepath='data/checkpoints/weights.hdf5', monitor='val_loss', verbose=1,
                                   save_best_only=True)
    # Helper: TensorBoard
    tb = TensorBoard(log_dir=os.path.join('data', 'logs', model))

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=100)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join('data', 'logs', model + '-' + 'training-' + \
                                        str(timestamp) + '.log'))

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory == True:
        # Get data.
        X, y = data.get_all_sequences_in_memory('train', data_type)
        X_val, y_val = data.get_all_sequences_in_memory('val', data_type)

    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type)
        val_generator = data.frame_generator(batch_size, 'val', data_type)

    print(X.shape)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model)

    print("############################")
    print(rm.input_shape)

    # Fit!
    if load_to_memory == True:
        # Use standard fit.

        rm.model.fit(
            X,
            y,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            # callbacks=[tb, early_stopper, csv_logger]
            callbacks=[tb, early_stopper, checkpointer],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            # callbacks=[tb, early_stopper, csv_logger, checkpointer],
            callbacks=[early_stopper, checkpointer],
            validation_data=val_generator,
            validation_steps=1,
            workers=4)


def main():
    """These are the main training settings. Set each before running
    this file."""
    if os.path.exists(os.path.join('data','checkpoints')) == False: 
        os.makedirs(os.path.join('data','checkpoints'))
        
    nb_classes = 7

    # model can be one of lrcn, mlp, vgg_16, conv3D
    model = 'conv3D'
    #saved_model = VGGFace(weights='vggface', include_top=True, input_shape=(224, 224, 3))

    #saved_model = FineTuning_EmotionRecognition_V0(weights_path="./pre_trained/weights.h5",model_name = "vggface2_resnet50", nb_classes = 7, dropout = [0.0, 0.0], mode = "train")
    #saved_model = ResnetBuilder.build_resnet_50((1,48,48), 7)
    #saved_model = Vggface2_ResNet50(input_dim = (48,48,3), nb_classes = nb_classes, optimizer='adam')

    input_image = Input(shape=(48, 48, 3), name='input')
    saved_model = build_vgg16(input_image, nb_classes)

    idx_features = -1
    idx_conv = -1
    saved_model.summary()


    for idx in range(len(saved_model.layers)):
        if saved_model.layers[idx].name == "fc_dropout3": #features / flatten_1
            idx_features = idx
        elif saved_model.layers[idx].name == "conv5_1": #conv5_1_1x1_reduce / conv2d_45
            idx_conv = idx

    for idx in range(len(saved_model.layers)):
        if idx < idx_conv:
            saved_model.layers[idx].trainable = False
        else:
            saved_model.layers[idx].trainable = True
    # for

    saved_model = Model(inputs=saved_model.input, outputs=saved_model.layers[idx_features].output)

    '''
    for layer in saved_model.layers[:-4]:
        layer.trainable = False

    for layer in saved_model.layers:
        print(layer, layer.trainable)
    '''
    #saved_model = FineTuning_EmotionRecognition_V0(weights_path="data/checkpoints/pre_trained.hdf5",input_shape = (224, 224, 3), model_name = "vggface2_resnet50",
    #                                               nb_classes = 7, dropout = [0.0, 0.0], mode = "train")  # None or weights file
    class_limit = None  # int, can be 1-101 or None
    seq_length = 20
    load_to_memory = True  # pre-load the sequences into memory
    batch_size = 16
    nb_epoch = 1000

    # Chose images or features and image shape based on network.
    if model in ['lrcn']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['vgg_16']:
        data_type = 'images'
        image_shape = (80, 80, 3)
    elif model in ['mlp']:
        data_type = 'features'
        image_shape = None
    elif model in ['conv3D']:
        data_type = 'images'
        image_shape = (48, 48, 3)
    else:
        raise ValueError("Invalid model. See train.py for options.")

    train(data_type, seq_length, model, saved_model=saved_model,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory, batch_size=batch_size, nb_epoch=nb_epoch)

if __name__ == '__main__':
    main()
