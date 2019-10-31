from keras.layers import Flatten, Dropout, Dense
from keras.models import Model
from keras.regularizers import l2

from model_vggface2 import Vggface2_ResNet50
from model_vggface1_senet50 import SENET50
from model_nasnet import  NASNetLarge, NASNetMobile
from model_densenet import DenseNet201
from model_inception_v3 import InceptionV3
from model_inception_resnet_v2 import InceptionResNetV2
from model_xception import  Xception
from model_resnet50 import  ResNet50


global weight_decay
weight_decay = 1e-4

def FineTuning_EmotionRecognition_V0(weights_path = None,
                                     model_name = "vggface1_senet50",
                                     nb_classes = 7,
                                     dropout = [0.0, 0.0],
                                     mode = "train"):
    """
    Model FineTuning for Emotion Recognition
    model_name = ['vggface1_senet50']
    """
    if model_name == 'vggface2_resnet50':
        base_model = Vggface2_ResNet50()
        #base_model.summary()
    elif model_name == 'vggface1_senet50':
        base_model = SENET50(include_top=False, weights = None, input_tensor=None, input_shape=None, pooling="max")
    elif model_name == 'imagenet_nasnet':
        base_model = NASNetMobile(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="max")
    elif model_name == 'imagenet_densenet':
        base_model = DenseNet201(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="max")
    elif model_name == 'imagenet_inception_v3':
        base_model = InceptionV3(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="max")
    elif model_name == 'imagenet_inception_resnet_v2':
        base_model = InceptionResNetV2(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="max")
    elif model_name == 'imagenet_xception':
        base_model = Xception(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="max")
    elif model_name == 'imagenet_resnet50':
        base_model = ResNet50(include_top=False, weights=None, input_tensor=None, input_shape=None, pooling="max")
    # if

    x = base_model.output


    if dropout[0]>0: x = Dropout(dropout[0])(x)
    x = Dense(2048, activation='relu', name='features')(x)
    if dropout[1] > 0: x = Dropout(dropout[1])(x)
    x = Dense(nb_classes,
              activation='softmax',
              name='predictions',
              use_bias=False, trainable=True,
              kernel_initializer='orthogonal',
              kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=base_model.input, outputs=x)

    if weights_path is not None:
        model.load_weights(weights_path, by_name = True)

    return model
# FineTuning_EmotionRecognition_V0