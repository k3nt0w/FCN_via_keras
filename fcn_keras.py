from keras.layers import merge, Lambda, Flatten, Dense, Convolution2D, MaxPooling2D, Input, Reshape, Permute, UpSampling2D, ZeroPadding2D
from keras.layers.core import Activation
from keras.models import Model
from keras.utils.visualize_util import model_to_dot, plot
from keras.utils.layer_utils import layer_from_config
from keras.utils import np_utils, generic_utils
from keras import backend as K

import cv2
import numpy as np
import matplotlib.pyplot as plt

## Please use tensorflow as backend

def create_model():

    FCN_CLASSES = 21

    #(samples, channels, rows, cols)
    input_img = Input(shape=(3, 224, 224))
    #(3*224*224)
    x = ZeroPadding2D(padding=(1, 1))(input_img)
    x = Convolution2D(64, 3, 3, activation='relu',border_mode='valid')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(64, 3, 3, activation='relu',border_mode='valid')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(64*112*112)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',border_mode='valid')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(128, 3, 3, activation='relu',border_mode='valid')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(128*56*56)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',border_mode='valid')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(256, 3, 3, activation='relu',border_mode='valid')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(256*28*28)

    #split layer
    p3 = x
    p3 = Convolution2D(FCN_CLASSES, 1, 1)(p3)
    #(21*28*28)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='valid')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='valid')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*14*14)

    #split layer
    p4 = x
    p4 = Convolution2D(FCN_CLASSES, 1, 1)(p4)
    u4 = UpSampling2D(size=(2,2))(p4)
    #(21*28*28)

    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='valid')(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = Convolution2D(512, 3, 3, activation='relu',border_mode='valid')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #(512*7*7)

    p5 = x
    p5 = Convolution2D(FCN_CLASSES, 1, 1,activation='relu',border_mode='valid')(p5)
    u5 = UpSampling2D(size=(4,4))(p5)
    #(21*28*28)

    # merge scores
    x = merge([p3, u4, u5], mode='concat', concat_axis=0)
    x = UpSampling2D(size=(8,8))(x)
    x = Reshape((FCN_CLASSES,224*224))(x)
    x = Permute((2,1))(x)
    out = Activation("softmax")(x)
    #(21,224,224)
    model = Model(input_img, out)
    return model

def visualize_model(model):
    plot(model, to_file='FCN_model.png',show_shapes=True)

def to_json(model):
    json_string = model.to_json()
    with open('FCN_via_Keras_architecture.json', 'w') as f:
        f.write(json_string)

if __name__ == "__main__":
    model = create_model()
    visualize_model(model)
    to_json(model)
