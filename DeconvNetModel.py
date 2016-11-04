from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Reshape, Permute, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D, Cropping2D
from keras.optimizers import SGD
from keras.utils.visualize_util import model_to_dot, plot

def conv_layers(model):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='same',input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(128, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(256, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(256, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(Convolution2D(512, 3, 3, activation='relu',border_mode='same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    return model

def bridge(model):
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Reshape((4096,1,1)))
    return model

def deconv_layers(model):
    model.add(Deconvolution2D(512, 7, 7,
            output_shape=(None, 512, 7, 7),
            subsample=(1, 1),
            border_mode='valid',
            activation="relu"))
    model.add(UpSampling2D((2,2))) #unpool_5
    for _ in range(3): #deconv_5
        model.add(Deconvolution2D(512, 3, 3,
                output_shape=(None, 512, 16, 16),
                subsample=(1, 1),
                border_mode='valid',
                activation="relu"))
        model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    # 512*14*14

    model.add(UpSampling2D((2,2))) #unpool_4
    for _ in range(2): #deconv_4
        model.add(Deconvolution2D(512, 3, 3,
                output_shape=(None, 512, 30, 30),
                subsample=(1, 1),
                border_mode='valid',
                activation="relu"))
        model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    model.add(Deconvolution2D(256, 3, 3,
            output_shape=(None, 256, 30, 30),
            subsample=(1, 1),
            border_mode='valid',
            activation="relu"))
    model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    #256*28*28

    model.add(UpSampling2D((2,2))) #unpool_3
    for _ in range(2): #deconv_3
        model.add(Deconvolution2D(256, 3, 3,
                output_shape=(None, 256, 58, 58),
                subsample=(1, 1),
                border_mode='valid',
                activation="relu"))
        model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    model.add(Deconvolution2D(128, 3, 3,
            output_shape=(None, 128, 58, 58),
            subsample=(1, 1),
            border_mode='valid',
            activation="relu"))
    model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    #128*56*56

    model.add(UpSampling2D((2,2))) #unpool_2
    model.add(Deconvolution2D(128, 3, 3,
            output_shape=(None, 128, 114, 114),
            subsample=(1, 1),
            border_mode='valid',
            activation="relu"))
    model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    model.add(Deconvolution2D(64, 3, 3,
            output_shape=(None, 64, 114, 114),
            subsample=(1, 1),
            border_mode='valid',
            activation="relu"))
    model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    #64*112*112

    model.add(UpSampling2D((2,2)))
    model.add(Deconvolution2D(64, 3, 3,
            output_shape=(None, 64, 226, 226),
            subsample=(1, 1),
            border_mode='valid',
            activation="relu"))
    model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    model.add(Deconvolution2D(21, 3, 3,
            output_shape=(None, 21, 226, 226),
            subsample=(1, 1),
            border_mode='valid',
            activation="relu"))
    model.add(Cropping2D(cropping=((1, 1), (1, 1))))
    model.add(Reshape((21, 224*224)))
    model.add(Permute((2, 1)))
    model.add(Activation(("softmax")))
    return model

def DeconvNet():
    model = Sequential()
    return deconv_layers(bridge(conv_layers(model)))

def visualize_model(model):
    plot(model, to_file='DeconNet_model.png',show_shapes=True)

def to_json(model):
    json_string = model.to_json()
    with open('DeconvNet_architecture.json', 'w') as f:
        f.write(json_string)

if __name__ == "__main__":
    model = Sequential()
    #model = bridge(conv_layers(model))
    model = deconv_layers(bridge(conv_layers(model)))

    visualize_model(model)
    to_json(model)
