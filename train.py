from model import create_model
from data import generate_arrays_from_file

from keras.models import model_from_json
import cv2
import numpy as np
import os

import os
os.environ['KERAS_BACKEND'] = 'theano'
#for using gpu
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

def train(path):
    model = create_model()
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta',
                  metrics=["accuracy"])
    model.fit_generator(generate_arrays_from_file(path),
        samples_per_epoch=20, nb_epoch=15)
    model.save_weights('weights.hdf5')
    return

if __name__ == "__main__":
    train('./train.txt')
