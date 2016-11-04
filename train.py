from FCN import FCN
from data import generate_arrays_from_file
from DeconvNetModel import DeconvNet

from keras.models import model_from_json
import cv2
import numpy as np
import os

import os
os.environ['KERAS_BACKEND'] = 'theano'
#for using gpu
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'

def train(path,model_name="FCN"):
    model_dec = { "FCN" : FCN, "DeconvNet" : DeconvNet}
    model = model_dec[model_name]()
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta',
                  metrics=["accuracy"])
    model.load_weights("weights_of_{}.hdf5".format(model_name))
    model.fit_generator(generate_arrays_from_file(path),
        samples_per_epoch=20, nb_epoch=20)
    model.save_weights('weights_of_{}.hdf5'.format(model_name))
    return

if __name__ == "__main__":
    #train(train.txt,model_name="DeconvNet")
    train('./test_train.txt',model_name="DeconvNet")
