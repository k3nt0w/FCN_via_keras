from model import create_model
from data import generate_arrays_from_file

from keras.models import model_from_json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def train(test=False):
    model = create_model()
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta',
                  metrics=["accuracy"])
    model.fit_generator(generate_arrays_from_file('./train.txt'),
        samples_per_epoch=1400, nb_epoch=100)
    model.save_weights('weights.hdf5')
    return

def make_graph(history):
    pass

if __name__ == "__main__":
    #train(test=True)
    train()
