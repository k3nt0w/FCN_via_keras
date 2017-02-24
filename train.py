import os
os.environ['KERAS_BACKEND'] = 'theano'

from model import FullyConvolutionalNetwork
from preprocess import *

import argparse
import h5py

from keras.optimizers import Adam
from keras import backend as K

def crossentropy(y_true, y_pred):
    return -K.sum(y_true*K.log(y_pred))

parser = argparse.ArgumentParser(description='FCN via Keras')
parser.add_argument('--train_dataset', '-tr', default='dataset', type=str)
parser.add_argument('--target_dataset', '-ta', default='dataset', type=str)
parser.add_argument('--txtfile', '-t', type=str, required=True)
parser.add_argument('--weight', '-w', default="", type=str)
parser.add_argument('--epoch', '-e', default=100, type=int)
parser.add_argument('--classes', '-c', default=21, type=int)
parser.add_argument('--batchsize', '-b', default=1, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--image_size', default=224, type=int)

args = parser.parse_args()
img_size = args.image_size
nb_class = args.classes
path_to_train = args.train_dataset
path_to_target = args.target_dataset
path_to_txt = args.txtfile

with open(path_to_txt,"r") as f:
    ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
nb_data = len(names)

FCN = FullyConvolutionalNetwork(img_height=img_size, img_width=img_size, FCN_CLASSES=nb_class)
adam = Adam(lr=args.lr)
train_model = FCN.create_model(train_flag=True)
train_model.compile(loss=crossentropy, optimizer='adam')
if len(args.weight):
    model.load_weights(args.weight, model)
print("Num data: {}".format(nb_data))

train_model.fit_generator(generate_arrays_from_file(names,path_to_train,path_to_target,img_size, nb_class),
                    samples_per_epoch=nb_data,
                    nb_epoch=args.epoch)

if not os.path.exists("weights"):
    os.makedirs("weights")

train_model.save_weights("weights/temp", overwrite=True)
f = h5py.File("weights/temp")

layer_names = [name for name in f.attrs['layer_names']]
fcn = FCN.create_model(train_flag=False)

for i, layer in enumerate(fcn.layers):
    g = f[layer_names[i]]
    weights = [g[name] for name in g.attrs['weight_names']]
    layer.set_weights(weights)

fcn.save_weights("weights/fcn_params", overwrite=True)

f.close()
os.remove("weights/temp")

print("Saved weights")
