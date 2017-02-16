import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from model import FullyConvolutionalNetwork
from preprocess import *

import argparse

from keras.optimizers import Adam
from keras import backend as K

parser = argparse.ArgumentParser(description='FCN via Keras')
parser.add_argument('--train_dataset', '-tr', default='dataset', type=str)
parser.add_argument('--target_dataset', '-ta', default='dataset', type=str)
parser.add_argument('--txtfile', '-t', type=str, required=True)
parser.add_argument('--weight', '-w', default="", type=str)
parser.add_argument('--epoch', '-e', default=20, type=int)
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


def categorical_crossentropy2D(y_true, y_pred):
    return -K.sum(y_true * K.log(y_pred))

with open(path_to_txt,"r") as f:
    ls = f.readlines()
names = [l.rstrip('\n') for l in ls]
nb_data = len(names)

FCN = FullyConvolutionalNetwork(img_height=img_size, img_width=img_size, FCN_CLASSES=nb_class)
adam = Adam(lr=args.lr)
model = FCN.create_model()

model.compile(loss=categorical_crossentropy2D, optimizer='adam')
if len(args.weight):
    model.load_weights(args.weight, model)
print("Num data: {}".format(nb_data))

"""
#test
Xpath = "demo_imgs/X1.jpg"
ypath = "demo_imgs/y1.png"
X = load_data(Xpath, 224, label=False)
y = load_data(ypath, 224, label=True)
model.fit(X,y)

pred = model.predict(X)[0]
print(pred[0,0,:].sum())
print(pred[0,0,:])
"""
model.fit_generator(generate_arrays_from_file(names,path_to_train,path_to_target,img_size),
                    samples_per_epoch=nb_data,
                    nb_epoch=args.epoch)
if not os.path.exists("./weight"):
    os.mkdir("./weight")

#model.save_weights('./weight"/fcn_params')
