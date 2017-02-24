from chainer import serializers
import numpy as np
from PIL import Image
import os
import argparse
import cv2
from time import time

from model import FullyConvolutionalNetwork
from preprocess import load_data
from color_map import make_color_map

parser = argparse.ArgumentParser(description='Chainer Fully Convolutional Network: predict')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--image_path', '-i', default=None, type=str)
parser.add_argument('--weight', '-w', default="weight/chainer_fcn.weight", type=str)
parser.add_argument('--classes', default=21, type=int)
parser.add_argument('--clop', "-c", default=True, type=bool)
parser.add_argument('--clopsize', "-s", default=224, type=int)
args = parser.parse_args()

img_name = args.image_path.split("/")[-1].split(".")[0]

color_map = make_color_map()
FCN = FullyConvolutionalNetwork()
model = FCN.create_model()
model.load_weights('weights/fcn_params')

o = load_data(args.image_path, size=args.clopsize, mode="original")
x = load_data(args.image_path, size=args.clopsize, mode="data")

start = time()
pred = model.predict(x)
pred = pred[0].argmax(axis=0)
print(pred.max())

row, col = pred.shape
dst = np.ones((row, col, 3))
for i in range(21):
    dst[pred == i] = color_map[i]
img = Image.fromarray(np.uint8(dst))

b,g,r = img.split()
img = Image.merge("RGB", (r, g, b))

trans = Image.new('RGBA', img.size, (0, 0, 0, 0))
w, h = img.size
for x in range(w):
    for y in range(h):
        pixel = img.getpixel((x, y))
        if (pixel[0] == 0   and pixel[1] == 0   and pixel[2] == 0)or \
           (pixel[0] == 255 and pixel[1] == 255 and pixel[2] == 255):
            continue
        trans.putpixel((x, y), pixel)
#o.paste(trans, (0,0), trans)

if not os.path.exists("out"):
    os.mkdir("out")
o.save("out/original.jpg")
trans.save("out/pred.png")

o = cv2.imread("out/original.jpg", 1)
p = cv2.imread("out/pred.png", 1)

pred = cv2.addWeighted(o, 0.4, p, 0.6, 0.0)

cv2.imwrite("out/pred_{}.png".format(img_name), pred)

os.remove("out/original.jpg")
os.remove("out/pred.png")
