import numpy as np
import cv2
import os

def binarylab(labels):
    x = np.zeros([224,224,21])
    for i in range(224):
        for j in range(224):
            x[i,j,labels[i][j]]=1
    return x

def generate_arrays_from_file(path):
    while 1:
        with open(path,"r") as f:
            names = f.readlines()
        ls = [line.rstrip('\n') for line in names]
        for name in ls:
            img, target = process_line(name)
            yield (img, target)

def process_line(name):
    path_to_train = "/Users/kento_watanabe/Desktop/data_for_fcn/train_voc2012_seg_224/"
    path_to_target = "/Users/kento_watanabe/Desktop/data_for_fcn/npy_target_224_voc12/"
    img = cv2.imread(path_to_train+ name +".jpg")
    img = img.astype('float32')
    img /= 255
    img = img.reshape((1,3,224,224))
    target = np.load(path_to_target+ name +".npy")
    target = np.asarray(binarylab(target))
    target = target.reshape((1,224*224,21))
    return img, target
