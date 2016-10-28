import numpy as np
import cv2
import os

def binarylab(labels):
    x = np.zeros([224,224,21])
    for i in range(224):
        for j in range(224):
            x[i,j,labels[i][j]]=1
    return x

def data(test=False):
    path_to_train = "/Users/kento_watanabe/Desktop/data_for_fcn/train_voc2012_seg_224/"
    path_to_target = "/Users/kento_watanabe/Desktop/data_for_fcn/npy_target_224_voc12/"

    if test:
        X_train = cv2.imread(path_to_train+"2011_003255.jpg")
        X_train = X_train.astype('float32')
        X_train /= 255
        X_train = X_train.reshape((1,3,224,224))

        Y_train = np.load(path_to_target+"2011_003255.npy")
        Y_train = np.asarray(binarylab(Y_train))
        Y_train = np.reshape(Y_train,(1,224*224,21))
        return X_train, Y_train, X_train, Y_train
    else:
        filename = "/Users/kento_watanabe/Desktop/work/fcn/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
        with open(filename,"r") as f:
            names = f.readlines()
            ls = [line.rstrip('\n') for line in names]

            #########test
            ls = ls[:100]

        L = len(ls)
        BOUND = L*0.9

        X_train = list()
        Y_train = list()
        x_append = X_train.append
        y_append = Y_train.append

        X_test = list()
        Y_test = list()
        xt_append = X_test.append
        yt_append = Y_test.append

        print("making dataset")
        for i,imgname in enumerate(ls):
            x_data = cv2.imread(path_to_train+ imgname +".jpg")
            x_data = x_data.astype('float32')
            x_data /= 255

            y_data = np.load(path_to_target+ imgname +".npy")
            y_data = np.asarray(binarylab(y_data))
            if i < BOUND:
                x_append(x_data)
                y_append(y_data)
            else:
                xt_append(x_data)
                yt_append(y_data)
    nb_train_data = len(X_train)
    nb_test_data  = len(X_test)
    assert len(X_train) == len(Y_train), "not same the number of data"
    X_train = np.asarray(X_train).reshape((nb_train_data,3,224,224))
    Y_train = np.asarray(Y_train).reshape((nb_train_data,224*224,21))
    X_test = np.asarray(X_test).reshape((nb_test_data,3,224,224))
    Y_test = np.asarray(Y_test).reshape((nb_test_data,224*224,21))

    return X_train, Y_train, X_test, Y_test
