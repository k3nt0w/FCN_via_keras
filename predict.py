import fcn_keras as fk

from keras.models import model_from_json
import cv2
import numpy as np
import matplotlib.pyplot as plt

def predict(img_path):
    model = fk.create_model()
    model.compile(loss="categorical_crossentropy",
                  optimizer='adadelta',
                  metrics=["accuracy"])
    model.load_weights("weights.hdf5")
    #model.load_weights("test_model_weight.hdf5")

    path_to_train = "/Users/kento_watanabe/Desktop/data_for_fcn/train_voc2012_seg_224/"
    data = cv2.imread(path_to_train+"2011_003255.jpg")
    data = data.astype('float32')
    data /= 255
    data = data.reshape((1,3,224,224))

    output = model.predict_on_batch(data)
    output = np.argmax(output[0],axis=1).reshape((224,224))
    print(output)
    return output

def visualize(output, plot=True):
    background = np.asarray([255,255,255])#0
    airplane = np.asarray([0,0,128])#1
    bicycle = np.asarray([0,128,0])#2
    bird = np.asarray([0,128,128])#3
    boat = np.asarray([128,0,0])#4
    bottle = np.asarray([128,0,128])#5
    bus = np.asarray([128,128,0])#6
    car = np.asarray([128,128,128])#7
    cat = np.asarray([0,0,64])#8
    chair = np.asarray([0,0,192])#9
    cow = np.asarray([0,128,64])#10
    table = np.asarray([0,128,192])#11
    dog = np.asarray([128,0,64])#12
    horse = np.asarray([128,0,192])#13
    moterbike = np.asarray([128,128,64])#14
    person = np.asarray([128,128,192])#15
    potted_plant = np.asarray([0,64,0])#16
    sheep = np.asarray([0,64,128])#17
    sofa = np.asarray([128,128,192])#18
    train = np.asarray([128,128,192])#19
    moniter = np.asarray([128,64,0])#20

    label_colors = np.array([background,airplane,bicycle,bird,boat,
    bottle,bus,car,cat,chair,cow,table,dog,horse,moterbike,person,
    potted_plant,sheep,sofa,train,moniter])

    img = np.zeros((224,224,3))

    for h,tmp in enumerate(output):
        for w,label in enumerate(tmp):
            img[h,w] = label_colors[label]
    cv2.imwrite("result.jpg",img)
    return

def test():
    path_to_train = "/Users/kento_watanabe/Desktop/data_for_fcn/train_voc2012_seg_224/"
    path_to_target = "/Users/kento_watanabe/Desktop/data_for_fcn/npy_target_224_voc12/"
    output = np.load(path_to_target+"2011_003255.npy")
    visualize(output)
    return

if __name__ == "__main__":
    visualize(predict("path"))
    #test()
