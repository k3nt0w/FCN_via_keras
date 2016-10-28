import numpy as np
import cv2
from multiprocessing import Pool
import math

PATH1 = "VOCdevkit/VOC2012/JPEGImages/"
PATH2 = "VOCdevkit/VOC2012/SegmentationClass/"
filename = "/Users/kento_watanabe/Desktop/work/fcn/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"

def resize_img():
    ls = [line.rstrip('\n') for line in open(filename,"r").readlines()]

    for imgname in ls:
        path_to_train = PATH1 + imgname + ".jpg"
        path_to_teacher = PATH2 + imgname + ".png"
        train_img = cv2.imread(path_to_train)
        target_img = cv2.imread(path_to_teacher)

        #resize
        shape = train_img.shape[:-1]
        shorter = shape[0] if shape[0] < shape[1] else shape[1]
        length = int(shorter/2)
        xc, yc = int(shape[0]/2), int(shape[1]/2)
        train_img = train_img[xc-length:xc+length, yc-length:yc+length]
        target_img = target_img[xc-length:xc+length, yc-length:yc+length]
        train_img = cv2.resize(train_img,(224,224))
        target_img = cv2.resize(target_img,(224,224))
        cv2.imwrite('train_voc2012_seg_224/{}.jpg'.format(imgname), train_img)
        cv2.imwrite('target_voc2012_seg_224/{}.png'.format(imgname), target_img)

def make_target_dataset(imgs):
    target = np.zeros((224, 224),dtype="int32")
    ### object : color (BGR)
    background = np.asarray([0,0,0])#0
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

    for imgname in imgs:
        path = "/Users/kento_watanabe/Desktop/data_for_fcn/target_voc2012_seg_224/"
        path_to_target = path + imgname + ".png"
        target_img = cv2.imread(path_to_target)
        dim = target_img.shape
        for x in range(dim[0]):
            for y in range(dim[1]):
                color = target_img[x][y]
                if np.allclose(color, background):
                    target[x,y] = 0
                elif np.allclose(color, airplane):
                    target[x,y] = 1
                elif np.allclose(color, bicycle):
                    target[x,y] = 2
                elif np.allclose(color, bird):
                    target[x,y] = 3
                elif np.allclose(color, boat):
                    target[x,y] = 4
                elif np.allclose(color, bottle):
                    target[x,y] = 5
                elif np.allclose(color, bus):
                    target[x,y] = 6
                elif np.allclose(color, car):
                    target[x,y] = 7
                elif np.allclose(color, cat):
                    target[x,y] = 8
                elif np.allclose(color, chair):
                    target[x,y] = 9
                elif np.allclose(color, cow):
                    target[x,y] = 10
                elif np.allclose(color, table):
                    target[x,y] = 11
                elif np.allclose(color, dog):
                    target[x,y] = 12
                elif np.allclose(color, horse):
                    target[x,y] = 13
                elif np.allclose(color, moterbike):
                    target[x,y] = 14
                elif np.allclose(color, person):
                    target[x,y] = 15
                elif np.allclose(color, potted_plant):
                    target[x,y] = 16
                elif np.allclose(color, sheep):
                    target[x,y] = 17
                elif np.allclose(color, sofa):
                    target[x,y] = 18
                elif np.allclose(color, train):
                    target[x,y] = 19
                elif np.allclose(color, moniter):
                    target[x,y] = 20
                else:
                    target[x,y] = 0
        np.save("/Users/kento_watanabe/Desktop/data_for_fcn/npy_target_224_voc12/{}.npy".format(imgname),target)
        print("Done {}".format(imgname))

def split_n_data(n):
    ls = [line.rstrip('\n') for line in open(filename,"r").readlines()]
    size = math.ceil(len(ls)/n)
    split_data = [ls[i:i+size] for i in range(0,len(ls),size)]
    return split_data

def multi_process():
    pool = Pool(processes=3)
    #result_data = [None for i in range(3)]
    split_data = split_n_data(3)
    pool.map(make_target_dataset, split_data)

if __name__ == "__main__":
    ls = [line.rstrip('\n') for line in open(filename,"r").readlines()]
    multi_process()
