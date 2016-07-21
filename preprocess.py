import cv2

def getDataSet():
    #リストの作成
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    for i in range(0,10):
        path = "/Users/path/to/dir"
        if i == 0:
            cutNum = 22000
            cutNum2 = 18000
        else:
            cutNum = 4000
            cutNum2 = 3600
        imgList = os.listdir(path+str(i))
        imgNum = len(imgList)
        for j in range(cutNum):
            imgSrc = cv2.imread(path+str(i)+"/"+imgList[j])
            if imgSrc is None:continue
            if j < cutNum2:
                X_train.append(imgSrc)
                y_train.append(i)
            else:
                X_test.append(imgSrc)
                y_test.append(i)

    return X_train,y_train,X_test,y_test
