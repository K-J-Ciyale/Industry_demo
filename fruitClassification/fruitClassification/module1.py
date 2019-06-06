
import cv2 as cv
from PIL import Image
import numpy as np
def video_demo():
#0是代表摄像头编号，只有一个的话默认为0
    capture=cv.VideoCapture(0) 
    while(True):
        ref,frame=capture.read()
        b = Image.fromarray(frame, mode='RGB')
        b.show()
        #等待30ms显示图像，若过程中按“Esc”退出
        c= cv.waitKey(30) & 0xff 
        if c==27:
            capture.release()
            break
import os
def getError(keyword):
    if keyword == 'Apple':
        return 0
    elif keyword == 'Banana':
        return 1
    elif keyword == 'Apricot':
        return 2
    elif keyword == 'Tomato':
        return 3
    elif keyword == 'Pear':  #梨
        return 4
    elif keyword == 'Pepper':   #辣椒
        return 5
    elif keyword == 'Plum': #李子
        return 6
    elif keyword == 'Chestnut': #板栗
        return 7
    elif keyword == 'Orange':
        return 8
    elif keyword == 'Carambula': #杨桃
        return 9
#video_demo()
def getPic():
    errorList = ['Apple', 'Banana', 'Apricot', 'Tomato', 'Pear', 'Pepper', 'Plum', 'Chestnut', 'Orange', 'Carambula']
    classDefinedPath = 'H:\\yan\\fruitClassification2\\fruits\\fruits-360\\Training'
    path = 'H:\\yan\\fruitClassification2\\fruits\\fruits-360\\Test'
    newPath = "H:\\yan\\fruitClassification2\\datasets\\fruitTest64_48"
    index = 0
    filePath = os.listdir(path)
    classPathList = os.listdir(classDefinedPath)
    for p in filePath:
        claasification = p.split(' ')[0]
        if (p in classPathList):
            idClassification = str(getError(claasification))
            if (claasification in errorList):
                picList = os.listdir(path + '\\' + p)
                for pic in picList:
                    index += 1
                    im = Image.open(path + '\\' + p + '\\' + pic)
                    name = path + '\\' + p + '\\' + claasification + str(index) + '_' + idClassification + '.jpg'            
                    im = im.resize((64, 48))
                    savePath = newPath + '\\' + claasification + str(index) + '_' + idClassification + '.jpg'
                    print(savePath)
                    im.save(savePath)
                    if (name != path + '\\' + p + '\\' + pic):
                        os.remove(path + '\\' + p + '\\' + pic)
getPic()