import cv2
import os
import numpy as np


def transfer_data(source, transfer_dir, class_num, start, end):
    for img in os.listdir(source):
        path_img = os.path.join(source, img)
        img_data = cv2.imread(path_img, cv2.IMREAD_COLOR)
        cv2.imwrite('%s\\%d\\%d.jpg' %(transfer_dir,class_num,start), img_data)
        start = start + 1
        if start == end:
            break
    return

def delete_data():
    for num_class in range(5, 26):
        for i in range(1, 21):
            os.remove('Data\\2d\\Test\\%d\\%d.png' %(num_class,i))
    return

def flip_image():
    pic_num = 1
    for img in os.listdir('TestFlip\\3'):
        path_img = os.path.join('TestFlip\\3', img)
        img_data = cv2.imread(path_img, cv2.IMREAD_COLOR)
        flip_img = cv2.flip(img_data, 1)
        cv2.imwrite('Test\\3\\3-%d.jpg' %pic_num, flip_img)
        pic_num = pic_num + 1


def concat(X_1, Y_1, X_2, Y_2):
    X1 = np.load('%s' %X_1)
    Y1 = np.load('%s' %Y_1)
    X2 = np.load('%s' % X_2)
    Y2 = np.load('%s' % Y_2)
    X = np.concatenate([X1, X2], axis=0)
    Y = np.concatenate([Y1, Y2], axis=0)
    np.save('X_validation', X)
    print(X.shape)
    np.save('Y_validation', Y)
    print(Y.shape)
    return

delete_data()