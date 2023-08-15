import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from mkl_random import shuffle
from sklearn import svm, datasets
from skimage.feature import hog
import os
import cv2
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

classes = ['accordian', "dollar_bill", "motorbike", "Soccer_Ball"]
pathse_train = ['train/accordian', 'train/dollar_bill', 'train/motorbike', 'train/Soccer_Ball']
pathse_test = ['test/accordian', 'test/dollar_bill', 'test/motorbike', 'test/Soccer_Ball']

train_data = []
test_data = []


def create_label(clas):
    if clas == 'accordian':
        return np.array([1, 0, 0, 0])
    elif clas == "dollar_bill":
        return np.array([0, 1, 0, 0])
    elif clas == "motorbike":
        return np.array([0, 0, 1, 0])
    elif clas == "Soccer_Ball":
        return np.array([0, 0, 0, 1])


def create_data(path_f, clas, t):
    for img in os.listdir(path_f):
        path = os.path.join(path_f, img)
        img_data = cv2.imread(path)
        img_data = img_data[:, :, [2, 1, 0]]
        img_data = cv2.resize(img_data, (128, 64))
        #fd, hog_img = hog(img_data, orientations = 9, pixels_per_cell= (8,8),cells_per_block = (2,2),visualize=True , multichannel = True)
        if t:
            train_data.append([np.array(img_data), create_label(clas)])
            shuffle(train_data)
        else:
            test_data.append([np.array(img_data), create_label(clas)])
            shuffle(test_data)


for i, j in zip(pathse_train, classes):
    create_data(i, j, True)

for i, j in zip(pathse_test, classes):
    create_data(i, j, False)

x_train = np.array([i[0] for i in train_data]).reshape(-1, 128, 64, 3)
y_train = np.array([i[1] for i in train_data])
print("x_train", x_train.shape)
print("y_train", y_train.shape)

x_test = np.array([i[0] for i in test_data]).reshape(-1, 128, 64, 3)
y_test = np.array([i[1] for i in test_data])
print("x_test", x_test.shape)
print("y_test", y_test.shape)

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)

svm = LinearSVC(C=0.5)
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
print("accu", accuracy_score(y_test, y_pred))
