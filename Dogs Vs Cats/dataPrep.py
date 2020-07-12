import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

DATADIR = 'C:/Users/gajjas/Dropbox/Projects/Machine Learning/Deep Learning Projects/Dogs Vs Cats/dataset/training_set'
CATEGORIES = ['dogs', 'cats']
IMG_SIZE = 50

def createTrainingData():
    training_data = []
    for category in CATEGORIES:

        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    return training_data

def saveData(dataX, dataY):
    pickle_out = open("X_train.pickle","wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    training_data = createTrainingData()
    X = []
    y = []

    for features,label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

    saveData(X, y)
    print("Data Successfully Saved!!!")