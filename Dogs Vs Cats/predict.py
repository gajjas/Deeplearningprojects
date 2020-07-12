import cv2
import tensorflow as tf
import sys, getopt

CATEGORIES = ["dog", "cat"]


def prepare(filepath):
    IMG_SIZE = 50
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def main():
    model = tf.keras.models.load_model("64x3-CNN.model")
    
    filepath = input("Input a filepath (from current directory) of the image of a Dog or Cat: ")
    prediction = model.predict([prepare(filepath)])

    print('This is prdicted to be a %s' %CATEGORIES[int(prediction[0][0])])

if __name__ == "__main__":
    main()