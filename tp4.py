# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:43:58 2018

@author: Célande
"""

from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from random import randint

"""
    1 image is grayscale
    28x28
    with value 0 -> 255
    
    X_train, y_train <=> 60 000 first data
    X_test, y_test <=> 10 000 data left
"""

# set the seed
np.random.seed(42)


# show an image from 1D array grayscale image
def show_img(img, nb_pixel):
    data = np.asarray(img)
    
    data = data.reshape(nb_pixel,nb_pixel)
    
    plt.imshow(data, cmap=plt.get_cmap('gray'))
    plt.show()

# change a grayscale 1D image into binary image
def data_format(data, nb_pixel):
    
    for i in range(0, data.size):
        if data[i] > 0:
            data[i] = 1
                    
    return data

# change image dataset into binary image dataset
def set_format(data, nb_pixel):
    result = []
    for i in range(0, data.shape[0]):
        result.append(data_format(data[i], nb_pixel))
    
    return np.array(result)

# load dataset and save into pkl files
def set_dataset(test_size, nb_pixel):
    mns = fetch_mldata ("MNIST original")
    
    X = mns.data
    y = mns.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    print("Setting train format")
    X_train = set_format(X_train, nb_pixel)
    print("Setting test format")
    X_test = set_format(X_test, nb_pixel)
    
    joblib.dump(X_train, "X_train.pkl")
    joblib.dump(y_train, "y_train.pkl")
    
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")
    
    print("Format set")

# load training dataset and save classifer into pkl file
def set_classifier(*nb_neurons):
    
    X_train = joblib.load("X_train.pkl")
    y_train = joblib.load("y_train.pkl")
    
    print("Setting classifier")
    classifier = MLPClassifier(hidden_layer_sizes=(nb_neurons), solver='adam', learning_rate_init = 0.001, random_state=42)
    print("Fitting classifier")
    classifier.fit(X_train, y_train)
    print("Classifier done")
    
    joblib.dump(classifier, "classifier.pkl")

# load testing dataset and classifier and score
def score_classifier():
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    
    classifier = joblib.load("classifier.pkl")
    print("Scoring classifier")
    print(classifier.score(X_test, y_test))

def predict_img(img, label, nb_pixel):
    show_img(img, nb_pixel)
    classifier = joblib.load("classifier.pkl")
    print("predict: {}, label: {}".format(classifier.predict(img.reshape(1, -1)), label))

def test_predict(nb_pixel):
    X_test = joblib.load("X_test.pkl")
    y_test = joblib.load("y_test.pkl")
    i = randint(0, X_test.shape[0])
    predict_img(X_test[i], y_test[i], nb_pixel)
    
if __name__ == '__main__':
    nb_pixel = 28
    nb_neurons = [10, 5, 2]
    test_size = 0.14
    #set_dataset(test_size, nb_pixel)
    #set_classifier(*nb_neurons)
    #score_classifier()
    
    test_predict(nb_pixel)