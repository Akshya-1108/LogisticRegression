from random import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
plt.style.use("seaborn-v0_8")


# Step-2 Visulaize the Dataset
def visualize(X,y):
    plt.scatter(X[:,0],X[:,1], c=y, cmap='seismic')
    plt.show()

# Step3 standardization dataset
def normalize(X):
    u = X.mean(axis=0)
    std = X.std(axis =0)

    return (X-u)/std

# step - 5 Model
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def hypothesis(X, theta):
    return sigmoid(np.dot(X,theta))

# Binary Cross Entropy
def error(y,yp):
     loss = - np.mean(y*np.log(yp) + (1-y)*np.log(1-yp))
     return loss

def preprocess(X):
    # if X.ndim == 1:
    #     X = X.reshape(-1, 1)
    m = X.shape[0]
    ones = np.ones((m, 1))
    X = np.hstack((ones, X))
    return X

def gradient (X,y,yp):
    m = X.shape[0]
    grad = -(1/m)*np.dot(X.T, (y-yp))
    return grad

def train ( X,y,max_itres = 100, learning_rate = 0.1):
    n = X.shape[1]
    theta = np.random.randn(n,1)
    error_list = []

    for i in range (max_itres):
        yp = hypothesis(X,theta)
        e = error(y, yp)
        error_list.append(e)
        grad = gradient(X,y,yp)
        theta = theta - learning_rate*grad

    plt.plot(error_list)
    plt.show()
    return theta


def predict (X,theta):
    h = hypothesis(X,theta)
    preds = np.zeros((X.shape[0], 1), dtype='int')
    preds[h>=0.5] = 1

    return preds

def accuracy(X,y,theta):
    preds = predict(X,theta)
    return((y==preds).sum())/ y.shape[0]*100


if __name__ == "__main__":
    # step 1 = Generate Toy(dummy) Dataset
    X, y = make_blobs(n_samples=2000, n_features=2, cluster_std=3,centers=2, random_state= 42)  #centres here denotes number of clusters you need for the desired datasets
    X = normalize(X)
    # Step -4 Train Test Split
    Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.3, shuffle=False, random_state=0)
    print(X.shape, y.shape)
    print(y)
    Xtrain = preprocess(Xtrain)
    Xtest = preprocess(Xtest)
    ytrain = ytrain.reshape(-1,1)
    ytest = ytest.reshape(-1,1)
    # visualize(X,y)
    theta = train (Xtrain, ytrain)


    plt.scatter(Xtrain[:,1], Xtrain[:,2], c=ytrain, cmap = 'seismic')

    x1 = np.linspace(-3,3,6)
    x2 = -(theta[0] + theta[1]*x1)/theta[2]
    plt.plot(x1,x2)
    plt.show()

    preds = predict(Xtest, theta)
    print(accuracy(Xtrain,ytrain,theta))
    print(accuracy(Xtest,ytest,theta))

