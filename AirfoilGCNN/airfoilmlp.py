#%%
import numpy as np 

import os
from matplotlib import pyplot as plt



import pandas as pd

import torch 

import torch.nn
import tensorflow as tf
from tensorflow import keras
from sklearn import model_selection as ms
import sklearn
from keras import backend


params = { 'font.size'        : 12,
           'xtick.labelsize'  : 'small',
           'ytick.labelsize'  : 'small',
           'axes.linewidth'   : 1.3 }
plt.rcParams.update(params)

def train(X, y, dropout = 0, weight_decay = 0, lr = 0.1, activation = 'relu', n_layers = 5, neurons = 256, epochs = 1000):


    kf = ms.KFold(n_splits=5, shuffle=False, random_state=None)
    testr2s = []
    trainr2s = []
    
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        train_x = np.array(X_train, dtype = 'float')
        test_x = np.array(X_test, dtype = 'float')
        test_y = np.array(y_test, dtype = 'float')
        train_y =np.array(y_train, dtype = 'float')


        neurons = neurons
        list_layers = []
        for i in range(n_layers):
            list_layers.append(keras.layers.Dense(neurons, activation=activation))
        list_layers.append(keras.layers.Dense(1, activation='linear'))
        list_layers.append(keras.layers.Dropout(0.13))
        model = keras.Sequential(list_layers)
       
        sgd = keras.optimizers.Adam(lr = lr, decay = weight_decay)
        model.compile(optimizer=sgd,
                    loss='mean_squared_error',
                    metrics=['mae'])
        print("Training model for {} epochs...".format(epochs))
        model.fit(train_x, train_y.T, epochs=epochs, verbose = 0)
        test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=0)



        mean_ys = np.mean(y_test)
        prediction = model.predict(test_x, verbose =0)

        sstot = np.sum((y_test - mean_ys)**2)
        sse = np.sum((prediction[:,0] - y_test)**2)
        testr2 = 1 - sse/sstot

        print("R2", testr2)
        mae = tf.keras.losses.MeanAbsoluteError
        testr2s.append(testr2)
        




        mean_ys = np.mean(y_train)
        prediction = model.predict(train_x, verbose =0)
        sstot = np.sum((y_train - mean_ys)**2)
        sse = np.sum((prediction[:,0] - y_train)**2)
        trainr2 = 1 - sse/sstot
        mae = tf.keras.losses.MeanAbsoluteError
        trainr2s.append(trainr2)
        if backend.backend() == "tensorflow":
            backend.clear_session()
    print("Test R2: " + str(np.mean(testr2s)) + " ± "  + str(np.std(testr2s)) + "Train R2 = " + str(np.mean(trainr2s)) +" ± " + str(np.std(trainr2s)))
    

    return np.mean(testr2s)




def main():

    labels = pd.read_csv("./data/labels.csv")
    print(labels.iloc[:, 1])

    params = { 'font.size'        : 12,
            'xtick.labelsize'  : 'small',
            'ytick.labelsize'  : 'small',
            'axes.linewidth'   : 1.3 }
    plt.rcParams.update(params)

    nodes = 700


    X = np.zeros((0, 2*nodes))
    c = 0
    lim = 1500
    y = np.zeros(0)

    for label in labels.iloc[0:lim,0]:
        assert(label == labels.iloc[c, 0])
        if label == 'fx79w660a':
            c = c+1
            continue

        adjvel = np.loadtxt('./data/' + str(nodes)+ 'manadjvel/' + label  + '_adjvel', dtype = 'double').reshape(1,-1)
    

        meanorig = np.mean(labels.iloc[0:lim, 1])
        stdorig = np.std(labels.iloc[0:lim, 1])

        if (labels.iloc[c, 1] > meanorig + 2*stdorig):
            print("drag too high, filtered")
            c = c+1
            continue
        if (labels.iloc[c, 1] < meanorig - 3*stdorig):
            print(meanorig - 4*stdorig)
            print(labels.iloc[c, 1], c)
            print("drag too low, filtered")
            c = c+1
            continue

        else:
            X = np.vstack((X, adjvel))
            y = np.append(y,labels.iloc[c, 1])
    
        c = c+1

    print(X.shape)


    mean = np.mean(y)
    std = np.std(y)
    ynorm = (y - mean)/std
    y = ynorm




    train(X, y, lr = 0.00355, dropout =  0.0453, weight_decay = 0.000162, activation = 'relu', n_layers=4, neurons = 512)

if __name__ == "__main__":
    main()


