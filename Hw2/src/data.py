import pandas as pd, numpy as np
from sklearn.utils import shuffle as pd_shuffle
import random
import math

def load_data(filename):
    data = pd.read_csv(filename, sep=",", header=None)
    return data

def separateFolds(data, n_folds=10):
    '''
    Helper function to divide data into n folds,
    from which choose 2 folds to be test data, 1 to be validation,
    and the rest to be train data
    '''
    # randomly shuffle data
    # data_s = pd_shuffle(data)

    folds = {}
    # length of each fold
    fold_len = int(data.shape[0]/n_folds)
    for i in range(n_folds):
        folds[i] = data.iloc[i*fold_len:(i+1)*fold_len,:]

    # shuffle fold indices
    idx = list(range(10))
    random.shuffle(idx)
    # test set
    test = pd.concat([folds[idx[0]], folds[idx[1]]])
    # validation set
    val = folds[idx[2]]
    train_folds = []
    # train set
    for i in idx[3:]:
        train_folds.append(folds[i])
    train = pd.concat(train_folds)

    return train, val, test

def ppData(data):
    '''
    Helper function to pre-process the input data, shape(N,D+1)
    -----------------------------------------------------------
    Output:
        X: numpy array, shape(N,D)
        y: numpy array, shape(1,N) - binary
    '''
    X = np.array(data.iloc[:,1:], dtype=np.float128)
    # # normalize X
    # mean = X.mean(axis=0)
    # X = X - mean

    classes = np.unique(data.iloc[:,0])
    # re-assign classes to -1 or 1 (1 if class is 1, -1 if class is 3)
    y = np.array([1.0 if num == classes[0] else -1.0 for num in data.iloc[:,0]], dtype=np.float128)
    y = np.reshape(y, [X.shape[0],1])
    return X, y

def generateBatch(data, k):
    '''
    Helper function to get random % of each class
    ---------------------------------------------
    Output:
        X_batch: np array, shape(k/2, D)
        y_batch: np array, shape(k/2, 1)
    '''
    N= data.shape[0]
    percent = float(k) / N

    if k == 1:
        X,y = ppData(data)
        batch = random.sample(range(N),k)
        X_batch, y_batch = X[batch], y[batch]

    else:
        X_batch = {}
        y_batch = {}

        classes = np.unique(data.iloc[:,0])
        for c in classes:
            df = data[data.iloc[:,0]==c]
            X_c = np.array(df.iloc[:,1:], dtype=np.float128)
            if c == classes[0]:
                y_c = np.array([1]*len(X_c), dtype=np.float128)
            else:
                y_c = np.array([-1]*len(X_c), dtype=np.float128)
            y_c = np.reshape(y_c, [X_c.shape[0],1])

            batch_len = int(percent * df.shape[0])
            batch = random.sample(range(len(df)), batch_len)
            X_batch[c], y_batch[c] = X_c[batch], y_c[batch]

        X_batch = np.concatenate((X_batch[classes[0]],X_batch[classes[1]]),axis=0)
        y_batch = y_t = np.concatenate((y_batch[classes[0]],y_batch[classes[1]]))

    return X_batch, y_batch
