import pandas as pd, numpy as np

def ppData(data):
    '''
    Helper function to pre-process the input data, shape(N,D+1)
    -----------------------------------------------------------
    Output: 
        X: numpy array, shape(N,D)
        y: numpy array, shape(1,N) - binary
    '''
    X = np.array(data.iloc[:,1:])

    classes = np.unique(data.iloc[:,0])
    Y = data.iloc[:,0]
    # re-assign classes to -1 or 1 (1 if class is 1, -1 if class is 3)
    y = [1.0 if num == classes[0] else -1.0 for num in Y]
    y = np.reshape(y, [X.shape[0],1])
    return X, Y, y

def load_data(filename):
    data = pd.read_csv(filename, sep=",", header=None)
    return data
