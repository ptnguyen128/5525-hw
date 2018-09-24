import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

class LDA():
    def __init__(self, data, num_dims, label, cv):
        '''
        Class initializer
        -------------------------------------
        Input:
        data: the dataset
        num_dims: number of dimensions to project into
        label: name of the target variable
        cv: number of folds for cross-validation
        -------------------------------------
        Output:
        '''
        self.data = data
        self.num_dims = num_dims
        self.label = label
        self.cv = cv

    def fit(self):
        '''
        Function to fit the data and perform LDA
        '''
        def calculate_parameters(data):

            # get the groups
            grouped = data.groupby(data.loc[:,self.label])
            self.classes = [k for k in grouped.groups.keys()]

            # mean for each class:
            data_class = {}
            X = {}
            means = {}
            for k in self.classes:
                data_class[k] = grouped.get_group(k)
                X[k] = data_class[k].drop(self.label,axis=1)
                means[k] = np.array(np.mean(X[k]))

            # mean of the total data
            mean_total = np.array(np.mean(data.drop(self.label,axis=1)))

            # between covariance matrix
            S_B = np.zeros((X[0].shape[1], X[0].shape[1]))
            for k in X.keys():
                S_B += np.multiply(len(X[c]),
                                    np.outer((means[c] - mean_total),
                                    np.transpose(means[c] - mean_total)))

    # Split the data into k folds
    def cross_val_split(data, folds, index):
        data_idx = []
        indices = [i for i in range(data.shape[0])]

        fold_size = int(len(data)/folds)
        for i in range(folds):
            fold_idx = []
            while len(fold_idx) < fold_size:
                idx = random.randrange(len(indices))
                fold_idx.append(indices.pop(idx))
            data_idx.append(fold_idx)

        test_idx = data_idx[index]
        del data_idx[index]
        train_idx = [item for sublist in data_idx for item in sublist]

        test = data.iloc[test_idx]
        train = data.iloc[train_idx]

        return train,test

    # k-fold cross-validation
    for i in range(self.cv):
        train, test = cross_val_split(self.data,self.cv,i)
