import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

class LDA():
    def __init__(self, data, label, n_dims):
        '''
        Class initializer
        -------------------------------------
        Input:
        data: the dataset
        label: name of the target variable
        cv: number of folds for cross-validation
        -------------------------------------
        '''
        self.data = data
        self.label = label
        self.n_dims = n_dims

    def fit_transform(self):
        '''
        Function to return the projected input data into n dimensions
        '''
        # get the groups
        grouped = self.data.groupby(self.data.loc[:,self.label])
        self.classes = [k for k in grouped.groups.keys()]

        # mean for each class:
        data_class = {}
        X = {}
        means = {}
        for k in self.classes:
            data_class[k] = grouped.get_group(k)
            X[k] = data_class[k].drop(self.label,axis=1)
            means[k] = np.array(np.mean(X[k]))

        self.means = means

        # mean of the total data
        mean_total = np.array(np.mean(self.data.drop(self.label,axis=1)))

        # between covariance matrix
        S_B = np.zeros((X[0].shape[1], X[0].shape[1]))
        for k in X.keys():
            S_B += np.multiply(len(X[k]),
                                np.outer((means[k] - mean_total),
                                np.transpose(means[k] - mean_total)))

        # within covariance matrix
        S_W = np.zeros((X[0].shape[1], X[0].shape[1]))
        for k in self.classes:
            S_k = X[k] - means[k]
            S_W += np.dot(S_k.T, S_k)

        # eigendecomposition
        S = np.dot(np.linalg.pinv(S_W), S_B)
        eigval, eigvec = np.linalg.eig(S)

        # sort eigenvalues in decreasing order
        eig = [(eigval[i], eigvec[:,i]) for i in range(len(eigval))]
        eig = sorted(eig, key=lambda x: x[0], reverse=True)

        # only take the top (number of dimensions projected into) vectors
        w = np.array([eig[i][1] for i in range(self.n_dims)])

        new_data = {}
        for k in X.keys():
            new_data[k] = np.dot(X[k], w.T)

        return new_data
