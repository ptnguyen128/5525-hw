import sys

import pandas as pd, numpy as np
import random

from train_test_split import train_test_split
from data import load_data

class GaussianModel():
    def __init__(self, data):
        '''
        Class initializer
        -------------------------------
        '''
        self.data = data
        self.label = label

        def to_dict(self, df):
            '''
            Function to turn a dataframe into a dictionary
            '''
            # get the groups
            grouped = df.groupby(df.loc[:,self.label])
            self.classes = [k for k in grouped.groups.keys()]

            # mean for each class:
            data_class = {}
            X = {}
            for k in self.classes:
                data_class[k] = grouped.get_group(k)
                X[k] = data_class[k].drop(self.label,axis=1)

            return X

        def calculate_gaussian(self, data):
            '''
            Function to perform generative Gaussian modeling and
            calculate class prior, mean, covariance
            '''
            # Project the original data into n dimensions
            X = self.to_dict(data)

            g_prior = {}
            g_mean = {}
            g_cov = {}

            # for each class
            for k in X.keys():
                # prior
                g_prior[k] = X[k].shape[0] / data.shape[0]
                # mean
                g_mean[k] = np.mean(X[k], axis = 0)
                # covariance
                g_cov[k] = np.cov(X[k],rowvar=False)
            return g_prior, g_mean, g_cov
