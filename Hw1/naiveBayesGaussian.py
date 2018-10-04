import sys

import pandas as pd, numpy as np
import random

from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

from train_test_split import train_test_split

class NaiveBayes():
    def __init__(self, X, y):
        '''
        Class initializer
        -------------------------------------
        Input:
        X: input data
        y: label data
        -------------------------------------
        '''
        self.X = X
		self.N, self.D = X.shape

		self.y = y
		self.classes = np.unique(y)
		self.K = len(self.classes)
		# turn labels into one-hot-coding
		self.t_one_hot = np.zeros((self.N,len(self.classes)))
		self.t_one_hot[np.arange(self.N), self.y] = 1				# shape (N, K)

    def to_dict(self, X, y):
        return {k: X_train.iloc[np.where(y==k)[0]] for k in self.classes}

    def gaussian_params(self):
        X_dict = self.to_dict(self.X, self.y)
        self.g_prior = {k: X_dict[k].shape[0]/self.X.shape[0] for k in self.classes}
        self.g_mean = {k: np.mean(X_dict[k], axis=0) for k in self.classes}
        self.g_std = {k: np.std(X_dict[k],axis=0, ddof=1) for k in self.classes}
        return

    def gaussian_pdf(self, x, mean, std):
        '''
        Function to calculate a Gaussian class-conditional density
        for point x
        '''
        A = 1/((2*np.pi)**(len(x)/2))
        B = 1/(np.prod(std))
        C = - np.sum(((x - mean)**2) / (2 * (std**2)))
        return A*B*np.exp(C)

    def predict(self, X):
        y_pred = []
        # for each data point
        for idx, x in X.iterrows():
            likelihood = []
            # likelihood for each class
            for k in self.classes:
                p = g_prior[k] * self.gaussian_pdf(x, g_mean[k], g_std[k])
                likelihood.append(np.log(p+1e-6))

            # predict label (one with highest likelihood)
            y_pred.append(np.argmax(likelihood))
        return y_pred
