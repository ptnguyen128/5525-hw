import sys

import pandas as pd, numpy as np
import random
import pickle

from train_test_split import train_test_split
from data import load_data

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

    def to_dict(self, X, y):
        return {k: X.iloc[np.where(y==k)[0]] for k in self.classes}

    def gaussian_params(self):
        X_dict = self.to_dict(self.X, self.y)
        self.g_prior = {k: X_dict[k].shape[0] / self.X.shape[0] for k in self.classes}
        self.g_mean = {k: np.mean(X_dict[k], axis=0) for k in self.classes}
        self.g_std = {k: np.std(X_dict[k], axis=0, ddof=1) for k in self.classes}
        return

    def gaussian_pdf(self, x, mean, std):
        '''
        Function to calculate a Gaussian class-conditional density
        for point x
        '''
        A = 1/((2*np.pi)**0.5)
        B = 1/(np.prod(std)+1e-6)
        C = - np.sum(((x - mean)**2) / (2 * (std**2)))
        return A*B*np.exp(C)

    def predict(self, X):
        y_pred = []
        # for each data point
        for idx, x in X.iterrows():
            likelihood = []
            # likelihood for each class
            for k in self.classes:
                p = self.g_prior[k] * self.gaussian_pdf(x, self.g_mean[k], self.g_std[k])
                likelihood.append(p)

            # predict label (one with highest likelihood)
            y_pred.append(np.argmax(likelihood))
        return y_pred

    def calculate_error(self, X_test, y_test):
        y_pred = self.predict(X_test)
        error = np.sum(y_pred != y_test) / float(len(y_test))
        return error


def naiveBayesGaussian(filename, num_splits, train_percent):
	'''
	Input:
	filename: boston / digits
	num_splits: number of 80-20 train-test splits for evaluation
	train_percent: vector containing percentages of training data to be used for training
	------------------------------------------------------------
	Output:
	test set error rates for each training set percent
	'''
	data, label = load_data(filename)

	error_matrix = np.zeros((num_splits, len(train_percent)))

	for i in range(num_splits):
		# Split the dataset into 80-20 train-test sets
		X_train, y_train, X_test, y_test = train_test_split(data, label=label)

		for j,p in enumerate(train_percent):
			# subset the training set
			train_max_idx = int(np.floor(p/100.00 * X_train.shape[0]))
			X_train_p = X_train.loc[:train_max_idx]
			y_train_p = y_train.loc[:train_max_idx]

			# fit Logistic Regression model
			naive_bayes = NaiveBayes(X_train_p, y_train_p)
			naive_bayes.gaussian_params()

			# calculate test error
			error_matrix[i,j] = naive_bayes.calculate_error(X_test, y_test)

	pickle_on = open("./pickle/%s_nb_error_matrix.pickle" % filename, 'wb')
	pickle.dump(error_matrix, pickle_on)
	pickle_on.close()

	mean_error = np.mean(error_matrix, axis=0)
	std_error = np.std(error_matrix, axis=0, ddof=1)

	return mean_error, std_error

if __name__ == '__main__':
    filename = sys.argv[1]
    num_splits = int(sys.argv[2])

    mean_error, std_error = naiveBayesGaussian(filename, num_splits, [10, 25, 50, 75, 100])

    print("Mean test errors are %s , with standard errors % s" % (mean_error, std_error))
