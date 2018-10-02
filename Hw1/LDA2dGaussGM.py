import sys, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random
from statistics import stdev

from LDA import LDA

from sklearn.datasets import load_digits


def cross_val_split(data, folds, index):
	'''
	Function to split the data into train-test sets for k-fold cross-validation
	'''
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

if __name__ == '__main__':
	filename = sys.argv[1]
	if filename == 'digits':
		# Load in the data
		digits = load_digits()
		data = pd.DataFrame(digits.data)
		label = 'class'
		data[label] = digits.target

		# Project the data into R2 (two-dimensional)
		lda = LDA(data, label=label, n_dims=2)

		# Cross-validation
		n_folds = int(sys.argv[2])

		# Get train and test error rates
		train_err, test_err = lda.gaussian_errors(cv=n_folds)
		
		# Print out the results
		print("Train error is % s, with standard deviation % s" 
			% (np.mean(train_err), stdev(train_err)))
		print("Test error is % s, with standard deviation % s" 
			% (np.mean(test_err), stdev(test_err)))
	else:
		print("Please enter a valid file name.")


