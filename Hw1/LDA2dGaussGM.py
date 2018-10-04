import sys
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

from LDA import LDA

from data import load_data
from train_test_split import cross_val_split

if __name__ == '__main__':
	filename = sys.argv[1]

	data, label = load_data(filename)

	# Project the data into R2 (two-dimensional)
	lda = LDA(data, label=label, n_dims=2)

	# Cross-validation
	n_folds = int(sys.argv[2])

	# Get train and test error rates
	train_err, test_err = lda.gaussian_errors(cv=n_folds)

	# Print out the results
	print("Train error is % s, with standard deviation % s"
		% (np.mean(train_err), np.std(train_err)))
	print("Test error is % s, with standard deviation % s"
		% (np.mean(test_err), np.std(test_err)))
