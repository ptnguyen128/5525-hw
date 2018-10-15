import sys

import numpy as np 
import pandas as pd
from cvxopt import matrix, solvers

from train_test_split import *
from SVM import softMarginSVM

NUM_SPLITS = 1
N_FOLDS = 10

def ppData(data):
	X = np.array(data.iloc[:,1:])
	classes = np.unique(data.iloc[:,0])
	# re-assign classes to -1 or 1 (1 if class is 1, -1 if class is 3)
	y = [1.0 if num == classes[0] else -1.0 for num in data.iloc[:,0]]
	y = np.reshape(y, [X.shape[0],1])
	return X, y


def myDualSVM(filename):
	data = pd.read_csv(filename, sep=",", header=None)

	#C = [0.01, 0.1, 1, 10, 100]
	C = [0.01, 0.1]

	# Random 80-20 train-test split
	for i in range(NUM_SPLITS):
		print("Split #",i)
		train, test = train_test_split(data, test_ratio=0.2)

		err_c = {}
		
		# 10-fold cross-validation
		for c in C:
			print("Performing %s-fold cross_validation with C=%s" % (N_FOLDS,c))

			errors = []

			for k in range(N_FOLDS):
				cv_train, cv_val = cross_val_split(train, folds=10, index=k)
				
				X_train, y_train = ppData(cv_train)
				X_val, y_val = ppData(cv_val)
				
				SVM = softMarginSVM(C=c)
				SVM.fit(X_train, y_train)
				errors.append(SVM.calculate_error(X_val, y_val))
			
			err_c[c] = np.mean(errors)

	return err_c

if __name__ == '__main__':
	filename = sys.argv[1]
	myDualSVM(filename)