import sys

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

class LogisticRegression():
	def __init__(self, X, y):
		'''
		Class initializer
		-------------------------------------
		Input:
		data: the dataset
		label: name of the target variable
		-------------------------------------
		'''
		self.X = X
		self.N, self.D = X.shape
		# add a column of ones to X
		ones = np.ones(N).reshape(1,N)
		self.phi = np.concatenate((ones,self.X), axis=1)		# shape (N, D+1)

		self.y = y
		self.classes = np.unique(y)
		self.K = len(self.classes)
		# turn labels into one-hot-coding
		self.t_one_hot = np.zeros((N,len(self.classes)))
		self.t_one_hot[np.arange(N), self.y] = 1				# shape (N, K)

		self.max_iter = 50
		self.threshold = 1e-10

		# initialize weights
		self.W = 0.001 * np.random.random((D+1, len(classes)))	# shape (D+1, K)


	# Sigmoid function
	def sigmoid(self, a):
		return 1/(1 + np.exp(-a))

	# Softmax function
	def softmax(self, a_k):
		e = np.exp(a - np.max(a, axis=1).reshape((-1,1)))
		e_total = np.sum(e, axis=1).reshape((-1,1))
		return e / e_total

	def IRLS(self, data):

		for i in range(self.max_iter):
			# activations
			a = np.dot(self.phi, self.W)								# shape (N, K)
			# posterior
			p = softmax(a)									# shape (N, K)

			# cross-entropy (we want to minimize this)
			E = - np.sum(self.t_one_hot * np.log(p + 1e-6))	

		return E

	def binary_logreg(self, data):
		'''
		Function to update weights
		'''
		# input data and label
		t = np.array(data[label])	# shape (N,)
		X = data.drop(self.label, axis=1)	# shape (N, D)
		# get dimensions of X
		N = X.shape[0]		# number of observations (rows)
		D = X.shape[1]		# number of features (columns)
		# add a column of ones to X
		ones = np.array([[1]*N]).T 					# shape (1, N)
		phi = np.concatenate((ones,X), axis=1)		# shape (N, D+1)
		# initialize the weights
		w = np.zeros(D+1)		# shape (D+1,)
		#w = np.random.randn(D+1)	# shape (D+1,)

		for i in range(self.max_iter):
			# get activations
			a = np.dot(phi,w)			# shape (N,)
			# posterior
			p = self.sigmoid(a)			# shape (N,)

			# cross-entropy (we want to minimize this)
			E = - np.sum(t*safe_ln(p) + (1-t)*safe_ln(1-p))

			# weighing matrix
			R = np.diag(p * (1 - p))	# shape (N, N)

			# update new w
			z = np.dot(phi, w) - np.dot((p-t), np.linalg.pinv(R)) 		# shape (N,)
			w = np.linalg.pinv(np.dot(np.dot(phi.T,R),phi)).dot(phi.T).dot(R).dot(z)	# shape (D+1,)

			new_p = self.sigmoid(np.dot(phi,w))
			new_E = -np.sum(t*safe_ln(new_p) + (1-t)*safe_ln(1-new_p))

			print(new_E - E)

			# If converge
			if new_E - E < self.threshold:
				print("Iteration: ", i)
				print("Updated weight: ", w)
				break


def to_label(data, target, percentile):
	'''
	Input: data, name of target column, percentile to partition data
	Output: data, but with the target column values
	changed from continuous to categorial (classes)
	'''
	frac = percentile / 100.0
	part_val = data[target].quantile(frac)
	data[target] = [1 if d > part_val else 0 for d in data[target]]
	return data

def train_test_split(data, label, test_ratio=0.2):
	'''
	Fuction to split the dataset into train-test sets
	based on the specified size of the test set
	'''
	test_idx = []
	indices = [i for i in range(data.shape[0])]

	test_size = test_ratio * len(data)
	while len(test_idx) < test_size:
		test_idx.append(random.randrange(len(indices)))

	train_idx = [i for i in indices if i not in test_idx]

	test = data.iloc[test_idx]
	train = data.iloc[train_idx]

	y_train = train[label]
	X_train = train.drop(label,axis=1)

	y_test = test[label]
	X_test = test.drop(label,axis=1)

	return X_train, y_train, X_test, y_test


def logisticRegression(filename):
	'''
	Input:
		filename: boston / digits
		num_splits: number of 80-20 train-test splits for evaluation
		train_percent: vector containing percentages of training data to be used for training
	------------------------------------------------------------
	Output:
		test set error rates for each training set percent
	'''
	if filename == 'digits':
		# Load in the data
		digits = load_digits()
		data = pd.DataFrame(digits.data)
		label = 'class'
		data[label] = digits.target

	elif filename == 'boston':
		# Load in the data
		boston = load_boston()
		data = pd.DataFrame(boston.data, columns=boston.feature_names)
		label = 'HomeVal50'
		data[label] = boston.target
		# Transform the target variable to binary
		to_label(data, label, 50)

	else:
		print("Please enter a valid file name.")

	# Split the dataset into 80-20 train-test sets
	X_train, y_train, X_test, y_test = train_test_split(data, label=label)
	print(X_train.head(5))

if __name__ == '__main__':
	filename = sys.argv[1]

	logisticRegression(filename)
