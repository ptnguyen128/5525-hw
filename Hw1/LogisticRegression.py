import sys

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

class LogisticRegression():
	def __init__(self, data, label):
		'''
		Class initializer
		-------------------------------------
		Input:
		data: the dataset
		label: name of the target variable
		-------------------------------------
		'''
		self.data = data
		self.label = label
		self.max_iter = 50
		self.threshold = 1e-10

	def to_dict(self, df):
	    '''
	    Function to turn a dataframe into a dictionary
	    '''
	    # get the groups
	    grouped = df.groupby(df.loc[:,self.label])
	    self.classes = [k for k in grouped.groups.keys()]

	    # for each class
	    data_class = {}
	    X = {}
		t = {}
	    for k in self.classes:
	        data_class[k] = grouped.get_group(k)
			t[k] = data_class[k][self.label]
	        X[k] = data_class[k].drop(self.label,axis=1)
	    return X, t

	# Sigmoid function
	def sigmoid(self, a):
		return 1/(1 + np.exp(-a))

	# Softmax function
	def softmax(self, a_k):
		e_k = np.exp(a_k)
		e_total = np.sum([np.exp(a[i]) for i in classes])
		return e_k / e_total

	def multi_logreg(self, data):
		# input data and label
		t = np.array(data[label])	# shape (N,)
		X = data.drop(self.label, axis=1)	# shape (N, D)
		# get dimensions of X
		N = X.shape[0]		# number of observations (rows)
		D = X.shape[1]		# number of features (columns)
		# add a column of ones to X
		ones = np.array([[1]*N]).T 					# shape (1, N)
		phi = np.concatenate((ones,X), axis=1)		# shape (N, D+1)

		# group the data by classes
		X_dict, t_dict = self.to_dict(data)

		w = {}
		a = {}
		p = {}
		E = 0
		# for each class
		for i in self.classes:
			# initialize weights
			w[i] =  0.001* np.random.randn(D+1)		# shape (D+1,)
			# activations for each class
			a[i] = np.dot(phi, w[i])				# shape (N,)
			# get posteriors
			p[i] = self.softmax(a[i])				# shape (N,)

			# cross-entropy (we want to minimize this)
			E -= t_dict[i]*safe_ln(p[i])

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


def safe_ln(p):
    for x in p:
        if x <= 0:
            return 0
        return np.log(x)

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

if __name__ == '__main__':
	filename = sys.argv[1]
	if filename == 'digits':
		# Load in the data
		digits = load_digits()
		data = pd.DataFrame(digits.data)
		label = 'class'
		data[label] = digits.target

		print(data.head(5))

	elif filename == 'boston':
		# Load in the data
		boston = load_boston()
		data = pd.DataFrame(boston.data, columns=boston.feature_names)
		label = 'HomeVal50'
		data[label] = boston.target
		# Transform the target variable to binary
		to_label(data, label, 50)

		log_reg = LogisticRegression(data, label)
		log_reg.binary_logreg(data)

	else:
		print("Please enter a valid file name.")
