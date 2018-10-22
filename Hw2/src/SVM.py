import numpy as np 
import pandas as pd
from cvxopt import matrix, solvers
import math
import random
from data import *
from sklearn.decomposition import PCA

MIN_LAGRANGE = 1e-8

class softMarginSVM():
	# https://cvxopt.org/applications/svm/index.html
	def __init__(self, data, C):
		self.C = C 
		self.X, self.Y, self.y = ppData(data)
		# # Project X into 2 dimensions
		pca = PCA(n_components=2)
		self.X_pca = pca.fit_transform(self.X)

	def fit(self, X, y):
		'''
		Calculate weights and bias for soft margin SVM
		'''
		N, D = X.shape 

		# Linear SVM
		P = matrix(np.outer(y,y) * np.dot(X, X.T))
		q = matrix(-np.ones(N))

		# inequality constraint a in [0,C]
		G = matrix(np.vstack((-np.eye(N),np.eye(N))))
		h = matrix(np.hstack((np.zeros(N), np.ones(N) * self.C )))

		# equality constraint
		A = matrix(y, (1,N))
		b = matrix(0.0)

		# solve QP
		solution = solvers.qp(P, q, G, h, A, b)
		# Lagrange a
		a = np.array(solution['x'])

		# only non-zero alphas for support vectors
		sv = a > MIN_LAGRANGE
		self.support_ = np.where(sv)[0]
		self.alphas = a[sv]
		
		# weights
		self.w = np.sum(a * y * X, axis = 0).reshape(-1,1)
		# bias
		self.b = np.mean(y[sv] - np.dot(X[self.support_], self.w))

	def predictSVM(self, x):
		x = x.reshape(-1,1)
		return np.sign(np.dot(self.w.T, x)+self.b)
	
	def calculate_error(self, X_test, y_test):
		y_pred = []
		for x in X_test:
			pred = self.predictSVM(x)
			y_pred.append(pred)
		error = np.sum(y_pred != y_test) / float(len(y_test))
		return error

class SVMPegasos():
	def __init__(self, k, lambd=10):
		self.k = k
		self.lambd = lambd
		self.w = np.zeros(())

	def fit(self, X, y):
		'''
		Calculate weights for Pegasos
		----------------------------------------------
		Input:
			X: numpy array, shape(N,D)
			y: numpy array, shape(1,N) - binary (-1 or 1)
		'''
		N, D = X.shape
		ktot = 1	# total number of iterations
		# initialize weights
		self.w = np.zeros((D,1))
		self.w.fill(1/math.sqrt(self.lambd))

		# iteratively until termination
		for t in range(1,ktot+1):
			# create mini-batch size k
			# TO-DO: pick k% from each class
			batch = random.sample(range(N), self.k)
			X_t, y_t = X[batch], y[batch] 
			eta = 1./ (self.lambd*t)

			plus = np.where(y_t * np.dot(X_t, self.w) < 1)[0]
			X_plus, y_plus = X_t[plus], y_t[plus]

			w_half = (1-eta*self.lambd)*self.w + (eta/self.k)*np.dot(X_plus.T,y_plus)

			if sum(w_half**2) < 1e-8:
				w_half = np.maximum(w_half, 1e-3)

			w_next = w_half * np.minimum(1, (1/math.sqrt(self.lambd))/math.sqrt(sum(w_half**2)))

			self.w = w_next
