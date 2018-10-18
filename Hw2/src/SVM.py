import numpy as np 
import pandas as pd
from cvxopt import matrix, solvers

MIN_LAGRANGE = 1e-8

class softMarginSVM():
	# https://cvxopt.org/applications/svm/index.html
	def __init__(self, C):
		self.C = C 

	def fit(self, X, y):
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
	
	def calculate_error(self, X_test, y_test):
		y_pred = []
		for x in X_test:
			pred = np.sign(np.dot(self.w.T, x)+self.b)
			y_pred.append(pred)
		error = np.sum(y_pred != y_test) / float(len(y_test))
		return error
