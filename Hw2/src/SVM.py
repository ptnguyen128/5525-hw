import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
import math, random, time, pickle
from data import *

MIN_LAGRANGE = 1e-8

class softMarginSVM():
	def __init__(self, data, C):
		self.C = C
		self.X, self.y = ppData(data)

	def fit(self, X, y):
		'''
		Calculate weights and bias for soft margin SVM
		@X: numpy array, shape (N,D)
		@y: numpy array, shape (N,1)
		'''
		N, D = X.shape

		# Linear SVM
		K = y * X
		P = matrix(np.dot(K, K.T))
		q = matrix(-np.ones(N))

		# inequality constraint a in [0,C]
		G = matrix(np.vstack((-np.eye(N),np.eye(N))))
		h = matrix(np.hstack((np.zeros(N), np.ones(N) * self.C )))

		# equality constraint
		A = matrix(y.reshape(1, -1))
		b = matrix(0.0)

		# solve QP
		solvers.options['show_progress'] = False
		solution = solvers.qp(P, q, G, h, A, b)

		# Lagrange multiplier a
		a = np.array(solution['x'])
		# only non-zero alphas for support vectors
		sv = a > MIN_LAGRANGE
		self.support_ = np.where(sv)[0]
		self.alphas = a[sv]
		# print(self.alphas)

		# weights
		self.w = np.sum(a * y * X, axis = 0).reshape(-1,1)
		# bias
		self.b = np.mean(y[sv] - np.dot(X[self.support_], self.w))

	def predictSVM(self, x):
		x = x.reshape(-1,1)
		return np.sign(np.dot(self.w.T, x)+self.b)

	def calculateMistakes(self,X_test, y_test):
		y_pred = []
		for x in X_test:
			pred = np.sign(np.dot(self.w.T, x)+self.b)
			y_pred.append(pred)
		return np.sum(y_pred != y_test)

	def calculate_error(self, X_test, y_test):
		mistakes = self.calculateMistakes(X_test, y_test)
		return mistakes / float(len(y_test))

class SVMPegasos():
	def __init__(self, k, lambd, n_runs):
		self.k = k
		self.lambd = lambd
		self.n_runs = n_runs

	def calculateLogLoss(self,X,y,w,lambd):
		N = X.shape[0]
		norm_sq = np.linalg.norm(w)**2
		loss = 0.0
		for i in range(N):
			pred = 1 - y[i]*np.dot(w.T,X[i])
			loss += np.log(1 + np.exp(pred))
		return loss/N + lambd*norm_sq

	def fit(self, data):
		'''
		Updates weight and loss, terminates when converge or reaches
		maximum number of iterations (ktot)
		'''
		N, D = data.shape
		ktot = 200 * N	# maximum number of iterations

		run_times = []

		for i in range(self.n_runs):
			print("Running #%s" % i)

			# initialize weights
			self.w = np.zeros((D-1,1))
			self.w.fill(1/math.sqrt(self.lambd))

			start_time = time.time()
			loss_values = []

			# iteratively until termination
			for t in range(1,ktot+1):
				# pick k% from each class and create mini-batch
				X_t, y_t = generateBatch(data, self.k)

				eta = 1./ (self.lambd*t)

				# where X suffers a non-zero loss
				plus = np.where(y_t * np.dot(X_t, self.w) < 1)[0]
				X_plus, y_plus = X_t[plus], y_t[plus]

				# scale w, and add w to (y*eta/k)x for all samples in A_t_plus
				w_half = (1-eta*self.lambd)*self.w + (eta/self.k)*np.dot(X_plus.T,y_plus)

				# handles division by 0
				if sum(w_half**2) < 1e-8:
					w_half = np.maximum(w_half, 1e-3)

				# w_{t+1}
				w_next = w_half * np.minimum(1, (1/math.sqrt(self.lambd))/math.sqrt(sum(w_half**2)))

				# update w for this iteration
				self.w = w_next

				# calculate log loss and defines convergence condition
				loss = self.calculateLogLoss(X_t, y_t, self.w, self.lambd)
				if t > 1:
					if abs(loss - previous_loss) < 1e-5:
						print("Converged...")
						break

				previous_loss = loss

				loss_values.append(loss)

			# pickle this value
			pickle_on = open("./pickle/PegasosLoss/loss_run_%s_k_%s.pickle" % (i,self.k), 'wb')
			pickle.dump(loss_values, pickle_on)
			pickle_on.close()

			end_time = time.time()
			run_times.append(end_time - start_time)

		run_times = np.array(run_times)

		return run_times

class SVMSoftplus():
	def __init__(self, k, lambd, n_runs):
		self.k = k
		self.lambd = lambd
		self.n_runs = n_runs

	def calculateLogLoss(self,X,y,w,lambd):
		N = X.shape[0]
		norm_sq = np.linalg.norm(w)**2
		loss = 0.0
		for i in range(N):
			pred = 1 - y[i]*np.dot(w.T,X[i])
			loss += np.log(1 + np.exp(pred))
		return loss/N + lambd*norm_sq

	def calculateGradient(self, w, y, X, lambd):
		N = len(y)
		gradient = np.zeros((w.shape))
		for i in range(0,N):
			exp = np.exp( 1 - y[i] * np.dot(w,X[i]) )
			grad = -(y[i] * X[i] / (1+exp))
			gradient += grad
		return (1/N)*gradient + 2*lambd*w

	def fit(self, data):
		'''
		Updates weight and loss, terminates when converge or reaches
		maximum number of iterations (ktot)
		'''
		N, D = data.shape
		if self.k == 1:
			ktot = 500		# algorithm never reaches convergence otherwise
		else:
			ktot = 200 * N	# maximum number of iterations

		run_times = []

		for i in range(self.n_runs):
			print("Running #%s" % i)

			# initialize weights
			self.w = np.zeros(D-1)
			self.w.fill(1/math.sqrt(self.lambd))

			start_time = time.time()
			loss_values = []

			# iteratively until termination
			for t in range(1,ktot+1):
				# print("Iteration %s" % t)
				# pick k% from each class and create mini-batch
				X_t, y_t = generateBatch(data, self.k)

				eta = 1./ np.sqrt(ktot)
				gradient = self.calculateGradient(self.w, y_t, X_t, self.lambd)
				_w = self.w - eta* gradient

				self.w = _w

				# calculate log loss and defines convergence condition
				loss = self.calculateLogLoss(X_t, y_t, self.w, self.lambd)
				# print(loss)
				if t > 1:
					if abs(loss - previous_loss) < 1e-5:
						print("Converged...")
						break

				previous_loss = loss

				loss_values.append(loss)

				# pickle this value
				pickle_on = open("./pickle/SoftplusLoss/loss_run_%s_k_%s.pickle" % (i,self.k), 'wb')
				pickle.dump(loss_values, pickle_on)
				pickle_on.close()

			end_time = time.time()
			run_times.append(end_time - start_time)

		run_times = np.array(run_times)

		return run_times
