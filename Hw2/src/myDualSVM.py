import sys

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from data import *
from SVM import softMarginSVM

NUM_SPLITS = 7
C = [0.01, 0.1, 1, 10, 100]

def myDualSVM(filename):
	data = load_data(filename)

	test_errors = np.zeros((NUM_SPLITS, len(C)))

	# Randomly split train, validation, and test sets
	for i in range(NUM_SPLITS):
		print("Split #%d" % i)

		# helper function in data.py
		train, val, test = separateFolds(data)
		X_val, y_val = ppData(train)
		X_test, y_test = ppData(test)

		errors_train = []
		errors_val = []
		sv = []
		geometric_margin = []

		# for each C
		for j in range(len(C)):
			print("Training with C=%s" % C[j])
			# Train SVM model
			SVM = softMarginSVM(train, C=C[j])
			SVM.fit(SVM.X, SVM.y)

			# pickle this model
			pickle_on = open("./pickle/dualSVM/split_%s_C_%s.pickle" % (i,C[j]), 'wb')
			pickle.dump(SVM, pickle_on)
			pickle_on.close()

			# train - validation misclassification
			errors_train.append(SVM.calculateMistakes(SVM.X, SVM.y))
			errors_val.append(SVM.calculateMistakes(X_val, y_val))

			# geometric margin values
			geometric_margin.append(1 / np.linalg.norm(SVM.w))

			test_errors[i][j] = SVM.calculate_error(X_test, y_test)

			# number of support vectors
			sv.append(len(SVM.support_))

		# pickle these values
		pickle_on = open("./pickle/dualPlotParams/errors_train_split_%s.pickle" % i, 'wb')
		pickle.dump(errors_train, pickle_on)
		pickle_on.close()

		pickle_on = open("./pickle/dualPlotParams/errors_val_split_%s.pickle" % i, 'wb')
		pickle.dump(errors_val, pickle_on)
		pickle_on.close()

		pickle_on = open("./pickle/dualPlotParams/g_margin_split_%s.pickle" % i, 'wb')
		pickle.dump(geometric_margin, pickle_on)
		pickle_on.close()

		pickle_on = open("./pickle/dualPlotParams/n_sv_split_%s.pickle" % i, 'wb')
		pickle.dump(sv, pickle_on)
		pickle_on.close()

	# pickle the test error matrix
	pickle_on = open("./pickle/test_errors_dualSVM.pickle", 'wb')
	pickle.dump(test_errors, pickle_on)
	pickle_on.close()

	return test_errors

def unPickle(p_name):
	'''
	Load the pre-calculated values
	'''
	pkl = []
	for i in range(NUM_SPLITS):
		pickle_off = open("./pickle/dualPlotParams/%s_split_%s.pickle" % (p_name,i), 'rb')
		pkl.append(pickle.load(pickle_off))
		pickle_off.close()
	return np.array(pkl)

def plot_C(mean, std, C, label):
	plt.bar(C, mean, yerr=std*1.96, capsize=5)
	plt.ylabel(label)
	plt.xlabel('C')
	plt.show()

if __name__ == '__main__':
	file = sys.argv[1]
	filename = './data/%s.csv' % file

	# Train data and get test errors across Cs
	# test_err = myDualSVM(filename)		# uncomment if you want to run the program from sratch

	# uncomment these if load pre-trained data
	# Test errors
	pickle_on = open("./pickle/test_errors_dualSVM.pickle", 'rb')
	test_err = pickle.load(pickle_on)
	pickle_on.close()

	test_mean = np.mean(test_err, axis=0)
	# print(test_mean)
	test_std = np.std(test_err, axis=0, ddof=1)

	# Training - validation errors across C
	train_err = unPickle("errors_train")
	train_err_mean = np.mean(train_err,axis=0)
	# print("Mean number of training errors are %s" % train_err_mean)
	val_err = unPickle("errors_val")
	val_err_mean = np.mean(val_err,axis=0)
	# print("Mean number of validation errors are %s" % val_err_mean)
	# Geometric margins across C
	g_margin = unPickle("g_margin")
	g_margin_mean = np.mean(g_margin,axis=0)
	# print("Mean geometric margins are %s" % g_margin_mean)

	# Number of support vectors across C
	n_sv = unPickle("n_sv")
	sv_mean = np.mean(n_sv, axis=0)
	# print(sv_mean)
	sv_std = np.std(n_sv,axis=0, ddof=1)
	# print("Mean number of support vectors are %s" % sv_mean)

	for i in range(len(C)):
		print("Mean test error when C=%s is %s, with standard deviation %s"
			% (C[i], test_mean[i], test_std[i]))
		print("Mean number of support vectors when C=%s is %s, with standard deviation %s"
			% (C[i], sv_mean[i], sv_std[i]))

	# C_vals = ['0.01', '0,1', '1', '10', '100']
	# # Plot number of support vectors vs. C
	# plot_C(sv_mean, sv_std, C_vals, '# support vectors')
	# # Plot test performance vs. C
	# plot_C(test_mean, test_std, C_vals, 'test error rate')
