import sys

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random
import pickle

def plot_percents(logreg_mean, logreg_std, nb_mean, nb_std, train_percent):
	plt.xlim([8, 102])
	plt.ylim([0.05, 0.6])
	plt.figure()
	plt.errorbar(train_percent, logreg_mean, yerr=logreg_std*1.96, label='Logistic Regression', fmt="-")
	plt.errorbar(train_percent, nb_mean, yerr=nb_std*1.96, label='Naive Bayes', fmt="-")
	plt.legend()
	plt.ylabel('Test Error Rates')
	plt.xlabel('Training Data Percentage')
	plt.show()

if __name__ == '__main__':
	filename = sys.argv[1]
	train_percent = [10, 25, 50, 75, 100]

	# Logistic Regression
	pickle_off = open("./pickle/%s_logreg_error_matrix.pickle" % filename, 'rb')
	logreg_error_matrix = pickle.load(pickle_off)

	print(logreg_error_matrix)

	log_reg_mean_error = np.mean(logreg_error_matrix, axis=0)
	log_reg_std_error = np.std(logreg_error_matrix, axis=0, ddof=1)

	print("Logistic Regression: Mean test errors are %s , with standard errors % s" % 
		(log_reg_mean_error, log_reg_std_error))

	# Naive Bayes
	pickle_off = open("./pickle/%s_nb_error_matrix.pickle" % filename, 'rb')
	nb_error_matrix = pickle.load(pickle_off)

	print(nb_error_matrix)

	nb_mean_error = np.mean(nb_error_matrix, axis=0)
	nb_std_error = np.std(nb_error_matrix, axis=0, ddof=1)

	print("Naive Bayes: Mean test errors are %s , with standard errors % s" % 
		(nb_mean_error, nb_std_error))

	plot_percents(log_reg_mean_error, log_reg_std_error, nb_mean_error, nb_std_error, train_percent)
