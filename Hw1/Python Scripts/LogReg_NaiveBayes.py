import sys

import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_digits
from sklearn.datasets import load_boston

from LogisticRegression import logisticRegression
from naiveBayesGaussian import naiveBayesGaussian

def plot_percents(logreg_mean, logreg_std, nb_mean, nb_std, train_percent):
	plt.xlim([5, 105])
	plt.ylim([0.05, 0.6])
	plt.errorbar(train_percent, logreg_mean, yerr=logreg_std*1.96, label='Logistic Regression', fmt="-")
	plt.errorbar(train_percent, nb_mean, yerr=nb_std*1.96, label='Naive Bayes', fmt="-")
	plt.legend()
	plt.ylabel('Test Error Rates')
	plt.xlabel('Training Data Percentage')
	plt.show()

if __name__ == '__main__':
	filename = sys.argv[1]
	num_splits = int(sys.argv[2])
	train_percent = [10, 25, 50, 75, 100]

	log_reg_mean_error, log_reg_std_error = logisticRegression(filename, num_splits, train_percent)
	nb_mean_error, nb_std_error = naiveBayesGaussian(filename, num_splits, train_percent)

	plot_percents(log_reg_mean_error, log_reg_std_error, nb_mean_error, nb_std_error, train_percent)
