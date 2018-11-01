import sys

import numpy as np
import pandas as pd

from data import *
from SVM import SVMSoftplus

def mySoftplus(filename):
	data = load_data(filename)
	ks = [1, 20, 200, 1000, 2000]

	for k in ks:
		print("Training for k = %s" % k)
		Softplus = SVMSoftplus(k=k, lambd=100, n_runs=5)

		run_times = Softplus.fit(data)
		avg_rt = np.mean(run_times, axis=0)
		std_rt = np.std(run_times, axis=0, ddof=1)

		print("For k = %s" % k)
		print("Mean run time is %0.3f, with standard deviation %0.3f" % (avg_rt, std_rt))

if __name__ == '__main__':
	file = sys.argv[1]
	filename = './data/%s.csv' % file
	mySoftplus(filename)
