import sys

import numpy as np 
import pandas as pd

from data import *
from train_test_split import *
from SVM import SVMPegasos

def myPegasos(filename):
	data = load_data(filename)
	X, y = ppData(data)
	Pegasos = SVMPegasos(k=100)
	Pegasos.fit(X,y)
	return Pegasos.w

if __name__ == '__main__':
	filename = './data/MNIST-13.csv'
	w = myPegasos(filename)
	print(w)