import sys, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

from LDA import LDA
from data import load_data

from sklearn.datasets import load_boston

def plot_hist(data, n_bins):
    '''
    Function to plot the histograms of the classes (overlapping)
    '''
    plt.hist(data[0], n_bins, color='blue', alpha=0.7)
    plt.hist(data[1], n_bins, color='red', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    filename = sys.argv[1]
    
    # Load in the data
        data, label = load_data(filename)
        
    if filename == 'boston':
        # Project data into R (one-dimensional)
        lda = LDA(data, label=label, n_dims=1)
        X_fit = lda.fit_transform(data)

        # Plot the histograms of the transformed data
        plot_hist(X_fit, 20)
    else:
        print("Please enter a valid file name.")
