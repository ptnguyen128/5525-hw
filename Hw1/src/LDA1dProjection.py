import sys, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

from LDA import LDA

from sklearn.datasets import load_boston

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

def plot_hist(data, n_bins):
    '''
    Function to plot the histograms of the classes (overlapping)
    '''
    plt.hist(data[0], n_bins, color='blue', alpha=0.7)
    plt.hist(data[1], n_bins, color='red', alpha=0.5)
    plt.show()

if __name__ == '__main__':
    filename = sys.argv[1]
    if filename == 'boston':
        # Load in the data
        boston = load_boston()
        data = pd.DataFrame(boston.data, columns=boston.feature_names)
        label = 'HomeVal50'
        data[label] = boston.target
        # Transform the target variable to binary
        to_label(data, label, 50)

        # Project data into R (one-dimensional)
        lda = LDA(data, label=label, n_dims=1)
        X_fit = lda.fit_transform(data)

        # Plot the histograms of the transformed data
        plot_hist(X_fit, 20)
    else:
        print("Please enter a valid file name.")
