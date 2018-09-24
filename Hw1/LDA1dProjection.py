import sys, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

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

if __name__ == '__main__':
    boston = load_boston()
    data = pd.DataFrame(boston.data, columns=boston.feature_names)
    data['HomeVal50'] = boston.target
    to_label(data, 'HomeVal50', 50)
    print(data.head(15))
