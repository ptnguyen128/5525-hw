import sys, os
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

from LDA import LDA

from sklearn.datasets import load_digits

if __name__ == '__main__':
    digits = load_digits()
    data = pd.DataFrame(digits.data)
    data['class'] = digits.target
    lda = LDA(data, label='class', n_dims=2)
    w = lda.fit()
    print(w)

# # Split the data into k folds
# def cross_val_split(data, folds, index):
#     data_idx = []
#     indices = [i for i in range(data.shape[0])]
#
#     fold_size = int(len(data)/folds)
#     for i in range(folds):
#         fold_idx = []
#         while len(fold_idx) < fold_size:
#             idx = random.randrange(len(indices))
#             fold_idx.append(indices.pop(idx))
#         data_idx.append(fold_idx)
#
#     test_idx = data_idx[index]
#     del data_idx[index]
#     train_idx = [item for sublist in data_idx for item in sublist]
#
#     test = data.iloc[test_idx]
#     train = data.iloc[train_idx]
#
#     return train,test
#
#
# # Generative Gaussian modeling
# def gaussian_modeling():
#     '''
#     Function to perform generative Gaussian modeling,
#     which estimates priors, means, and covariances for each class
#     '''
#     self.priors = self.g_means = self.g_cov = {}
#
#     for k in X.keys():
#         self.priors[k] = X[k].shape[0] / self.data.shape[0]
#         self.g_means[k] = X[k].mean()
#         self.g_cov[k] = np.cov(X[k],rowvar=False)
