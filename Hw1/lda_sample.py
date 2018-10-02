import sys, itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.datasets import load_digits

class LDA():
    def __init__(self, data, num_dims = 1, labelcol = -1, split_ratio = 0.8):
        '''
        Class constructor.
        --------------------------------
        data : the entire dataset
        num_dims : number of dimensions to project data into
        convert_data : flag to specify whether the data is to be converted to categorical
        percentile : if convert_data is True, then specify the percentile for conversion
        threshold : flag to indicate whether to do thresholding or gaussian modeling for classification
        labelcol : which column in the csv data contains the label
        split_ratio : split ratio for train-test split
        '''
        self.data = data
        self.num_dims = num_dims
        self.labelcol = labelcol
        self.split_ratio = split_ratio

    '''
    Utility function to drop some column from the given pandas dataframe.
    '''
    def drop_col(self, data, col):
        return data.drop(data.columns[[col]], axis = 1)

    '''
    Main function to apply LDA
    '''
    def fit(self):
        # Function estimates the LDA parameters
        def estimate_params(data):
            # group data by label column
            grouped = data.groupby(self.data.ix[:,self.labelcol])

            # calculate means for each class
            means = {}
            for c in self.classes:
                means[c] = np.array(self.drop_col(self.classwise[c], self.labelcol).mean(axis = 0))

            # calculate the overall mean of all the data
            overall_mean = np.array(self.drop_col(data, self.labelcol).mean(axis = 0))

            # calculate between class covariance matrix
            # S_B = \sigma{N_i (m_i - m) (m_i - m).T}
            S_B = np.zeros((data.shape[1] - 1, data.shape[1] - 1))
            for c in means.keys():
                S_B += np.multiply(len(self.classwise[c]),
                                   np.outer((means[c] - overall_mean), 
                                            (means[c] - overall_mean)))

            # calculate within class covariance matrix
            # S_W = \sigma{S_i}
            # S_i = \sigma{(x - m_i) (x - m_i).T}
            S_W = np.zeros(S_B.shape) 
            for c in self.classes: 
                tmp = np.subtract(self.drop_col(self.classwise[c], self.labelcol).T, np.expand_dims(means[c], axis=1))
                S_W = np.add(np.dot(tmp, tmp.T), S_W)

            # objective : find eigenvalue, eigenvector pairs for inv(S_W).S_B
            mat = np.dot(np.linalg.pinv(S_W), S_B)
            eigvals, eigvecs = np.linalg.eig(mat)
            eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

            # sort the eigvals in decreasing order
            eiglist = sorted(eiglist, key = lambda x : x[0], reverse = True)

            # take the first num_dims eigvectors
            w = np.array([eiglist[i][1] for i in range(self.num_dims)])

            self.w = w
            self.means = means
            return

        traindata, testdata = self.cross_val_split(self.data,folds=10,index=0)

        # group data by label column
        grouped = data.groupby(self.data.ix[:,self.labelcol])
        self.classes = [c for c in grouped.groups.keys()]
        self.classwise = {} 
        for c in self.classes:
            self.classwise[c] = grouped.get_group(c)

        # estimate the LDA parameters
        estimate_params(data)
        
        self.gaussian_modeling()
        # append the training and test error rates for this iteration
        trainerror = self.calculate_score_gaussian(traindata) / float(traindata.shape[0])
        testerror = self.calculate_score_gaussian(testdata) / float(testdata.shape[0])

        return trainerror, testerror

    '''
    Function to estimate gaussian models for each class.
    Estimates priors, means and covariances for each class.
    '''
    def gaussian_modeling(self):
        self.priors = {}
        self.gaussian_means = {}
        self.gaussian_cov = {}

        for c in self.means.keys():
            inputs = self.drop_col(self.classwise[c], self.labelcol)
            proj = np.dot(self.w, inputs.T).T
            self.priors[c] = inputs.shape[0] / float(self.data.shape[0])
            self.gaussian_means[c] = np.mean(proj, axis = 0)
            self.gaussian_cov[c] = np.cov(proj, rowvar=False)

    '''
    Utility function to return the probability density for a gaussian, given an 
    input point, gaussian mean and covariance.
    '''
    def pdf(self, point, mean, cov):
        cons = 1./((2*np.pi)**(len(point)/2.)*np.linalg.det(cov)**(-0.5))
        return cons*np.exp(-np.dot(np.dot((point-mean),np.linalg.inv(cov)),(point-mean).T)/2.)

    '''
    Function to calculate error rates based on gaussian modeling.
    '''
    def calculate_score_gaussian(self, data):
        classes = sorted(list(self.means.keys()))
        inputs = data.drop('class', axis=1)
        # project the inputs
        proj = np.dot(self.w, inputs.T).T
        # calculate the likelihoods for each class based on the gaussian models
        likelihoods = np.array([[self.priors[c] * self.pdf([x[ind] for ind in 
                                                            range(len(x))], self.gaussian_means[c], 
                               self.gaussian_cov[c]) for c in 
                        classes] for x in proj])
        # assign prediction labels based on the highest probability
        labels = np.argmax(likelihoods, axis = 1)
        errors = np.sum(labels != data.ix[:, self.labelcol])
        return errors

    def cross_val_split(self, data, folds, index):
        '''
        Function to split the data into train-test sets for k-fold cross-validation
        '''
        data_idx = []
        indices = [i for i in range(data.shape[0])]

        fold_size = int(len(data)/folds)
        for i in range(folds):
            fold_idx = []
            while len(fold_idx) < fold_size:
                idx = random.randrange(len(indices))
                fold_idx.append(indices.pop(idx))
            data_idx.append(fold_idx)

        test_idx = data_idx[index]
        del data_idx[index]
        train_idx = [item for sublist in data_idx for item in sublist]

        test = data.iloc[test_idx]
        train = data.iloc[train_idx]

        return train,test
    
if __name__ == '__main__':
    digits = load_digits()
    data = pd.DataFrame(digits.data)
    label = 'class'
    data[label] = digits.target

    labelcol = -1
    lda = LDA(data, num_dims=2, labelcol=labelcol)
    trainerror, testerror = lda.fit()
    print(lda.gaussian_means)
    #print(trainerror)
    #print(testerror)
    #print(verifyLDA(data, data, labelcol))
    #lda.plot_proj_2D(data)
    #lda.plot_bivariate_gaussians()
