import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import random

class LDA():
    def __init__(self, data, label, n_dims):
        '''
        Class initializer
        -------------------------------------
        Input:
        data: the dataset
        label: name of the target variable
        n_dims: number of dimensions to project into
        -------------------------------------
        '''
        self.data = data
        self.label = label
        self.n_dims = n_dims

    def to_dict(self, df):
        '''
        Function to turn a dataframe into a dictionary
        '''
        # get the groups
        grouped = df.groupby(df.loc[:,self.label])
        self.classes = [k for k in grouped.groups.keys()]

        # mean for each class:
        data_class = {}
        X = {}
        for k in self.classes:
            data_class[k] = grouped.get_group(k)
            X[k] = data_class[k].drop(self.label,axis=1)

        return X

    def get_disc(self):
        '''
        Function to return the discriminant directions w
        in the general case (S_B and S_W computed from the
        original input data)
        '''
        # group the whole data by labels
        X = self.to_dict(self.data)

        # get the mean of each group
        means = {}
        for k in self.classes:
            means[k] = np.array(np.mean(X[k]))

        self.means = means

        # mean of the total data
        mean_total = np.array(np.mean(self.data.drop(self.label,axis=1)))

        # between covariance matrix
        S_B = np.zeros((X[0].shape[1], X[0].shape[1]))
        for k in X.keys():
            S_B += np.multiply(len(X[k]),
                                np.outer((means[k] - mean_total),
                                np.transpose(means[k] - mean_total)))

        # within covariance matrix
        S_W = np.zeros((X[0].shape[1], X[0].shape[1]))
        for k in self.classes:
            S_k = X[k] - means[k]
            S_W += np.dot(S_k.T, S_k)

        # eigendecomposition
        S = np.dot(np.linalg.pinv(S_W), S_B)
        eigval, eigvec = np.linalg.eig(S)

        # sort eigenvalues in decreasing order
        eig = [(eigval[i], eigvec[:,i]) for i in range(len(eigval))]
        eig = sorted(eig, key=lambda x: x[0], reverse=True)

        # only take the top (number of dimensions projected into) vectors
        w = np.array([eig[i][1] for i in range(self.n_dims)])

        return w

    def fit_transform(self, data):
        '''
        Function to project the data into n dimensions
        ----------------------------------------------
        Input: the original data that will be projected
        Output: the data projected into n dimensions
        '''
        # group the input data by labels
        X_data = self.to_dict(data)
        # get the discriminant vectors
        w = self.get_disc()

        # project the input data into n dimensions
        new_data = {}
        for k in X_data.keys():
            new_data[k] = np.dot(X_data[k], w.T)

        return new_data

    def gaussian_model(self, data):
        '''
        Function to perform generative Gaussian modeling and
        calculate class prior, mean, covariance
        '''
        # Project the original data into n dimensions
        X = self.fit_transform(data)

        g_prior = {}
        g_mean = {}
        g_cov = {}

        # for each class
        for k in X.keys():
            # prior
            g_prior[k] = X[k].shape[0] / data.shape[0]
            # mean
            g_mean[k] = np.mean(X[k], axis = 0)
            # covariance
            g_cov[k] = np.cov(X[k],rowvar=False)
        return g_prior, g_mean, g_cov

    def gaussian_errors(self, cv):
        '''
        Function to calculate train and test error rates
        '''

        def class_cond(x, sigma, mu):
            '''
            Function to calculate a Gaussian class-conditional density
            for point x
            '''
            A = 1/((2*np.pi)**(len(x)/2))
            B = 1/(np.linalg.det(sigma)**(1/2))
            C = (-1/2)*(np.dot(np.dot((x-mu).T,np.linalg.inv(sigma)),(x-mu)))
            
            return A*B*np.exp(C)

        def calculate_error(self, data):
            '''
            Function to calculate prediction error rate
            '''
            # Prediction
            y_pred = []
            y_true = data[self.label]
            w = self.get_disc()
            X = data.drop(self.label,axis=1)
            # for each data point
            for idx, x in X.iterrows():
                # project into n dimensions
                x = np.dot(x,w.T)

                likelihood = []
                # calculate the likelihood for each class
                for k in g_prior.keys():
                    p = g_prior[k] * class_cond(x, g_cov[k], g_mean[k])
                    likelihood.append(p)
                # predict the label (one with highest likelihood)
                y_pred.append(np.argmax(likelihood))

            # Calculate error rate
            error = np.sum(y_pred != y_true) / len(y_true)
            return error

        train_errors = []
        test_errors = []
        # Perform cross-validation
        for i in range(cv):
            # Split data into train and test sets
            train, test = self.cross_val_split(self.data, folds=cv, index=i)

            # Train the model on the training set
            g_prior, g_mean, g_cov = self.gaussian_model(train)
            
            # Calculate the train error
            train_errors.append(calculate_error(self,train))
            # Calculate the test error
            test_errors.append(calculate_error(self,test))

        return train_errors, test_errors

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

        
