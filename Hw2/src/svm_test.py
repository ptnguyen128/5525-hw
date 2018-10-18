import pandas as pd
import numpy as np
from plotBoundary import *
import pandas as pd
import numpy as np
from plotBoundary import *
# import your SVM training code
from SVM import softMarginSVM

# parameters
name = 'iris'
print ('======Training======')
# load data from csv files
train = pd.read_csv('data/data_'+name+'_train.csv', header=None)
# use deep copy here to make cvxopt happy
X = np.array(train.iloc[:,:2])
Y = train.iloc[:,-1]
y = [float(num) for num in train.iloc[:,-1]]
y = np.reshape(y, [X.shape[0],1])

# Carry out training, primal and/or dual
SVM = softMarginSVM(C=0.1)
SVM.fit(X,y)

# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
    x = x.reshape(-1,1)
    return np.sign(np.dot(SVM.w.T, x)+SVM.b)

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


# print ('======Validation======')
# # load data from csv files
# validate = pd.read_csv('data/data_'+name+'_validate.csv', header=None)
# X = np.array(validate.iloc[:,:-1])
# Y = np.array([float(num) for num in validate.iloc[:,-1]])
# Y = np.reshape(Y, [X.shape[0],1])
# # plot validation results
# plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

