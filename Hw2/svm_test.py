from numpy import *
from plotBoundary import *
# import your SVM training code
from SVM import softMarginSVM

# parameters
name = 'iris'
print ('======Training======')
# load data from csv files
train = loadtxt('data/data_'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Carry out training, primal and/or dual
SVM = softMarginSVM(C=0.1)
SVM.fit(X,Y)

# Define the predictSVM(x) function, which uses trained parameters
def predictSVM(x):
	return np.sign(np.dot(SVM.w.T, x))

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Train')


print ('======Validation======')
# load data from csv files
validate = loadtxt('data/data_'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')

