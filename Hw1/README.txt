-----------------------------
Name: Thao Nguyen
Student ID: 5415745
Email: nguy3409@umn.edu
------------------------------

All .py files are in the src folder. Please change your working directory into this folder before executing any command.

Once you're in this folder, commands can be executed as follows:

- Names of datasets are either `boston` or `digits`. Data should be automatically loaded.
- Commands for question 3:
	$python LDA1dProjection.py boston
	$python LDA2dGaussGM.py digits [n_folds]
- Commands for question 4:
	$python [filename].py [dataset_name] [num_splits]    

	# (i.e. `$python LogisticRegression.py boston 10`)

	$python LogReg_NaiveBayes.py [dataset_name]     # generate comparison plots

- It will take a considerably long time to train logistic regression on the Digits data, so I trained all models and saved the matrix of error rates in the ./pickle/ folder. The `LogReg_NaiveBayes.py` file will load the error rates from these pickle files and plot these error rates.