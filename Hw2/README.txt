

Requirements for the dataset:
- must be put in the /data/ folder
- in .csv format
- each row is one observation
- first column is target variable (binary), all the other columns are features

General python commands to execute python scripts needed to run SVM algorithms:
  $python ./src/[script_name].py [dataset_name]
For example:
  $python ./src/myDualSVM.py MNIST-13
  $python ./src/myPegasos.py MNIST-13
  $python ./src/mySoftplus.py MNIST-13
