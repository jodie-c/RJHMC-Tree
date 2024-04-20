Datasets need to be saved in a .pickle file to be imported correctly to RJHMC-Tree algorithm.

Dataset format - python dictionary with following keys:

x_train: 	array containing training input values of shape (n_train,nx)
y_train: 	array containing training output values of shape (n_train,)
x_test:		array containing testing input values of shape (n_test,nx)
y_test:		array containing testing output values pf shape (n_test,)
nx: 		number of inputs (dimension) 
ny: 		number of output classes (classification only)
n_train: 	number of training datapoints/observations
n_test:		number of testing datapoints/observations 
output_type: 	specifies the output type (either class/real for classification/regression)

To run comparison to other methods, the following keys are also required:
n_dim:	 	number of inputs (dimension) -- required for SMC
is_sparse:  	should always be set to False -- required for SMC

Note: SMC/MCMC method requires .p extension, Wu requires .txt extension (and separated into train/test). See individual readme files in src for further details.

################################################################################

######### Synthetic datasets:
- cgm 				-- from (Chipman, et al. 1998) (note categorical input has been changed to real)
				-- creates tree with true structure: children_left = [1 3 5 -1 7 -1 -1 -1 -1], children_right = [2 4 6 -1 8 -1 -1 -1 -1], index = [1 0 0 0], tau = [4 3 7 5], mu = [1 5 8 8 2]

######### Real-world datasets:
- iris 				-- https://archive.ics.uci.edu/ml/datasets/Iris
- breast-cancer-wisconsin 	-- https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Original%29
- raisin			-- http://archive.ics.uci.edu/ml/datasets/Raisin+Dataset
- wine 				-- http://archive.ics.uci.edu/ml/datasets/Wine

