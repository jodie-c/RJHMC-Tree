##############################################
	    	RJHMC-Tree
##############################################

RJHMC-Tree is a method for exploring the posterior distribution of Bayesian decision trees by which HMC is incorporated into the proposal structure. Two methods are presented in this paper - HMC-DF and HMC-DFI which use HMC-within-MH and pure HMC respectively.


##############################################
	Requirements/Installation
##############################################

RJHMC-Tree is implemented in Python. Use the following commands to create the required environment to run the method.

- ensure you are in the correct directory
- run ./scripts/setup.sh via terminal/command prompt

Note that to set up the environment with the M1 processor, the jax/jaxlib packages need to be installed via conda-forge from within the environment using:
pip uninstall jax jaxlib
conda install -c conda-forge jaxlib==0.3.14
conda install -c conda-forge jax==0.3.17

Manual changes have been made to the NumPyro package when running setup.sh - these need to be copied across correctly to run the method.

##############################################
		  Usage
##############################################
Once requirements have been installed the RJHMC-Tree methods can be run. Use the following commands:

Activate environment from the correct folder (the rjhmc-env must be active to correctly use RJHMC-Tree) using:

source ./envs/rjhmc-env/bin/activate

Example usage:

python ./src/rjhmc-tree/rjhmc-tree.py --dataset iris --datapath ./datasets/iris/ --out_dir ./results/iris/rjhmc-tree/ --h_init 0.01 --h_final 0.01 --alpha 0.7 --beta 1.0 --N 800 --N_warmup 200 --save 1 --tag 1 --init_id 1  --method 1

python ./src/rjhmc-tree/rjhmc-tree.py --dataset raisin --datapath ./datasets/raisin/ --out_dir ./results/raisin/rjhmc-tree/ --h_init 0.005 --h_final 0.001 --alpha 0.45 --beta 2.0 --N 800 --N_warmup 200 --save 1 --tag 1 --init_id 1 --method 0

For further help and information on arguments:

python ./src/rjhmc-tree/rjhmc-tree.py -h 

All methods were run using train_rjhmc_tree.sh script located in the "scripts" directory using Intel Cascade Lake CPU's via a HPC cluster (OS: Rocky Linux release 8.9 (Green Obsidian)). Uncomment the relevant lines corresponding to the desired method script to run. Some relative paths may need to be altered. 

##############################################
	    Training/Evaluation
##############################################

Running the rjhmc-tree.py file will run both the training and evaluation simultaneously. 

The easiest way to run training/evaluation is via the train_rjhmc_script.sh file in the scripts directory. These commands were used to generate the results presented in the paper. To run the algorithm, uncomment the lines beneath either heading, then uncomment the line with the dataset of interest (or add a new line for a different dataset). The script is then run using the following commands:

- change directory to ./scripts
- uncomment relevant lines in train_rjhmc_script.sh
- run ./train_rjhmc_script.sh 

Tree metrics/information/statistics are saved and output in a pickle file in the specified output directory (changed using the --out_dir argument). 

There are several synthetic testing datasets that can be selected via the --dataset argument, available options are listed in the load_data method in utils.py. More information about formatting datasets into the appropriate structure can be found in ./datasets/README.txt

##############################################
		 Results
##############################################

Results from the paper were generated via CPU cores, either using Intel Cascade Lake CPU's via a HPC cluster or using a MacBook Pro (16-inch, 2021) with an Apple M1 Pro processor. 

Results for the RJHMC-Tree method are included in the results directory. If Matlab is installed, figures and tabular results from the paper can be reproduced by running plot_results.m. Converted results for all methods are included under the matlab subdirectory. Uncomment the dataset of interest on lines 6-10 and run in the plot_results.m file and run to produce results relating to that dataset.
