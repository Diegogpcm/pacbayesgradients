This ZIP file includes python code to reproduce results in the paper. Credit for most of the code goes to Pérez-Ortiz et al., as it was shared in their referenced paper.

To reproduce CIFAR-10 results, run the command 'python run_experiment.py'. To reproduce MNIST results, change 'sigma_prior' parameters in 'run_experiment.py' to [0.03, 0.04], 'model' to 'fcn', 'objective' to ['fgrad', 'fquad', 'f_rts', 'fgrad_acc', 'fquad_acc', 'f_rts_acc'] and 'layers' to None. 'run_experiment.py' produces a csv file with hyperparameters and results of each experiment run.

Model architectures can be found in models.py. 4.01 and 5.01 layer values represent smaller (thinner) versions of 4 and 5 layer models. Layer values only affect cnn architecture.

Additionally, most csv files containing the results in the paper are provided, as well as notebooks to process them (e.g., compute mean and 2 sigma).