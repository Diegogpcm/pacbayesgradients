import torch
from pbb.utils import runexp
from itertools import product
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

parameters = {
    'seed' : list(range(10)),
    'name_data' : ['cifar10'], # 'mnist', 'cifar10'
    'objective' : ['fgrad_acc', 'fquad_acc', 'f_rts_acc'], #'fgrad', 'fquad', 'f_rts', 'fgrad_acc', 'fquad_acc', 'f_rts_acc'
    'prior_type' : ['rand',],
    'model' : ['cnn'],  #'fcn', 'cnn'
    'sigma_prior' : [0.05, 0.06, 0.07], 
    'pmin' : [1e-5],
    'learning_rate' : [5e-3],
    'momentum' : [0.95],
    'learning_rate_prior' : [0.005],
    'momentum_prior' : [0.99],
    'delta' : [0.025],
    'layers' : [None],  # For cnn only, 4, 4.01, 5, 5.01, 9, 13, 15 - FCN has fixed architecture
    'delta_test' : [0.01],
    'mc_samples' : [5000],  #150.000 for main paper results
    'kl_penalty' : [1],
    'train_epochs' : [100],
    'verbose': [False],
    'device': [DEVICE],
    'prior_epochs': [70],
    'dropout_prob': [0.2],
    'perc_train': [1.0],
    'verbose_test':[False],
    'perc_prior':[1e-10],  # Unused as prior is rand in our paper
    'batch_size' : [250],
    
}

def grid_parameters(parameters):
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))
        
experiment_params = list(grid_parameters(parameters))
experiment_results = [runexp(**params) for params in experiment_params]

exp_output = [{**i, **j} for i, j in zip(experiment_params, experiment_results)]
    
df = pd.DataFrame(exp_output)
df.to_csv('experiment_output.csv')
    







