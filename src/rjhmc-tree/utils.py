import random as pyrandom
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import jax.numpy as jnp
import copy
from numpyro.infer.util import log_likelihood
from class_defs import BayesianTree
from scipy.special import expit
# import calc_metrics

################## Command-line related function ################## 
def parser_general_options(parser): 
    general_options = parser.add_argument_group("General Options")
    general_options.add_argument('--dataset', dest='dataset', default='toy-real',
        help='dataset to be used (default: %(default)s)')
    general_options.add_argument('--save', dest='save', default=0, type=int,
        help='do you wish to save the results? (1=Yes/0=No)') 
    general_options.add_argument('--plot_info', dest='plot_info', default=0, type=int,
        help='do you wish to plot transition and metric information? (1=Yes/0=No)') 
    general_options.add_argument('--tag', dest='tag', default='', 
            help='additional tag to identify results from a particular run')
    general_options.add_argument('--init_id', dest='init_id', default=1, type=int,
            help='seed value to change initialisation for multiple runs')
    general_options.add_argument('--datapath', dest='datapath', default='',
            help='path to the dataset')
    general_options.add_argument('--out_dir', dest='out_dir', default='.', 
            help='output directory for pickle files (NOTE: make sure directory exists) (default: %(default)s)')
    return parser

def parser_mcmc_options(parser): 
    mcmc_options = parser.add_argument_group("RJHMC-Tree Options")
    mcmc_options.add_argument('--N', dest='N', default=1000, type=int,
        help='number of sample iterations of RJHMC-Tree chain (default: %(default)s)')
    mcmc_options.add_argument('--N_warmup', dest='N_warmup', default=50, type=int,
        help='number of warm-up iterations of RJHMC-Tree chain (default: %(default)s)')
    mcmc_options.add_argument('--children_left', nargs='+', dest='children_left', default=[1,-1,-1], type=int,
            help='node id of children to left of current node (-1 means leaf/terminal node) (default: %(default)s, single split)')
    mcmc_options.add_argument('--children_right', nargs='+', dest='children_right', default=[2,-1,-1], type=int,
            help='node id of children to right of current node (-1 means leaf/terminal node) (default: %(default)s, single split)')
    mcmc_options.add_argument('--warmup_probs', nargs='+', dest='warmup_probs', default=[0.5,0.25,0.25], type=float,
            help='probability of moves during warmup phase in order [grow,prune,stay] (default: %(default)s)')
    mcmc_options.add_argument('--sample_probs', nargs='+', dest='sample_probs', default=[0.2,0.2,0.6], type=float,
            help='probability of moves during sampling phase in order [grow,prune,stay] (default: %(default)s)')
    return parser

def parser_outer_options(parser): # options relevant to the outer loop (topology moves)
    outer_options = parser.add_argument_group("Outer Loop Options")
    outer_options.add_argument('--alpha', dest='alpha', default=0.95, type=float,
        help='alpha parameter for prior on tree structure (default: %(default)s)')   
    outer_options.add_argument('--beta', dest='beta', default=1, type=float,
        help='beta parameter for prior on tree structure (default: %(default)s)')
    return parser

def parser_inner_options(parser): # options relevant to the inner loop (HMC; tau/index/mu/sigma moves)
    inner_options = parser.add_argument_group("Inner Loop Options")
    inner_options.add_argument('--method', dest='method', default=1, type=int,
        help='which method to run - 0 == HMC-DF, 1 == HMC-DFI (default: %(default)s)')
    inner_options.add_argument('--h_init', dest='h_init', default=0.1, type=float,
        help='initial value for gating function parameter for soft splits (default: %(default)s)')
    inner_options.add_argument('--h_final', dest='h_final', default=0.005, type=float,
        help='final value for gating function parameter for soft splits (default: %(default)s)')
    inner_options.add_argument('--alpha_llh', dest='alpha_llh', default=[1], type=float,
        help='alpha concentration parameter for Dir-Multi (classification) likelihood (default: %(default)s)')
    inner_options.add_argument('--scale_llh', dest='scale_llh', default=3/2, type=float,
        help='inverse-gamma scale parameter for regression likelihood (default: %(default)s)')
    inner_options.add_argument('--mu_mean_llh', dest='mu_mean_llh', default=0.0, type=float,
        help='normal distribution mean parameter for regression likelihood (default: %(default)s)')
    inner_options.add_argument('--mu_var_llh', dest='mu_var_llh', default=1.0, type=float,
        help='normal distribution variance parameter for regression likelihood (default: %(default)s)')
    inner_options.add_argument('--hmc_num_warmup', dest='hmc_num_warmup', default=500, type=int,
        help='number of warm-up samples in each run of inner HMC loop (default: %(default)s)')
    return parser

def process_command_line():
    parser = argparse.ArgumentParser()
    parser = parser_general_options(parser)
    parser = parser_mcmc_options(parser)
    parser = parser_outer_options(parser)
    parser = parser_inner_options(parser)
    args = parser.parse_args()
    return args

################## Dataset related functions ##################
def normalise_data(data):
    data['transform'] = {'min':np.amin(data['x_train'],axis=0), 'range':np.amax(data['x_train'],axis=0) - np.amin(data['x_train'],axis=0)}
    # data['transform'] = {'min':np.amin(data['x_train']), 'range':np.amax(data['x_train']) - np.amin(data['x_train'])}
    data['x_train'] = (data['x_train'] - data['transform']['min']) / data['transform']['range'] 
    data['x_test'] = (data['x_test'] - data['transform']['min']) / data['transform']['range'] 
    return data

def convert_data(data): 
    data['x_train'] = jnp.asarray(data['x_train'])
    data['y_train'] = jnp.asarray(data['y_train'])
    if(data['n_test'] > 0):
        data['x_test'] = jnp.asarray(data['x_test'])
        data['y_test'] = jnp.asarray(data['y_test'])
    if(np.shape(data['x_train'])[0] != data['n_train']):
        data['x_train'] = np.transpose(data['x_train'])
        assert (np.shape(data['x_train'])[0]==data['n_train']) and (np.shape(data['x_train'])[1]==data['nx'])
    return data

def process_dataset(data):
    data = normalise_data(data)
    data = convert_data(data)
    return data

def load_data(settings):
    data = {}
    if settings.dataset == 'toy-class':
        data = load_toy_class_data()
    elif settings.dataset == 'toy-class-noise':
        data = load_toy_class_noise_data()
    elif settings.dataset == 'toy-class-1d': ## TODO Check
        data = load_toy_class_1d_data()
    elif settings.dataset == 'toy-real-1d': ## TODO Check
        data = load_toy_real_1d()
    elif settings.dataset == 'test-1': ## TODO CHECK IF THIS IS DIFFERENT???
        data = load_test_dataset_1()
    else:
        try:
            dt = pickle.load(open(settings.datapath + settings.dataset + '.pickle', "rb"))
        except:
            raise Exception('Unknown dataset: ' + settings.datapath + settings.dataset)
        data = import_external_dataset(dt)
    return data 

def import_external_dataset(dt):
    if('output_type' not in dt):
        raise Exception('Output type is not specified in dataset (must be either class/real for classification/regression).')   
    if(np.shape(dt['x_train'])[0] != dt['n_train']):
        dt['x_train'] = np.transpose(dt['x_train'])
        assert (np.shape(dt['x_train'])[0]==dt['n_train']) and (np.shape(dt['x_train'])[1]==dt['nx'])
    if('x_test' not in dt): # handles case when importing just training data
        dt['x_test'] = []
        dt['y_test'] = []
        dt['n_test'] = 0
        if(np.shape(dt['x_test'])[0]!=dt['n_test']):
            dt['x_test'] = np.transpose(dt['x_test'])
            assert (np.shape(dt['x_test'])[0]==dt['n_test']) and (np.shape(dt['x_test'])[1]==dt['nx'])
    return dt

def load_toy_non_sym():
    """ Toy dataset which is not symmetric. """
    tau1 = 0.5
    tau2 = 0.7 # x(0) > 0.5
    tau3 = 0.3 # x(0) <= 0.5
    indx1 = 0
    indx2 = 1 
    indx3 = 1
    nx = 2 
    ny = 2
    n_train = 1000
    n_test = n_train

    x_train = np.empty((n_train,nx))
    x_train[:500,indx1] = np.random.uniform(0.1,tau1-0.05,500)
    x_train[500:,indx1] = np.random.uniform(tau1+0.05,0.9,500)
    x_train[:250,indx3] = np.random.uniform(0.1,tau3-0.05,250)
    x_train[250:500,indx3] = np.random.uniform(tau3+0.05,0.9,250)
    x_train[500:750,indx2] = np.random.uniform(0.1,tau2-0.05,250)
    x_train[750:,indx2] = np.random.uniform(tau2+0.05,0.9,250)

    x_test = np.empty((n_test,nx))
    x_test[:500,indx1] = np.random.uniform(0.1,tau1-0.05,500)
    x_test[500:,indx1] = np.random.uniform(tau1+0.05,0.9,500)
    x_test[:250,indx3] = np.random.uniform(0.1,tau3-0.05,250)
    x_test[250:500,indx3] = np.random.uniform(tau3+0.05,0.9,250)
    x_test[500:750,indx2] = np.random.uniform(0.1,tau2-0.05,250)
    x_test[750:,indx2] = np.random.uniform(tau2+0.05,0.9,250)

    y_train = np.empty(n_train)
    y_train[(x_train[:,indx1] <= tau1) & (x_train[:,indx3] <= tau3)] =  1 + np.random.normal(0,0.1,250)
    y_train[(x_train[:,indx1] <= tau1) & (x_train[:,indx3] > tau3)] =  0 + np.random.normal(0,0.1,250)
    y_train[(x_train[:,indx1] > tau1) & (x_train[:,indx2] <= tau2)] =  0 + np.random.normal(0,0.1,250)
    y_train[(x_train[:,indx1] > tau1) & (x_train[:,indx2] > tau2)] =  1 + np.random.normal(0,0.1,250)

    y_test = np.empty(n_test)
    y_test[(x_test[:,indx1] <= tau1) & (x_test[:,indx3] <= tau3)] =  1 + np.random.normal(0,0.1,250)
    y_test[(x_test[:,indx1] <= tau1) & (x_test[:,indx3] > tau3)] =  0 + np.random.normal(0,0.1,250)
    y_test[(x_test[:,indx1] > tau1) & (x_test[:,indx2] <= tau2)] =  0 + np.random.normal(0,0.1,250)
    y_test[(x_test[:,indx1] > tau1) & (x_test[:,indx2] > tau2)] =  1 + np.random.normal(0,0.1,250)

    data = {'x_train': x_train, 'y_train': y_train, 'ny': ny, \
            'nx': nx, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'output_type':'real'}
    return data

def load_toy_class_data():
    p = 3 # number of predictors
    n = 100 # number of training observations
    n_test = 100 # number of testing observations
    x_train = np.zeros([n,p])
    y_train = np.zeros(n)
    x_test = np.zeros([n_test,p])
    y_test = np.zeros(n_test)
    eps = 0.0 # deadzone region
    tau_true = 0.5

    x_train[:,0] = np.random.uniform(0,1,n) # x0 is random, does not effect output
    x_train[:,2] = np.random.uniform(0,1,n) # x2 is random, does not effect output
    indx = np.random.uniform(size=n) < tau_true
    x_train[indx,1] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    x_train[~indx,1] = np.random.uniform(tau_true+eps,1,size=n-sum(indx))
    indx = x_train[:,1] <= tau_true
    y_train[indx] = 0
    y_train[~indx] = 1

    x_test[:,0] = np.random.uniform(0,1,n_test) # x0 is random, does not effect output
    x_test[:,2] = np.random.uniform(0,1,n_test) # x2 is random, does not effect output
    indx = np.random.uniform(size=n_test) < tau_true
    x_test[indx,1] = np.random.uniform(0,tau_true-eps,size=sum(indx))
    x_test[~indx,1] = np.random.uniform(tau_true+eps,1,size=n_test-sum(indx))
    indx = x_test[:,1] <= tau_true
    y_test[indx] = 0
    y_test[~indx] = 1

    data = {'x_train': x_train, 'y_train': y_train, 'ny': 2, \
            'nx': p, 'n_train': n, 'x_test': x_test, 'y_test': y_test, \
            'n_test': n_test, 'output_type':'class'}
    return data

def load_toy_class_noise_data():
    data = load_toy_class_data()
    # Create noisy dataset - randomly swap values of some indicies
    indx = pyrandom.sample(range(0,data['n_train']), int(np.floor(data['n_train']/10)))
    data['y_train']= np.array(data['y_train'])
    data['y_train'][indx] = 1 - data['y_train'][indx]
    data['y_train'] = data['y_train']
    indx = pyrandom.sample(range(0,data['n_test']), int(np.floor(data['n_test']/10)))
    data['y_test']= np.array(data['y_test'])
    data['y_test'][indx] = 1 - data['y_test'][indx]
    data['y_test'] = data['y_test']
    return data

def load_toy_class_1d_data():
    """ Create basic 1d dataset - Gaussian distributed x values, y in {0,1}. """
    p = 1 # number of predictors
    n_train = 100 # number of observations
    n_test = n_train
    x_train = np.zeros([n_train,p])
    y_train = np.zeros(n_train)
    x_test = np.zeros([n_test,p])
    y_test = np.zeros(n_test)
    tau_true = 0.5

    indx = np.random.uniform(size=n_train) < tau_true
    x_train[indx] = np.random.normal(0.25,0.08,size=(sum(indx),1))
    y_train[indx] = 0
    x_train[~indx] = np.random.normal(0.75,0.08,size=(n_train-sum(indx),1))
    y_train[~indx] = 1

    indx = np.random.uniform(size=n_test) < tau_true
    x_test[indx] = np.random.normal(0.25,0.08,size=(sum(indx),1))
    y_test[indx] = 0
    x_test[~indx] = np.random.normal(0.75,0.08,size=(n_test-sum(indx),1))
    y_test[~indx] = 1
    data = {'x_train': x_train, 'y_train': y_train, 'ny': 2, \
        'nx': p, 'n_train': n_train, 'x_test': x_test, 'y_test': y_test, \
        'n_test': n_test, 'output_type':'class'}
    return data

def load_toy_real_1d():
    data = load_toy_class_1d_data()
    data['y_train'] = data['y_train'] + np.random.normal(0,0.1,np.shape(data['y_train'])) # convert to regression
    data['y_test'] = data['y_test'] + np.random.normal(0,0.1,np.shape(data['y_test'])) # convert to regression
    data['output_type'] = 'real'
    return data

def load_test_dataset_1():
    nx = 2 
    n_train_pc = 100
    ny = 2
    n_train = n_train_pc * ny
    n_test = n_train
    y_train = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    y_test = np.r_[np.ones(n_train_pc, dtype='int'), \
            np.zeros(n_train_pc, dtype='int')]
    x_train = np.random.randn(n_train, nx)
    x_test = np.random.randn(n_train, nx)
    mag = 5
    for i, y_ in enumerate(y_train):
        if y_ == 0:
            x_train[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_train[i, :] += np.array([tmp, -tmp]) * mag
    for i, y_ in enumerate(y_test):
        if y_ == 0:
            x_test[i, :] += np.sign(np.random.rand() - 0.5) * mag
        else:
            tmp = np.sign(np.random.rand() - 0.5) 
            x_test[i, :] += np.array([tmp, -tmp]) * mag
    data = {'x_train': x_train, 'y_train': y_train, 'ny': ny, \
            'nx': nx, 'n_train': n_train, 'x_test': x_test, \
            'y_test': y_test, 'n_test': n_test, 'output_type':'class'}
    return data

def plot_dataset(data, ax=None, plot_now=False):
    if(data['nx'] == 1): # one predictor variable 
        if(ax is None):
            plot_now = True
            plt.figure(figsize=(15,10))  
        plt.plot(data['x_train'],data['y_train'],'*')
        plt.xlabel("x")
        plt.ylabel("y")
    elif(data['nx'] == 2): # two predictor variables
        if(ax is None):
            plot_now = True
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data['x_train'][:,0],data['x_train'][:,1],data['y_train'])
        ax.set_xlabel('x0')
        ax.set_ylabel('x1')
        ax.set_zlabel('y')
    else: # otherwise, visualise separately
        if(ax is None):
            plot_now = True
            fig, ax = plt.subplots(1,data['nx'],figsize=(15,10)) 
        for i in range(data['nx']):
            ax[i].plot(np.array(data['x_train'])[:,i],data['y_train'],'*')
            ax[i].set_xlabel("x"+str(i))
            ax[i].set_ylabel("y")

    if(plot_now == True):
        plt.show()

################## Tree Related Functions ##################
def get_child_id(node_id,x,split,index,internal_map): # needed to generate large dataset quickly
    tmp = 2 * (node_id + 1)
    if(x[index[internal_map[node_id]]] <= split[internal_map[node_id]]):
        return tmp - 1
    else:
        return tmp

def traverse(x,index,split,leaf_nodes,internal_map):  # needed to generate large dataset quickly
    node_id = 0
    while (node_id not in leaf_nodes):
        node_id = get_child_id(node_id,x,split,index,internal_map)
    return node_id

def traverse_all_data(x,index,tau,internal_nodes,leaf_nodes): # needed to generate large dataset quickly
    node_ids = np.zeros(len(x),dtype=int)
    leaf_map = np.full(max(leaf_nodes)+1,100)
    leaf_map[leaf_nodes] = np.arange(len(leaf_nodes))
    internal_map = np.full(max(internal_nodes)+1,100)
    internal_map[internal_nodes] = np.arange(len(internal_nodes))
    for i in range(len(x)):
        node_ids[i] = leaf_map[traverse(x[i],index,tau,leaf_nodes,internal_map)]
    return node_ids

def evaluate_tree_df(tree,data,h):
    if(data['n_test'] == 0):
        raise AssertionError('No testing dataset to evaluate tree.')
    x_train = data['x_train']# * data['transform']['range'] + data['transform']['min'] # undo data transformation
    if(data['output_type']=='class'):
        test_indicies = np.full([data['ny'],data['n_test']],-1)
        for i in range(data['ny']):
            test_indicies[i,:] = (data['y_test'] == tree.y_classes[i])
        for i,node in enumerate(tree.internal_nodes): 
            if(data['nx']==1): # only one input variable
                node.psi_train = expit((x_train-node.params['tau'])/h)
                node.psi_test = expit((data['x_test']-node.params['tau'])/h)
            else:
                node.psi_train = expit((x_train[:,node.params['index']]-node.params['tau'])/h)
                node.psi_test = expit((data['x_test'][:,node.params['index']]-node.params['tau'])/h)
        leaf_probs_train = np.zeros((tree.nl,data['n_train'])) 
        leaf_probs_test = np.zeros((tree.nl,data['n_test'])) 
        theta = np.full(tree.nl,-1) # holds which class output is assigned to the leaf node w.r.t calculated frequency
        for ii, node in enumerate(tree.leaf_nodes):
            phi_train = 1
            phi_test = 1
            for i,eta in enumerate(node.ancestors): 
                phi_train = phi_train * (eta.psi_train ** (node.path[i])) * ((1-eta.psi_train) ** (1-node.path[i]))
                phi_test = phi_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
            leaf_probs_train[ii,:] = phi_train
            leaf_probs_test[ii,:] = phi_test
            theta[ii] = np.argmax(np.sum(phi_train*tree.indicies,axis=1))
        leaf_ids_train = np.argmax(leaf_probs_train,axis=0)
        leaf_ids_test = np.argmax(leaf_probs_test,axis=0)
        miss_train = np.sum(data['y_train'] != theta[leaf_ids_train])/data['n_train']
        miss_test = np.sum(data['y_test'] != theta[leaf_ids_test])/data['n_test']
        for node in tree.internal_nodes: # clean up
            delattr(node, 'psi_train')
            delattr(node, 'psi_test')
        return {'miss_train':miss_train,'miss_test':miss_test}
    else:
        for i,node in enumerate(tree.internal_nodes): 
            if(data['nx']==1): # only one input variable
                node.psi_train = expit((x_train-node.params['tau'])/h)
                node.psi_test = expit((data['x_test']-node.params['tau'])/h)
            else:
                node.psi_train = expit((x_train[:,node.params['index']]-node.params['tau'])/h)
                node.psi_test = expit((data['x_test'][:,node.params['index']]-node.params['tau'])/h)
        leaf_probs_train = np.zeros((tree.nl,data['n_train'])) 
        leaf_probs_test = np.zeros((tree.nl,data['n_test'])) 
        mu = np.zeros(tree.nl)
        for ii, node in enumerate(tree.leaf_nodes):
            mu[ii] = node.params['mu']
            phi_train = 1
            phi_test = 1
            for i,eta in enumerate(node.ancestors): 
                phi_train = phi_train * (eta.psi_train ** (node.path[i])) * ((1-eta.psi_train) ** (1-node.path[i]))
                phi_test = phi_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
            leaf_probs_train[ii,:] = phi_train
            leaf_probs_test[ii,:] = phi_test
        leaf_ids_train = np.argmax(leaf_probs_train,axis=0)
        leaf_ids_test = np.argmax(leaf_probs_test,axis=0)
        mse_train = np.sum(np.power(mu[leaf_ids_train]-data['y_train'],2))/data['n_train']
        mse_test = np.sum(np.power(mu[leaf_ids_test]-data['y_test'],2))/data['n_test']
        for node in tree.internal_nodes: # clean up
            delattr(node, 'psi_train')
            delattr(node, 'psi_test')
        return {'mse_train':mse_train,'mse_test':mse_test}


def evaluate_tree_dfi(tree,data,h):
    if(data['n_test'] == 0):
        raise AssertionError('No testing dataset to evaluate tree.')
    x_train = data['x_train']# * data['transform']['range'] + data['transform']['min'] # undo data transformation
    if(data['output_type']=='class'):
        test_indicies = np.full([data['ny'],data['n_test']],-1)
        for i in range(data['ny']):
            test_indicies[i,:] = (data['y_test'] == tree.y_classes[i])
        for i,node in enumerate(tree.internal_nodes): 
            if(data['nx']==1): # only one input variable
                node.psi_train = expit((x_train[:,0]-node.params['tau'])/h)
                node.psi_test = expit((data['x_test'][:,0]-node.params['tau'])/h)
            else:
                node.psi_train = expit((np.dot(x_train,node.params['index'])-node.params['tau'])/h)
                node.psi_test = expit((np.dot(data['x_test'],node.params['index'])-node.params['tau'])/h)
        leaf_probs_train = np.zeros((tree.nl,data['n_train'])) 
        leaf_probs_test = np.zeros((tree.nl,data['n_test'])) 
        theta = np.full(tree.nl,-1) # holds which class output is assigned to the leaf node w.r.t calculated frequency
        for ii, node in enumerate(tree.leaf_nodes):
            phi_train = 1
            phi_test = 1
            for i,eta in enumerate(node.ancestors): 
                phi_train = phi_train * (eta.psi_train ** (node.path[i])) * ((1-eta.psi_train) ** (1-node.path[i]))
                phi_test = phi_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
            leaf_probs_train[ii,:] = phi_train
            leaf_probs_test[ii,:] = phi_test
            theta[ii] = np.argmax(np.sum(phi_train*tree.indicies,axis=1))
        leaf_ids_train = np.argmax(leaf_probs_train,axis=0)
        leaf_ids_test = np.argmax(leaf_probs_test,axis=0)
        miss_train = np.sum(data['y_train'] != theta[leaf_ids_train])/data['n_train']
        miss_test = np.sum(data['y_test'] != theta[leaf_ids_test])/data['n_test']
        for node in tree.internal_nodes: # clean up
            delattr(node, 'psi_train')
            delattr(node, 'psi_test')
        return {'miss_train':miss_train,'miss_test':miss_test}
    else:
        for i,node in enumerate(tree.internal_nodes): 
            if(data['nx']==1): # only one input variable
                node.psi_train = expit((x_train[:,0]-node.params['tau'])/h)
                node.psi_test = expit((data['x_test'][:,0]-node.params['tau'])/h)
            else:
                node.psi_train = expit((np.dot(x_train,node.params['index'])-node.params['tau'])/h)
                node.psi_test = expit((np.dot(data['x_test'],node.params['index'])-node.params['tau'])/h)
        leaf_probs_train = np.zeros((tree.nl,data['n_train'])) 
        leaf_probs_test = np.zeros((tree.nl,data['n_test'])) 
        mu = np.zeros(tree.nl)
        for ii, node in enumerate(tree.leaf_nodes):
            mu[ii] = node.params['mu']
            phi_train = 1
            phi_test = 1
            for i,eta in enumerate(node.ancestors): 
                phi_train = phi_train * (eta.psi_train ** (node.path[i])) * ((1-eta.psi_train) ** (1-node.path[i]))
                phi_test = phi_test * (eta.psi_test ** (node.path[i])) * ((1-eta.psi_test) ** (1-node.path[i]))
            leaf_probs_train[ii,:] = phi_train
            leaf_probs_test[ii,:] = phi_test
        leaf_ids_train = np.argmax(leaf_probs_train,axis=0)
        leaf_ids_test = np.argmax(leaf_probs_test,axis=0)
        mse_train = np.sum(np.power(mu[leaf_ids_train]-data['y_train'],2))/data['n_train']
        mse_test = np.sum(np.power(mu[leaf_ids_test]-data['y_test'],2))/data['n_test']
        for node in tree.internal_nodes: # clean up
            delattr(node, 'psi_train')
            delattr(node, 'psi_test')
        return {'mse_train':mse_train,'mse_test':mse_test}

def calculate_true_statistics(tree,data,settings):
    settings_true = copy.deepcopy(settings)
    tree_true = copy.deepcopy(tree)

    if(settings_true.dataset == 'large-real'): # large dataset
        settings_true.children_left = [1, 3, 5, 7, -1, 9, -1, -1, 11, -1, -1, -1, -1]
        settings_true.children_right = [2, 4, 6, 8, -1, 10, -1, -1, 12, -1, -1, -1, -1]
        sigma = 0.1 # assuming constant variance across all leaf nodes 
        index = np.array([[1,0,0,0,0],[0,0,0,1,0],[0,1,0,0,0],[0,0,1,0,0],[0,1,0,0,0],[0,0,0,0,1]])
        mu = np.array([3,2,6,1,7,4,5])
        tau = np.array([7,5,3.5,2,3,6])
    elif(settings_true.dataset == 'cgm'): # CGM dataset
        settings_true.children_left = [1, 3, 5, -1, 7, -1, -1, -1, -1]
        settings_true.children_right = [2, 4, 6, -1, 8, -1, -1, -1, -1]
        sigma = 0.2 # assuming constant variance across all leaf nodes 
        index = np.array([[0,1],[1,0],[1,0],[1,0]])
        mu = np.array([1,5,8,8,2])
        tau = np.array([4.5,3.5,7.5,5.5])
    elif(settings_true.dataset == 'toy-non-sym'): # toy-non-sym dataset	
        settings_true.children_left = [1, 3, 5, -1, -1, -1, -1]	
        settings_true.children_right = [2, 4, 6, -1, -1, -1, -1]	
        index = np.array([[1,0],[0,1],[0,1]])	
        mu = np.array([1,0,0,1])	
        tau = np.array([0.5,0.3,0.7])
        sigma = 0.1 #0.04
    elif((settings_true.dataset == 'toy-class') or (settings_true.dataset == 'toy-class-noise')): # toy classification dataset
        settings_true.children_left = [1,-1,-1]
        settings_true.children_right = [2,-1,-1]
        index = np.array([[0,1,0]])
        tau = np.array([0.5])
        theta = np.array([[1.0,0.0],[0.0,1.0]])
    elif(settings_true.dataset == 'test-1'): 
        settings_true.children_left = [1, 3, 5, -1, -1, -1, -1]	
        settings_true.children_right = [2, 4, 6, -1, -1, -1, -1]
        index = np.array([0,1,1])
        tau = np.array([0.5,0.5,0.5])
        theta = np.array([[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]])
    else:
        print("NOTE: True tree statistics have not been implemented for this dataset. \n\n\n")
        return None, None, None
        # raise NotImplementedError("True tree statistics have not been implemented for this dataset.")

    tree_true = BayesianTree(data=data,settings=settings_true)
    for i,node in enumerate(tree_true.leaf_nodes):
        if(data['output_type']=='class'):
            node.params = {'theta':theta[i]}
        else:
            node.params = {'mu':mu[i]}
    for i,node in enumerate(tree_true.internal_nodes):
        if(settings.method == 0):
            node.params = {'index':np.argmax(index[i]),'tau':tau[i]}
        else:
            node.params = {'index':index[i],'tau':tau[i]}

    if(settings.method == 0):
        metrics_true = evaluate_tree_df(tree_true,data,settings.h_final)
    else:
        metrics_true = evaluate_tree_dfi(tree_true,data,settings.h_final)
    if(data['output_type']=='class'):
        print("Missclassification of true tree: training - " + str(metrics_true['miss_train']) + " testing - " + str(metrics_true['miss_test']))
        if(settings.method == 0):
            pars = {'index':jnp.array([[np.argmax(index)]]),'tau':jnp.array([(tau-data['transform']['min'][np.argmax(index,axis=1)])/data['transform']['range'][np.argmax(index,axis=1)]])}
        else:
            pars = {'index':jnp.array([index]),'tau':jnp.array([(tau-data['transform']['min'][np.argmax(index,axis=1)])/data['transform']['range'][np.argmax(index,axis=1)]])}
    else:
        print("MSE of true tree: training - " + str(metrics_true['mse_train']) + " testing - " + str(metrics_true['mse_test']))	
        if(settings.method == 0):
            pars = {'index':jnp.array([[np.argmax(index)]]),'mu':jnp.array([mu]),'sigma':jnp.array([sigma]),'tau':jnp.array([(tau-data['transform']['min'][np.argmax(index,axis=1)])/data['transform']['range'][np.argmax(index,axis=1)]])}
        else:
            pars = {'index':jnp.array([index]),'mu':jnp.array([mu]),'sigma':jnp.array([sigma]),'tau':jnp.array([(tau-data['transform']['min'][np.argmax(index,axis=1)])/data['transform']['range'][np.argmax(index,axis=1)]])}

    if(settings.method == 0):
        log_llh_true = log_likelihood(tree_true.tree_model_df,pars,data,settings.h_final)
    else:
        log_llh_true = log_likelihood(tree_true.tree_model_dfi,pars,data,settings.h_final)
    print("Log likelihood of true tree: " + str(log_llh_true))

    return log_llh_true['log_prob'], metrics_true, tree_true.nl