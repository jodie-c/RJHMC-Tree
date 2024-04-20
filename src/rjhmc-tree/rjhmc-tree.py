# Import required packages
import time
import copy
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import numpyro
from numpyro.infer import MCMC, HMC, DiscreteHMCGibbs, NUTS, init_to_value
from numpyro.infer.util import initialize_model
from numpyro.util import fori_collect
from numpyro.infer.hmc import hmc
from numpyro.infer.util import log_density
from jax import random
from jax.scipy.special import logsumexp
from utils import *
from class_defs import *

# Set plotting style
plt.style.use('seaborn-darkgrid')
# sns.set()  # nice plot aesthetic

################## Main Function ##################
def main():
    settings = process_command_line()

    fname = settings.dataset + "-init_id-" + str(settings.init_id) \
        + "-h_i-" + str(settings.h_init) + "-h_f-" + str(settings.h_final) \
        + "-N_warmup-" + str(settings.N_warmup) + "-N-" + str(settings.N) \
        + "-alpha-" + str(settings.alpha) + "-beta-" + str(settings.beta) \
        + "-method-" + str(settings.method) + "-tag-" + str(settings.tag) + "-results.pickle"

    # Resetting random seed
    np.random.seed(settings.init_id * 1000)
    rng_key = random.PRNGKey(settings.init_id) 
    data = load_data(settings)
    print("\nSettings:")
    print(vars(settings))
    # plot_dataset(data)

    data = process_dataset(data) # converts to jnp arrays; scales input variables to be on interval [0,1] --> required for B/D/R global proposals

    tree = BayesianTree(data=data,settings=settings) # set up Bayesian Tree object

    # Calculate true tree statistics - returns None if true tree is not implemented (i.e. if using real-world dataset)
    log_llh_true, metrics_true, nl_true = calculate_true_statistics(tree,data,settings)

    # Set up sampling information for the inner MCMC loop methods
    info = MCMC_Info(move_probs = settings.warmup_probs, # if no warmup, this will be changed immediately 
                    h_info = {'h_init':settings.h_init,'h_final':settings.h_final},
                    num_warmup = settings.hmc_num_warmup,
                    filename = fname) # initalisation is included by default

    # --------------------------- Run Hierarachical MCMC Algorithm --------------------------- #
    time_start = time.perf_counter()
    if(info.init): # run initialisation of tree parameters and log-likelihood
        rng_key, subkey = random.split(rng_key,2)
        if(settings.method == 0): # HMC-DF
            kernel = DiscreteHMCGibbs(NUTS(tree.tree_model_df,h_info=info.h_info),random_walk=True)
        else: # HMC-DFI
            kernel = NUTS(tree.tree_model_dfi,h_info=info.h_info)
        mcmc= MCMC(kernel, num_samples=info.num_samps, num_warmup=info.num_warmup,progress_bar=False)
        mcmc.run(subkey,data,h=info.h_info['h_init'])
        run_samps = mcmc.get_samples()

        # Compute log-likelihood and log-probability of new tree
        if(data['nx'] > 1):
            if(data['output_type'] == 'class'):
                pars_llh = {'index':jnp.array(run_samps['index']),'tau':jnp.array(run_samps['tau'])}
                pars_ld = {'index':jnp.array(run_samps['index'][0]),'tau':jnp.array(run_samps['tau'][0])}
            else:
                pars_llh = {'index':jnp.array(run_samps['index']),'tau':jnp.array(run_samps['tau']),'mu':jnp.array(run_samps['mu']),'sigma':jnp.array(run_samps['sigma'])}
                pars_ld = {'index':jnp.array(run_samps['index'][0]),'tau':jnp.array(run_samps['tau'][0]),'mu':jnp.array(run_samps['mu'][0]),'sigma':jnp.array(run_samps['sigma'][0])}
            if(settings.method == 0):
                tree.log_llh = log_likelihood(tree.tree_model_df,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
                tree.log_prob = log_density(tree.tree_model_df,(data,settings.h_final),{},params=pars_ld)[0] + tree.prior['logp']  # update log-posterior
            else:
                tree.log_llh = log_likelihood(tree.tree_model_dfi,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
                tree.log_prob = log_density(tree.tree_model_dfi,(data,settings.h_final),{},params=pars_ld)[0] + tree.prior['logp']  # update log-posterior
        else:
            if(data['output_type'] == 'class'):
                pars_llh = {'tau':jnp.array(run_samps['tau'])}
                pars_ld = {'tau':jnp.array(run_samps['tau'][0]),'mu':jnp.array(run_samps['mu'][0]),'sigma':jnp.array(run_samps['sigma'][0])}
            else:
                pars_llh = {'tau':jnp.array(run_samps['tau']),'mu':jnp.array(run_samps['mu']),'sigma':jnp.array(run_samps['sigma'])}
                pars_ld = {'tau':jnp.array(run_samps['tau'][0]),'mu':jnp.array(run_samps['mu'][0]),'sigma':jnp.array(run_samps['sigma'][0])}
            if(settings.method == 0):
                tree.log_llh = log_likelihood(tree.tree_model_df,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
                tree.log_prob = log_density(tree.tree_model_df,(data,settings.h_final),{},params=pars_ld)[0] + tree.prior['logp']  # update log-posterior
            else:
                tree.log_llh = log_likelihood(tree.tree_model_dfi,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
                tree.log_prob = log_density(tree.tree_model_dfi,(data,settings.h_final),{},params=pars_ld)[0] + tree.prior['logp']  # update log-posterior

        # Add proposed index/thresholds to tree structure 
        tree.update_params(pars_ld,data)

        if(settings.method == 0): # HMC-DF
            tree.num_unique_feats = len(np.unique([run_samps['index']])) 
            tree.metrics = evaluate_tree_df(tree,data,info.h_info['h_final']) # compute metrics of inital tree
        else:
            tree.num_unique_feats = -1 # NOTE unique feats don't make sense for soft trees
            tree.metrics = evaluate_tree_dfi(tree,data,info.h_info['h_final']) # compute metrics of inital tree

        if(data['output_type'] == 'class'):
            print("Missclassification rate of initial tree: training - " + str(tree.metrics['miss_train']) + " testing - " + str(tree.metrics['miss_test']))
            metrics = [tree.metrics['miss_test']]
            metrics_train = [tree.metrics['miss_train']]
        else:
            print("Mean-square-error of initial tree: training - " + str(tree.metrics['mse_train']) + " testing - " + str(tree.metrics['mse_test']))
            metrics = [tree.metrics['mse_test']]
            metrics_train = [tree.metrics['mse_train']]
        print("Log-prob of initial tree: ", tree.log_prob)

    # Initialise arrays to store metrics
    num_leaves = [np.size(tree.leaf_nodes)]
    num_unique_feats = [tree.num_unique_feats]
    log_llh = [tree.log_llh]
    log_prob = [tree.log_prob]
    tree_samples = {}
    tree_samples['0'] = {"I":tree.internal_nodes,"L":tree.leaf_nodes}

    time_init = time.perf_counter()
    info.warmup = True

    for i in range(settings.N+settings.N_warmup-1):
        rng_key, subkey = random.split(rng_key,2)
        if(i == settings.N_warmup):
            time_warmup = time.perf_counter()
            info.prob_grow_prune = settings.sample_probs[0]+settings.sample_probs[1]      # probability of grow/prune move
            info.prob_grow = settings.sample_probs[0]/info.prob_grow_prune     # probability of grow move
            info.warmup = False
            # info.accept = 0 # uncomment to reset accepts after warm-up phase
        tree = rjhmc_tree_sample(i,tree,info,data,settings,subkey)
        tree_samples[str(i+1)] = {"I":tree.internal_nodes,"L":tree.leaf_nodes}
        num_leaves.append(np.size(tree.leaf_nodes))
        log_llh.append(tree.log_llh)
        log_prob.append(tree.log_prob)
        num_unique_feats.append(tree.num_unique_feats)
        if(data['output_type']=='class'):
            metrics.append(tree.metrics['miss_test'])
            metrics_train.append(tree.metrics['miss_train'])
        else:
            metrics.append(tree.metrics['mse_test'])
            metrics_train.append(tree.metrics['mse_train'])

    time_finish = time.perf_counter()
    print("Time taken to sample (+warm-up) N=" + str(settings.N) + " trees: " + str(time_finish-time_start))

    if(data['output_type'] == 'class'):
        print("Missclassification rate of final tree: training - " + str(metrics_train[-1]) + " testing - " + str(metrics[-1]))
    else:
        print("Mean-square-error of final tree: training - " + str(metrics_train[-1]) + " testing - " + str(metrics[-1]))

    if(settings.plot_info == 1):
        fig, axes = plt.subplots(3,1)
        axes[0].plot(num_leaves)
        if(nl_true is not None):
            axes[0].axhline(nl_true,ls='--',color='r')
        axes[0].set_title("Number of leaf nodes")
        axes[1].plot(log_llh)
        axes[1].set_title("Log-Likelihood") 
        if(log_llh_true is not None): 
            axes[1].axhline(log_llh_true,ls='--',color='r')   
        axes[2].plot(metrics)
        axes[2].set_title("Missclassification/Mean-square-error")
        if(metrics_true is not None): 
            axes[2].axhline(metrics_true['mse_test'],ls='--',color='r')  
        fig.tight_layout()   
        fig.savefig(settings.out_dir + info.filename + "-metrics.pdf")
        plt.close()

    print("\n\n\n\nNumber of Move Accepts (including warm-up): ", info.accept)
    print("\nPercentage of Move Accepts (including warm-up): ", (info.accept/(settings.N+settings.N_warmup)*100))

    if(settings.save == 1):
        print('file path: ' + settings.out_dir + info.filename)
        results = {'settings': settings, 'mcmc_info': info,
                    'samples': tree_samples,
                    'time_total': time_finish-time_start, 'time_method': time_finish-time_init, \
                    'time_init': time_init-time_start, 'time_method_sans_warmup': time_finish-time_warmup,\
                    'log_llh': log_llh, 'log_prob': log_prob, \
                    'metrics': metrics, 'metrics_train':metrics_train, \
                    'num_leaves': num_leaves, \
                    'num_unique_feats': num_unique_feats,\
                    'num_accepts': info.accept, \
                    'percent_accept': (info.accept/(settings.N+settings.N_warmup)*100), 
                }
        if(log_llh_true is not None):
            results['log_llh_true'] = log_llh_true
            results['metrics_true'] = metrics_true
            results['nl_true'] = nl_true
        pickle.dump(results, open(settings.out_dir+info.filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

#--------------------------------------------------
# Single iteration of the RJHMC-Tree method 
# Note logp_rv_ratio is updated in multiple places depending on grow or prune transition
def rjhmc_tree_sample(iteration,tree,info,data,settings,key):
    subkey1, subkey2, subkey3, subkey4 = random.split(key,4)
    # Structural/topological proposal --> global proposals
    print("\n\n---------------- Iter: ",iteration," ----------------\n")
    tree_new = copy.deepcopy(tree) 
    u = random.uniform(subkey1)
    if (u < info.prob_grow_prune): # generate grow/prune proposal
        tree_new, logp_transition_ratio, logp_struct_ratio, logp_rv_ratio = generate_transition(tree_new,info,data,settings) 
    else: # stay proposal
        print("\n----- stay proposal -----\n")
        logp_struct_ratio = 0 # stay at same tree structure and only do local proposal
        logp_transition_ratio = 0 # no transition so ratio cancels out --> log(1) = 0
        logp_rv_ratio = 0 # no additional random variables drawn for stay proposal (no jump in dimension)
        info.stay_prop += 1

    # Set up current parameters
    tau_tmp_m = np.zeros((tree.n,))
    if(settings.method == 0):
        index_tmp_m = np.zeros((tree.n,),dtype=int)
    else:
        index_tmp_m = np.zeros((tree.n,data['nx']))
    for i,node in enumerate(tree.internal_nodes):
        index_tmp_m[i] = node.params['index']
        tau_tmp_m[i] = node.params['tau']
    if(data['output_type'] == 'class'):
        if(settings.method == 0):
            kernel = DiscreteHMCGibbs(NUTS(tree.tree_model_df,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_m})),random_walk=True,init_strategy=init_to_value(values={'index':index_tmp_m}))
        else:
            kernel = NUTS(tree.tree_model_dfi,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_m,'index':index_tmp_m}))
    else:     
        mu_tmp_m = np.zeros((tree.nl,))
        for i,node in enumerate(tree.leaf_nodes):
            mu_tmp_m[i] = node.params['mu']
            sigma_tmp_m = node.params['sigma']
        if(settings.method == 0):
            kernel = DiscreteHMCGibbs(NUTS(tree.tree_model_df,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_m,'mu':mu_tmp_m,'sigma':sigma_tmp_m})),random_walk=True,init_strategy=init_to_value(values={'index':index_tmp_m}))
        else:
            kernel = NUTS(tree.tree_model_dfi,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_m,'index':index_tmp_m,'mu':mu_tmp_m,'sigma':sigma_tmp_m}))

    # # Simulate new proposal via HMC in current dimension/tree
    # model_info = initialize_model(subkey2, tree.model_tree_soft, model_args=(data,settings.h_final),init_strategy=init_to_value(values={'index':index_tmp_m,'tau':tau_tmp_m,'mu':mu_tmp_m,'sigma':sigma_tmp_m}))
    # init_kernel, sample_kernel = hmc(model_info.potential_fn,algo="HMC")
    # hmc_state = init_kernel(model_info.param_info,step_size=info.step_size,adapt_step_size=False,
    #                         # inverse_mass_matrix=0.001*jnp.eye(tree.n),step_size=0.2, # 0.015
    #                         num_warmup=info.num_warmup,rng_key=subkey2)
    # proposal = fori_collect(0, info.num_samps+info.num_warmup, sample_kernel, hmc_state, progbar=False, return_last_val=True)[1]
    # run_samps_m = model_info.postprocess_fn(proposal.z)

    # HMC-within-MH step --> local proposals
    mcmc = MCMC(kernel, num_samples=info.num_samps, num_warmup=info.num_warmup, progress_bar=False)
    mcmc.run(subkey2,data,h=info.h_info['h_init'])
    run_samps_m = mcmc.get_samples()

    # Warm start parameter values in new dimension
    tau_tmp_n = np.zeros((tree_new.n,))
    if(settings.method == 0):
        index_tmp_n = np.zeros((tree_new.n,),dtype=int)
    else:
        index_tmp_n = np.zeros((tree_new.n,data['nx']))
    j = 0
    for i,node in enumerate(tree_new.internal_nodes):
        if 'tau' in node.params:
            index_tmp_n[i] = run_samps_m['index'][0][j]
            tau_tmp_n[i] = run_samps_m['tau'][0][j]
            j += 1
        else:
            tau_tmp_n[i] = np.random.beta(1,1)
            logp_rv_ratio -= dist.Beta(1,1).log_prob(tau_tmp_n[i]) # new rv for grow move
            if(settings.method == 0):
                index_tmp_n[i] = np.random.choice(data['nx'])
                logp_rv_ratio -= dist.Categorical(probs=np.array([1/data['nx']]*data['nx'])).log_prob(index_tmp_n[i])
            else:
                index_tmp_n[i] = np.random.dirichlet([1]*data['nx'])
                logp_rv_ratio -= dist.Dirichlet(np.ones((data['nx'],))).log_prob(index_tmp_n[i]) # new rv for grow move
    if(data['output_type']=='class'):
        if(settings.method == 0):
            kernel = DiscreteHMCGibbs(NUTS(tree_new.tree_model_df,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_n})),random_walk=True,init_strategy=init_to_value(values={'index':index_tmp_n}))
        else:
            kernel = NUTS(tree_new.tree_model_dfi,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_n,'index':index_tmp_n}))
    else:
        mu_tmp_n = np.zeros((tree_new.nl,))
        j = 0
        for i,node in enumerate(tree_new.leaf_nodes):
            if 'mu' in node.params:
                mu_tmp_n[i] = run_samps_m['mu'][0][j]
                sigma_tmp_n = run_samps_m['sigma'][0]
                j += 1
            else:
                mu_tmp_n[i] = np.random.normal(tree_new.mu_mean_llh,tree.mu_var_llh)
                logp_rv_ratio -= dist.Normal(tree_new.mu_mean_llh,tree.mu_var_llh).log_prob(mu_tmp_n[i]) # new rv for grow move
        if(settings.method == 0):
            kernel = DiscreteHMCGibbs(NUTS(tree_new.tree_model_df,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_n,'mu':mu_tmp_n,'sigma':sigma_tmp_n})),random_walk=True,init_strategy=init_to_value(values={'index':index_tmp_n}))
        else:
            kernel = NUTS(tree_new.tree_model_dfi,h_info=info.h_info,init_strategy=init_to_value(values={'tau':tau_tmp_n,'index':index_tmp_n,'mu':mu_tmp_n,'sigma':sigma_tmp_n}))

    # model_info = initialize_model(subkey3, tree_new.model_tree_soft, model_args=(data,settings.h_final),init_strategy=init_to_value(values={'index':index_tmp_n,'tau':tau_tmp_n,'mu':mu_tmp_n,'sigma':sigma_tmp_n}))
    # init_kernel, sample_kernel = hmc(model_info.potential_fn,algo="HMC")
    # hmc_state = init_kernel(model_info.param_info,step_size=info.step_size,adapt_step_size=False,
    #                         # inverse_mass_matrix=0.001*jnp.eye(tree.n),step_size=0.2, # 0.015
    #                         num_warmup=info.num_warmup,rng_key=subkey3)
    # proposal = fori_collect(0, info.num_samps+info.num_warmup, sample_kernel, hmc_state, progbar=False, return_last_val=True)[1]
    # run_samps_n = model_info.postprocess_fn(proposal.z)

    # HMC-within-MH step --> local proposals
    mcmc = MCMC(kernel, num_samples=info.num_samps, num_warmup=info.num_warmup, progress_bar=False)
    mcmc.run(subkey3,data,h=info.h_info['h_init'])
    run_samps_n = mcmc.get_samples()

    # Compute log-likelihood and log-probability of new tree
    tree_new.prior['logp'] += logp_struct_ratio
    if(data['nx'] > 1):
        if(data['output_type'] == 'class'):
            pars_llh = {'index':jnp.array(run_samps_n['index']),'tau':jnp.array(run_samps_n['tau'])}
            pars_ld = {'index':jnp.array(run_samps_n['index'][0]),'tau':jnp.array(run_samps_n['tau'][0])}
        else:
            pars_llh = {'index':jnp.array(run_samps_n['index']),'tau':jnp.array(run_samps_n['tau']),'mu':jnp.array(run_samps_n['mu']),'sigma':jnp.array(run_samps_n['sigma'])}
            pars_ld = {'index':jnp.array(run_samps_n['index'][0]),'tau':jnp.array(run_samps_n['tau'][0]),'mu':jnp.array(run_samps_n['mu'][0]),'sigma':jnp.array(run_samps_n['sigma'][0])}
        if(settings.method == 0):
            tree_new.log_llh = log_likelihood(tree_new.tree_model_df,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
            tree_new.log_prob = log_density(tree_new.tree_model_df,(data,settings.h_final),{},params=pars_ld)[0] + tree_new.prior['logp']  # update log-posterior
        else:
            tree_new.log_llh = log_likelihood(tree_new.tree_model_dfi,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
            tree_new.log_prob = log_density(tree_new.tree_model_dfi,(data,settings.h_final),{},params=pars_ld)[0] + tree_new.prior['logp']  # update log-posterior
    else:
        if(data['output_type'] == 'class'):
            pars_llh = {'tau':jnp.array(run_samps_n['tau'])}
            pars_ld = {'tau':jnp.array(run_samps_n['tau'][0]),'mu':jnp.array(run_samps_n['mu'][0]),'sigma':jnp.array(run_samps_n['sigma'][0])}
        else:
            pars_llh = {'tau':jnp.array(run_samps_n['tau']),'mu':jnp.array(run_samps_n['mu']),'sigma':jnp.array(run_samps_n['sigma'])}
            pars_ld = {'tau':jnp.array(run_samps_n['tau'][0]),'mu':jnp.array(run_samps_n['mu'][0]),'sigma':jnp.array(run_samps_n['sigma'][0])}
        if(settings.method == 0):
            tree_new.log_llh = log_likelihood(tree_new.tree_model_df,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
            tree_new.log_prob = log_density(tree_new.tree_model_df,(data,settings.h_final),{},params=pars_ld)[0] + tree_new.prior['logp']  # update log-posterior
        else:
            tree_new.log_llh = log_likelihood(tree_new.tree_model_dfi,pars_llh,data,h=settings.h_final)['log_prob'] # sample tree based on potential energy/weights
            tree_new.log_prob = log_density(tree_new.tree_model_dfi,(data,settings.h_final),{},params=pars_ld)[0] + tree_new.prior['logp']  # update log-posterior

    tree_new.update_params(pars_ld,data)

    if(settings.method == 0): # HMC-DF
        tree_new.num_unique_feats = len(np.unique([run_samps_n['index']])) 
        tree_new.metrics = evaluate_tree_df(tree_new,data,info.h_info['h_final']) # compute metrics of inital tree
    else:
        tree_new.num_unique_feats = -1 # NOTE unique feats don't make sense for soft trees
        tree_new.metrics = evaluate_tree_dfi(tree_new,data,info.h_info['h_final']) # compute metrics of inital tree

    # Compute ratio pi_star_n(x'')/pi_star_n(x*)
    if(settings.method == 0): # HMC-DF
        if(data['output_type']=='class'):
            logp_star_n_ratio = log_density(tree_new.tree_model_df,(data,settings.h_final),{},params={'index':index_tmp_n,'tau':tau_tmp_n})[0] - log_density(tree_new.tree_model_df,(data,settings.h_final),{},params={'index':jnp.array(run_samps_n['index'][0]),'tau':jnp.array(run_samps_n['tau'][0])})[0]
        else:
            logp_star_n_ratio = log_density(tree_new.tree_model_df,(data,settings.h_final),{},params={'index':index_tmp_n,'tau':tau_tmp_n,'mu':mu_tmp_n,'sigma':sigma_tmp_n})[0] - log_density(tree_new.tree_model_df,(data,settings.h_final),{},params={'index':jnp.array(run_samps_n['index'][0]),'tau':jnp.array(run_samps_n['tau'][0]),'mu':jnp.array(run_samps_n['mu'][0]),'sigma':jnp.array(run_samps_n['sigma'][0])})[0]
    else:
        if(data['output_type'] == 'class'):
            logp_star_n_ratio = log_density(tree_new.tree_model_dfi,(data,settings.h_final),{},params={'index':index_tmp_n,'tau':tau_tmp_n})[0] - log_density(tree_new.tree_model_dfi,(data,settings.h_final),{},params={'index':jnp.array(run_samps_n['index'][0]),'tau':jnp.array(run_samps_n['tau'][0])})[0]
        else:
            logp_star_n_ratio = log_density(tree_new.tree_model_dfi,(data,settings.h_final),{},params={'index':index_tmp_n,'tau':tau_tmp_n,'mu':mu_tmp_n,'sigma':sigma_tmp_n})[0] - log_density(tree_new.tree_model_dfi,(data,settings.h_final),{},params={'index':jnp.array(run_samps_n['index'][0]),'tau':jnp.array(run_samps_n['tau'][0]),'mu':jnp.array(run_samps_n['mu'][0]),'sigma':jnp.array(run_samps_n['sigma'][0])})[0]

    # Compute ratio pi_star_m(x)/pi_star_m(x')
    if(settings.method == 0): # HMC-DF
        if(data['output_type']=='class'):
            logp_star_m_ratio = log_density(tree.tree_model_df,(data,settings.h_final),{},params={'index':index_tmp_m,'tau':tau_tmp_m})[0] - log_density(tree.tree_model_df,(data,settings.h_final),{},params={'index':jnp.array(run_samps_m['index'][0]),'tau':jnp.array(run_samps_m['tau'][0])})[0]
        else:
            logp_star_m_ratio = log_density(tree.tree_model_df,(data,settings.h_final),{},params={'index':index_tmp_m,'tau':tau_tmp_m,'mu':mu_tmp_m,'sigma':sigma_tmp_m})[0] - log_density(tree.tree_model_df,(data,settings.h_final),{},params={'index':jnp.array(run_samps_m['index'][0]),'tau':jnp.array(run_samps_m['tau'][0]),'mu':jnp.array(run_samps_m['mu'][0]),'sigma':jnp.array(run_samps_m['sigma'][0])})[0]
    else:
        if(data['output_type']=='class'):
            logp_star_m_ratio = log_density(tree.tree_model_dfi,(data,settings.h_final),{},params={'index':index_tmp_m,'tau':tau_tmp_m})[0] - log_density(tree.tree_model_dfi,(data,settings.h_final),{},params={'index':jnp.array(run_samps_m['index'][0]),'tau':jnp.array(run_samps_m['tau'][0])})[0]
        else:
            logp_star_m_ratio = log_density(tree.tree_model_dfi,(data,settings.h_final),{},params={'index':index_tmp_m,'tau':tau_tmp_m,'mu':mu_tmp_m,'sigma':sigma_tmp_m})[0] - log_density(tree.tree_model_dfi,(data,settings.h_final),{},params={'index':jnp.array(run_samps_m['index'][0]),'tau':jnp.array(run_samps_m['tau'][0]),'mu':jnp.array(run_samps_m['mu'][0]),'sigma':jnp.array(run_samps_m['sigma'][0])})[0]        

    # Calculate acceptance probability - two-sided RJHMC
    # Note log_prob = -U(q) = -log(D|q)-log(q), therefore also includes priors on parameters
    if(info.warmup): # still in warm-up stage - use biased acceptance probability
        lalpha = logp_transition_ratio + logp_rv_ratio + tree_new.log_prob - tree.log_prob
    else:
        lalpha = logp_transition_ratio + logp_star_n_ratio + logp_star_m_ratio + logp_rv_ratio + tree_new.log_prob - tree.log_prob

    # Do MH update
    if(np.log(random.uniform(subkey4)) < lalpha):
        info.accept += 1
        if(settings.plot_info == 1):
            fig, axes = plt.subplots(1,2)
            tree.draw_tree(ax=axes[0])
            tree_new.draw_tree(ax=axes[1])
            if(data['output_type'] == 'class'):
                axes[0].set_title("tree log-prob: "+"{:.2f}".format(tree.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree.log_llh)+ \
                    "\n miss train: {:.4f}".format(tree.metrics['miss_train'])+" test: {:.4f}".format(tree.metrics['miss_test']),fontsize="small")
                axes[1].set_title("tree-new log-prob: "+"{:.2f}".format(tree_new.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree_new.log_llh)+ \
                    "\n miss train: {:.4f}".format(tree_new.metrics['miss_train'])+" test: {:.4f}".format(tree_new.metrics['miss_test']),fontsize="small")
            else:
                axes[0].set_title("tree log-prob: "+"{:.2f}".format(tree.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree.log_llh)+ \
                    "\nmse: "+"{:.4f}".format(tree.metrics['mse_test'])+"\nsigma: "+"{:.2f}".format(tree.leaf_nodes[0].params['sigma']),fontsize="small")
                axes[1].set_title("tree-new log-prob: "+"{:.2f}".format(tree_new.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree_new.log_llh)+ \
                    "\nmse: "+"{:.4f}".format(tree_new.metrics['mse_test'])+"\nsigma: "+"{:.2f}".format(tree_new.leaf_nodes[0].params['sigma']),fontsize="small")
            fig.suptitle("move accepted",color="green",fontsize="small")
            fig.tight_layout()
            fig.savefig(settings.out_dir+info.filename+"-iter"+str(iteration)+".pdf")
            plt.close()
        print("\n Tree ACCEPTED \n")
        if(data['output_type'] == 'class'):
            print("Missclasification -- train: ",tree_new.metrics['miss_train'],"\ttest: ",tree_new.metrics['miss_test'],"\nlogp: ",tree_new.log_prob)
        else:
            print("Mean-square-error -- train: ",tree_new.metrics['mse_train'],"\ttest: ",tree_new.metrics['mse_test'],"\nlogp: ",tree_new.log_prob)
        del tree
        return tree_new
    else:
        if(settings.plot_info == 1):
            fig, axes = plt.subplots(1,2)
            tree.draw_tree(ax=axes[0])
            tree_new.draw_tree(ax=axes[1])
            if(data['output_type'] == 'class'):
                axes[0].set_title("tree log-prob: "+"{:.2f}".format(tree.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree.log_llh)+ \
                    "\n miss train: {:.4f}".format(tree.metrics['miss_train'])+" test: {:.4f}".format(tree.metrics['miss_test']),fontsize="small")
                axes[1].set_title("tree-new log-prob: "+"{:.2f}".format(tree_new.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree_new.log_llh)+ \
                    "\n miss train: {:.4f}".format(tree_new.metrics['miss_train'])+" test: {:.4f}".format(tree_new.metrics['miss_test']),fontsize="small")
            else:
                axes[0].set_title("tree log-prob: "+"{:.2f}".format(tree.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree.log_llh)+ \
                    "\nmse: "+"{:.4f}".format(tree.metrics['mse_test'])+"\nsigma: "+"{:.2f}".format(tree.leaf_nodes[0].params['sigma']),fontsize="small")
                axes[1].set_title("tree-new log-prob: "+"{:.2f}".format(tree_new.log_prob)+ \
                    "\nlog-llh: "+"{:.2f}".format(tree_new.log_llh)+ \
                    "\nmse: "+"{:.4f}".format(tree_new.metrics['mse_test'])+"\nsigma: "+"{:.2f}".format(tree_new.leaf_nodes[0].params['sigma']),fontsize="small")
            fig.suptitle("move rejected",color="red",fontsize="small")
            fig.tight_layout()
            fig.savefig(settings.out_dir+info.filename+"-iter"+str(iteration)+".pdf")
            plt.close()
        print("\n Tree REJECTED \n")
        if(data['output_type'] == 'class'):
            print("Missclasification -- train: ",tree_new.metrics['miss_train'],"\ttest: ",tree_new.metrics['miss_test'],"\nlogp: ",tree_new.log_prob)
        else:
            print("Mean-square-error -- train: ",tree_new.metrics['mse_train'],"\ttest: ",tree_new.metrics['mse_test'],"\nlogp: ",tree_new.log_prob)
        del tree_new
        return tree

#--------------------------------------------------
# Generate grow or prune global proposal
#--------------------------------------------------
def generate_transition(tree, info, data, settings):
    """
    Generates either a grow or prune proposal to transition from one dimension to another. 
    Also calculates the probability of the transition move and the change in topology ratio

    This function was adapted from: brtfuns.cpp: Base BART model class helper functions. Copyright (C) 2012-2016 Matthew T. Pratola, Robert E. McCulloch and Hugh A. Chipman
    """
    # Find leaf nodes we could grow at (split on) and prob of a grow proposal at x
    leaves_grow, prob_grow = get_prob_grow(tree, info.prob_grow) 
    nognds = tree.get_no_grandchildren()

    logp_rv_ratio = 0 # initialise density of random variables

    if(np.random.rand() < prob_grow): # if true, do grow transition, else prune
        print("\n----- grow proposal -----\n")
        info.grow_prop += 1 

        #--------------------------------------------------
        # Draw leaf node, choose node uniformly from list in leaves_grow
        node = np.random.choice(leaves_grow) # the leaf node we might grow at

        #--------------------------------------------------
        # Compute things needed for metropolis ratio
        logp_nx = -np.log(len(leaves_grow)) # proposal prob of choosing node nx
        depth = node.get_depth()
        p_grow_nx = p_split(tree,depth) # prior prob of growing at this node

        # Prior probs of growing at new children (l and r) of proposal
        p_grow_l1 = p_split(tree,depth+1) # depth of new nodes would be one more
        p_grow_l2 = p_grow_l1
        
        #--------------------------------------------------
        # Prob of proposing death at ny
        if(len(leaves_grow)>1): # can grow at new tree Ty because splittable nodes left
            logp_prune_ny = np.log(1 - info.prob_grow)
        else: # nx was the only node you could split on
            if((p_grow_l1 == 0) & (p_grow_l2 == 0)): # cannot grow at y
                logp_prune_ny = 0 # prob prune = 1
            else: # y can grow at either child
                logp_prune_ny = np.log(1 - info.prob_grow)

        #--------------------------------------------------
        # Probability of choosing the no-grandchildren node at y
        if(node.parent is None): # no parent, nx is the top and only node
            logp_nog_ny = 0 # prob=1
        else:
            if(node.parent in nognds): # if parent is a nog, number of nogs same at x and y
                logp_nog_ny = -np.log(len(nognds))
            else: # if parent is not a nog, y has one more nog
                logp_nog_ny = -np.log(len(nognds) + 1)

        #--------------------------------------------------
        # Compute MH acceptance probability ratio for global changes
        logp_transition_ratio = logp_prune_ny + logp_nog_ny - logp_nx - np.log(prob_grow)
        logp_struct_ratio = np.log(p_grow_nx) + np.log(1-p_grow_l1) + np.log(1-p_grow_l2) - np.log(1-p_grow_nx)

        #--------------------------------------------------
        new_left = Node(parent=node,left=None,right=None,params=node.params,ancestors=node.ancestors+[node],path=node.path+[0]) # keep same params in left child node
        new_right= Node(parent=node,left=None,right=None,ancestors=node.ancestors+[node],path=node.path+[1])
        node.left = new_left
        node.right = new_right
        node.params = {}
        new_left.node_id = new_left.get_node_id()	
        new_right.node_id = new_right.get_node_id()
        tree.reset_tree()
    else: # prune proposal
        print("\n----- prune proposal -----\n")
        info.prune_prop += 1
        #--------------------------------------------------
        # Draw no grandchildren node, any no grandchildren node is a possibility
        node = np.random.choice(nognds)

        #--------------------------------------------------
        # Compute things needed for metropolis ratio
        depth = node.get_depth()
        p_grow_ny = p_split(tree,depth) # probability that this node grows

        #--------------------------------------------------
        # Prob of grow move at y
        if(node.parent is None): #is the nog node nx the top node
            PBy = 1.0 
        else:
            PBy = info.prob_grow

        #--------------------------------------------------
        # Prob of choosing the nog as bot to split on when y 
        ngood = len(leaves_grow)
        logp_nog_ny= -np.log(ngood-1)

        p_grow_l1 = p_split(tree,node.left.get_depth())
        p_grow_l2 = p_split(tree,node.right.get_depth())

        logp_prune_nx = np.log(1.0 - prob_grow) # prob of a prune step at x
        logp_nog_nx = -np.log(len(nognds))

        #--------------------------------------------------
        # Compute MH acceptance probability ratio for global changes
        logp_transition_ratio = np.log(PBy) + logp_nog_ny - logp_prune_nx - logp_nog_nx
        logp_struct_ratio = np.log(1-p_grow_ny) - np.log(p_grow_ny) - np.log(1-p_grow_l1) - np.log(1-p_grow_l2)

        #--------------------------------------------------
        if(node.get_node_id() != 0): # only propose prune tranisition if not root node
            logp_rv_ratio += dist.Beta(1,1).log_prob(node.params['tau'])
            if(settings.method == 0):
                logp_rv_ratio += dist.Categorical(probs=np.array([1/data['nx']]*data['nx'])).log_prob(node.params['index'])
            else:
                logp_rv_ratio += dist.Dirichlet(np.ones((data['nx'],))).log_prob(node.params['index'])
            if(data['output_type']=='real'):
                logp_rv_ratio += dist.Normal(tree.mu_mean_llh,tree.mu_var_llh).log_prob(node.right.params['mu']) # right child becomes rv
            node.params = node.left.params # take params from left child 
            node.right = None
            node.left = None
            tree.reset_tree()
        else: # no update to tree structure
            print("error: trying to propose prune transition when root node")

    return tree, logp_transition_ratio, logp_struct_ratio, logp_rv_ratio

    
def p_split(tree,depth):
    return tree.prior['alpha']/np.power(1.0+depth,tree.prior['beta'])

def get_prob_grow(tree, prob_grow):
    # compute prob of a grow based on number of internal nodes
    leaves = tree.leaf_nodes
    if (len(tree.internal_nodes) == 1): # just one node
        prob = 1
    else:
        prob = prob_grow

    return leaves, prob # leaf nodes, probability of growing

if __name__ == "__main__":
    main()