import numpy as np
import matplotlib.pyplot as plt
import random 
import argparse
from scipy.stats import norm
import time

from data_gen_uncond import * 
from functions_uncond import *
from utils_uncond import *

"""
Estimate r_0 (unconditional case)
"""

#set parameters 
parser = argparse.ArgumentParser(description="Localized Lasso Conditional Case Illustration")
parser.add_argument("--N",default=500,help="sample size")
parser.add_argument("--M",default=4,help="binary vector dimension")
parser.add_argument("--s",default=2,help="cardinality of the support of r_0")
args = parser.parse_args()

N = int(args.N)
M = int(args.M) 
s = int(args.s)
p = 2**M - M - 1 #dimension of generalized correlation coefficient 

"""randomly generate marginal distributions in [0.2,0.8]"""
alpha = np.random.uniform(0.1,0.9,size=M)
alpha = np.array(alpha)
alpha = alpha.reshape(-1,)
print('marginals', alpha) 

# set r_0 and generate probabilities.
r_0, probabilities = r0_setup(s,M,alpha)

_, comb_list = assign_r0_general_vectorized(s,M)
all_vectors = list(product([0, 1], repeat=M))

lambdas_FO = np.linspace(0.01,1.0,10)
lambdas_PI = np.linspace(0.01,1.0,10)


error_dict, prob_error_dict = initialize_storage()

for i in range(50):
    random.seed(i)
    #############################################################
    """generate samples"""
    # Sample N i.i.d. binary vectors
    samples = sample_vectors(N, all_vectors, probabilities)
    y_samples = np.array(samples)
    Y = y_samples
    #############################################################
    r_0 = np.array(r_0)
    r_0 = r_0.reshape(1,-1)
    
    """sample average to estimate the marginal distributions"""
    alpha_hat = np.mean(y_samples,axis=0)

    #compute W with estimated alpha and W with true alpha
    z = Z_vectors(y_samples,alpha_hat)
    W, W_idx = W_func(M,z)

    #use true alpha
    z_oracle = Z_vectors(y_samples,alpha)
    W_oracle, _ = W_func(M,z_oracle)

    ##########################################################################################################################
    """Result for SAA estimator"""
    SAA_naive = np.mean(W,axis=0)
    r_SAA_naive = SAA_naive.reshape(1,-1)
    error_dict['SAA'].append(L2_estimation_error(r_SAA_naive, r_0))
    _, probabilities_SAA = precompute_probabilities(M, alpha_hat, r_SAA_naive, comb_list)  
    prob_error_dict['expected']['SAA'].append(np.dot(np.abs(probabilities_SAA - probabilities), probabilities))
    prob_error_dict['max']['SAA'].append(np.max(np.abs(probabilities_SAA - probabilities)))
    
    ##########################################################################################################################

    PI_k_fold_scores_0, PI_k_fold_scores_r0, \
    FO_k_fold_scores_0, FO_k_fold_scores_r0,\
    PI_alpha0_k_fold_scores_0, PI_alpha0_k_fold_scores_r0 = cross_validation_folds(Y, lambdas_PI, lambdas_FO, r_0, p, W_oracle)
    
    # Plug-in, First-Order, Plug-in(oracle) estimators with cross-validation
    PI_best_lam_0, PI_prediction_0 = optimize_and_store_results(Y, W, PI_k_fold_scores_0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                [0.0]*np.zeros((p,1)), error_dict['PI_0'])
    PI_best_lam_r0, PI_prediction_r0 = optimize_and_store_results(Y, W, PI_k_fold_scores_r0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                  r_0, error_dict['PI_r0'])
    FO_best_lam_0, FO_prediction_0 = optimize_and_store_results(Y, W, FO_k_fold_scores_0, lambdas_FO, optimize_parameters_FO_cvxpy, 
                                                                [0.0]*np.zeros((p,1)), error_dict['FO_0'])
    FO_best_lam_r0, FO_prediction_r0 = optimize_and_store_results(Y, W, FO_k_fold_scores_r0, lambdas_FO, optimize_parameters_FO_cvxpy, 
                                                                  r_0, error_dict['FO_r0'])
    PI_alpha0_best_lam_0, PI_alpha0_prediction_0 = optimize_and_store_results(Y, W, PI_alpha0_k_fold_scores_0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                              [0.0]*np.zeros((p,1)), error_dict['PI_alpha0_0'])
    PI_alpha0_best_lam_r0, PI_alpha0_prediction_r0 = optimize_and_store_results(Y, W, PI_alpha0_k_fold_scores_r0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                                r_0, error_dict['PI_alpha0_r0'])
    # Plug-in, First-Order, Plug-in(oracle) estimators with theoretical Lambda  
    PI_lambda_star_0_theory = norm.ppf(1-0.05/(2*p))/np.sqrt(N) + M/np.sqrt(N)
    PI_lambda_star_r0_theory = norm.ppf(1-0.05/(2*p))/np.sqrt(N) + M/np.sqrt(N)
    FO_lambda_star_0_theory = norm.ppf(1-0.05/(2*p))/np.sqrt(N) + M/N
    FO_lambda_star_r0_theory = norm.ppf(1-0.05/(2*p))/np.sqrt(N) + M/N
    PI_alpha0_lambda_star_0_theory = norm.ppf(1-0.05/(2*p))/np.sqrt(N)
    PI_alpha0_lambda_star_r0_theory = norm.ppf(1-0.05/(2*p))/np.sqrt(N)
    
    PI_prediction_0_theory, _ = optimize_parameters_plugin_cvxpy(Y, PI_lambda_star_0_theory, W, [0.0]*np.zeros((p,1)))
    error_dict['PI_0_theory'].append(L2_estimation_error(r_0, PI_prediction_0_theory))
    PI_prediction_r0_theory, _= optimize_parameters_plugin_cvxpy(Y, PI_lambda_star_r0_theory, W, r_0)
    error_dict['PI_r0_theory'].append(L2_estimation_error(r_0, PI_prediction_r0_theory))
  
    FO_prediction_0_theory, _ = optimize_parameters_FO_cvxpy(Y, FO_lambda_star_0_theory, W, [0.0]*np.zeros((p,1)))
    FO_prediction_0_theory = FO_prediction_0_theory.reshape(1,-1)
    error_dict['FO_0_theory'].append(L2_estimation_error(r_0, FO_prediction_0_theory))
    
    FO_prediction_r0_theory, _ = optimize_parameters_FO_cvxpy(Y, FO_lambda_star_r0_theory, W, r_0)
    FO_prediction_r0_theory = FO_prediction_r0_theory.reshape(1,-1)
    error_dict['FO_r0_theory'].append(L2_estimation_error(r_0, FO_prediction_r0_theory))

    PI_alpha0_prediction_0_theory, _ = optimize_parameters_plugin_cvxpy(Y, PI_alpha0_lambda_star_0_theory, W, [0.0]*np.zeros((p,1)))
    error_dict['PI_alpha0_0_theory'].append(L2_estimation_error(r_0, PI_alpha0_prediction_0_theory))
    PI_alpha0_prediction_r0_theory, _ = optimize_parameters_plugin_cvxpy(Y, PI_alpha0_lambda_star_r0_theory, W, r_0)
    error_dict['PI_alpha0_r0_theory'].append(L2_estimation_error(r_0, PI_alpha0_prediction_r0_theory))
  

    # Store probability errors for each method
    methods_non_oracle = ['PI_0', 'PI_r0', 'FO_0', 'FO_r0',
               'PI_0_theory', 'PI_r0_theory', 'FO_0_theory', 'FO_r0_theory']
    methods_oracle = ['PI_alpha0_0','PI_alpha0_r0','PI_alpha0_0_theory','PI_alpha0_r0_theory']
    predictions = [PI_prediction_0, PI_prediction_r0, FO_prediction_0, FO_prediction_r0,
                   PI_prediction_0_theory, PI_prediction_r0_theory, FO_prediction_0_theory, FO_prediction_r0_theory]
    
    predictions_alpha0 = [PI_alpha0_prediction_0, PI_alpha0_prediction_r0,PI_alpha0_prediction_0_theory, PI_alpha0_prediction_r0_theory]
    
    for method, prediction in zip(methods_non_oracle, predictions):
        _, probabilities_pred = precompute_probabilities(M, alpha_hat, prediction, comb_list)
        prob_error_dict['expected'][method].append(np.dot(np.abs(probabilities_pred - probabilities), probabilities))
        prob_error_dict['max'][method].append(np.max(np.abs(probabilities_pred - probabilities)))

    for method, prediction in zip(methods_oracle, predictions_alpha0):
        _, probabilities_pred = precompute_probabilities(M, alpha, prediction, comb_list)
        prob_error_dict['expected'][method].append(np.dot(np.abs(probabilities_pred - probabilities), probabilities))
        prob_error_dict['max'][method].append(np.max(np.abs(probabilities_pred - probabilities)))

# Output results (use error_dict and prob_error_dict for organized result retrieval)
print_final_results(error_dict, prob_error_dict)


