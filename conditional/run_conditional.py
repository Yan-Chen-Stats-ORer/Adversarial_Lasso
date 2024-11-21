import numpy as np
import matplotlib.pyplot as plt
import random 
import argparse
from scipy.stats import norm

from data_gen import * 
from functions import *
from utils import *

#set parameters 
parser = argparse.ArgumentParser(description="Localized Lasso Conditional Case Illustration")
parser.add_argument("--N",default=100,help="sample size")
parser.add_argument("--M",default=4,help="binary vector dimension")
parser.add_argument("--s",default=2,help="cardinality of the support of r_0")
# Example specific value of x, at which we are interested in estimating r_0(x)
parser.add_argument("--x",default=0.5,help="covariate of interest for estimating r_0(x)")
args = parser.parse_args()

N = int(args.N)
M = int(args.M) 
s = int(args.s)
x = float(args.x)  
p = 2**M - M - 1 #dimension of generalized correlation coefficient 
h = (M/N)**(0.2)# Bandwidth
print('bandwidth',h)

"""generate the marginal distributions"""
theta = np.random.uniform(low=0.0,high=2.0,size=M)
alpha_x = compute_alpha(x,theta) 
"""define r_0(x)"""
rho, support_r0 = draw_rho_at_x(s,M,x)
r_0_specific, comb_list_specific = assign_r0_general_at_x(M, support_r0, rho)
print("r_0 at x =", x, ":", r_0_specific)
"""compute probabilities for all possible M-dimensional binary vectors"""
_, probabilities_x = precompute_probabilities_x(M, alpha_x, r_0_specific, comb_list_specific)
print('probabilities_x',probabilities_x)

var_init = [0.0] * (2 * p)
lambdas_FO = np.linspace(0.1,5.0,10)
lambdas_PI = np.linspace(0.1,5.0,10)

error_dict, prob_error_dict = initialize_storage()

for i in range(1):
    random.seed(i)
    #############################################################
    """generate samples"""
    X = sample_X(N)
    rho = draw_rho_general(X,support_r0)
    #generate marginal at all X 
    alpha = compute_alpha_vectorized(X, theta) 
    #generate correlation coefficient at all X 
    r_0_all, comb_list = assign_r0_general_vectorized(M, X , support_r0) 
    #compute probabilities for all bundles at each observation 
    all_vectors, all_probabilities = precompute_probabilities_vectorized(M, alpha, r_0_all, comb_list)
    
    #ensure the probabilities are non-negative 
    if all_probabilities.any() < 0:
        continue

    #############################################################

    r_0 = np.array(r_0_all).reshape(N, -1)
    alpha = np.array(alpha).reshape(N, -1)
    X = X.reshape(N, 1)
    samples = sample_vectors_vectorized(N, all_vectors, all_probabilities)
    Y = np.array(samples)
    W_oracle, _ = compute_W_matrix(Y, alpha) #W_orcale with the true marginals 

    alpha_tilde, _ = local_linear_regression(X, Y, x, h)
    W, _ = compute_W_matrix(Y, alpha_tilde)#compute W matrix with estimated alpha from local linear regression 
    
    ##########################################################################################################################
    
    """Result for NW estimator"""
    NW_estimator = nadaraya_watson(X, h, x, W)
    error_dict['NW'].append(L2_estimation_error(NW_estimator, r_0_specific))
    _, probabilities_NW = precompute_probabilities_x(M, alpha_tilde, NW_estimator, comb_list_specific)
    prob_error_dict['expected']['NW'].append(np.dot(np.abs(probabilities_NW - probabilities_x), probabilities_x))
    prob_error_dict['max']['NW'].append(np.max(np.abs(probabilities_NW - probabilities_x)))
    
    ##########################################################################################################################

    PI_k_fold_scores_0, PI_k_fold_scores_r0, \
    FO_k_fold_scores_0, FO_k_fold_scores_r0,\
    PI_alpha0_k_fold_scores_0, PI_alpha0_k_fold_scores_r0 = cross_validation_folds(X, Y, lambdas_PI, lambdas_FO, r_0, p, h, x, r_0_specific, W_oracle)
    
    # Plug-in, First-Order, Plug-in(oracle) estimators with cross-validation
    PI_best_lam_0, PI_prediction_0 = optimize_and_store_results(X, Y, W, PI_k_fold_scores_0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                p, h, x, [0.0]*np.zeros((N, p)), error_dict['PI_0'])
    PI_best_lam_r0, PI_prediction_r0 = optimize_and_store_results(X, Y, W, PI_k_fold_scores_r0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                  p, h, x, r_0, error_dict['PI_r0'])
    FO_best_lam_0, FO_prediction_0 = optimize_and_store_results(X, Y, W, FO_k_fold_scores_0, lambdas_FO, optimize_parameters_FO_cvxpy, 
                                                                p, h, x, [0.0]*np.zeros((N, p)), error_dict['FO_0'])
    FO_best_lam_r0, FO_prediction_r0 = optimize_and_store_results(X, Y, W, FO_k_fold_scores_r0, lambdas_FO, optimize_parameters_FO_cvxpy, 
                                                                  p, h, x, r_0, error_dict['FO_r0'])
    PI_alpha0_best_lam_0, PI_alpha0_prediction_0 = optimize_and_store_results(X, Y, W, PI_alpha0_k_fold_scores_0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                                p, h, x, [0.0]*np.zeros((N, p)), error_dict['PI_alpha0_0'])
    PI_alpha0_best_lam_r0, PI_alpha0_prediction_r0 = optimize_and_store_results(X, Y, W, PI_alpha0_k_fold_scores_r0, lambdas_PI, optimize_parameters_plugin_cvxpy, 
                                                                                p, h, x, r_0, error_dict['PI_alpha0_r0'])
    # Plug-in, First-Order, Plug-in(oracle) estimators with theoretical Lambda  
    PI_lambda_star_0_theory = compute_theoretical_lambda_plugin(W_oracle, [0.0]*np.zeros((N, p)), p, M, N, h,X,x)
    PI_lambda_star_r0_theory = compute_theoretical_lambda_plugin(W_oracle, r_0, p, M, N, h,X,x)
    FO_lambda_star_0_theory = compute_theoretical_lambda_FO(W_oracle, [0.0]*np.zeros((N, p)), p, M, N, h,X,x)
    FO_lambda_star_r0_theory = compute_theoretical_lambda_FO(W_oracle, r_0, p, M, N, h,X,x)
    PI_alpha0_lambda_star_0_theory = compute_theoretical_lambda_plugin_oracle(W_oracle, [0.0]*np.zeros((N, p)), p, M, N, h,X,x)
    PI_alpha0_lambda_star_r0_theory = compute_theoretical_lambda_plugin_oracle(W_oracle, r_0, p, M, N, h,X,x)
    
    PI_prediction_0_theory, _, _ = optimize_parameters_plugin_cvxpy(X, Y, x, h, PI_lambda_star_0_theory, W, p, [0.0]*np.zeros((N, p)))
    error_dict['PI_0_theory'].append(L2_estimation_error(r_0_specific, PI_prediction_0_theory))
    PI_prediction_r0_theory, _, _ = optimize_parameters_plugin_cvxpy(X, Y, x, h, PI_lambda_star_r0_theory, W, p, r_0)
    error_dict['PI_r0_theory'].append(L2_estimation_error(r_0_specific, PI_prediction_r0_theory))
  
    FO_prediction_0_theory, _, _ = optimize_parameters_FO_cvxpy(X, Y, x, h, FO_lambda_star_0_theory, W, p, [0.0]*np.zeros((N, p)))
    error_dict['FO_0_theory'].append(L2_estimation_error(r_0_specific, FO_prediction_0_theory))
    FO_prediction_r0_theory, _, _ = optimize_parameters_FO_cvxpy(X, Y, x, h, FO_lambda_star_r0_theory, W, p, r_0)
    error_dict['FO_r0_theory'].append(L2_estimation_error(r_0_specific, FO_prediction_r0_theory))

    PI_alpha0_prediction_0_theory, _, _ = optimize_parameters_plugin_cvxpy(X, Y, x, h, PI_alpha0_lambda_star_0_theory, W, p, [0.0]*np.zeros((N, p)))
    error_dict['PI_alpha0_0_theory'].append(L2_estimation_error(r_0_specific, PI_alpha0_prediction_0_theory))
    PI_alpha0_prediction_r0_theory, _, _ = optimize_parameters_plugin_cvxpy(X, Y, x, h, PI_alpha0_lambda_star_r0_theory, W, p, r_0)
    error_dict['PI_alpha0_r0_theory'].append(L2_estimation_error(r_0_specific, PI_alpha0_prediction_r0_theory))
  

    # Store probability errors for each method
    methods_non_oracle = ['PI_0', 'PI_r0', 'FO_0', 'FO_r0',
               'PI_0_theory', 'PI_r0_theory', 'FO_0_theory', 'FO_r0_theory']
    methods_oracle = ['PI_alpha0_0','PI_alpha0_r0','PI_alpha0_0_theory','PI_alpha0_r0_theory']
    predictions = [PI_prediction_0, PI_prediction_r0, FO_prediction_0, FO_prediction_r0,
                   PI_prediction_0_theory, PI_prediction_r0_theory, FO_prediction_0_theory, FO_prediction_r0_theory]
    
    predictions_alpha0 = [PI_alpha0_prediction_0, PI_alpha0_prediction_r0,PI_alpha0_prediction_0_theory, PI_alpha0_prediction_r0_theory]
    
    for method, prediction in zip(methods_non_oracle, predictions):
        _, probabilities = precompute_probabilities_x(M, alpha_tilde, prediction, comb_list_specific)
        prob_error_dict['expected'][method].append(np.dot(np.abs(probabilities - probabilities_x), probabilities_x))
        prob_error_dict['max'][method].append(np.max(np.abs(probabilities - probabilities_x)))

    for method, prediction in zip(methods_oracle, predictions_alpha0):
        _, probabilities = precompute_probabilities_x(M, alpha_x, prediction, comb_list_specific)
        prob_error_dict['expected'][method].append(np.dot(np.abs(probabilities - probabilities_x), probabilities_x))
        prob_error_dict['max'][method].append(np.max(np.abs(probabilities - probabilities_x)))

# Output results (use error_dict and prob_error_dict for organized result retrieval)
print_final_results(error_dict, prob_error_dict)



