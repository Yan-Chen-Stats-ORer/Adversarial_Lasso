import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import itertools
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import pandas as pd
from sklearn.datasets import make_spd_matrix
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
from itertools import combinations
from scipy.optimize import minimize
import cvxpy as cp
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import warnings
import cvxpy as cp
import mosek

from data_gen_uncond import *

# Define a callback function or create an environment
with mosek.Env() as env:
    with env.Task() as task:
        # Set the maximum number of iterations
        task.putintparam(mosek.iparam.intpnt_max_iterations, 5000)  

# Set MOSEK solver parameters

# Suppress the specific CVXPY warning
warnings.filterwarnings("ignore", message="Constraint #.* contains too many subexpressions.*")


def compute_wk_weights(W, r_init):
    """
    Compute w_k and w_k^1 weights.
    """
    p = W.shape[1]
    N_val = W.shape[0]

    r_init = r_init.reshape(p,1)
    r_init = cp.Constant(r_init)

    score_r0 = W/(1+W@r_init)
    W_square = cp.square(score_r0)
    average_W_square = cp.sum(W_square, axis=0)/N_val
    weight = cp.sqrt(average_W_square)

    return weight


def plugin_function_cvxpy(Y, lambda_, W, p, r_init):
    """
    Objective function to minimize using cvxpy.
    """
    N = Y.shape[0]
    r = cp.Variable(p)
    # Reshape a and b for broadcasting
    r = cp.reshape(r, (1, p))  # Shape (1, p)
    
    # Compute inner product and kernel weights
    inner_product = cp.diag(cp.matmul(W, r.T))  # Shape (N,)
    # inner_product = cp.reshape(inner_product,(-1,1))

    # Compute the first term
    log_term = cp.log(1 + inner_product)
    term1 = cp.sum(log_term) / N 
    
    # Compute w_k and w_k1 (assumes this function is already implemented)
    w_k = compute_wk_weights(W, r_init)
    w_k = cp.reshape(w_k,(p,1))
    # Compute the penalty term
    penalty = lambda_ * cp.sum(cp.multiply(cp.abs(r),w_k))
    
    # Define the objective
    objective = term1 - penalty
    
    return objective, r

def optimize_parameters_plugin_cvxpy(Y, lambda_, W, r_init):
    """
    Optimize parameters a and b using cvxpy for the plugin function.
    """
    p = W.shape[1]
    
    objective, r = plugin_function_cvxpy(Y, lambda_, W, p, r_init)

    # Define the problem
    problem = cp.Problem(cp.Maximize(objective))
    
    # Solve the problem
    problem.solve(verbose=False,solver='MOSEK',
                  mosek_params={"MSK_IPAR_INTPNT_MAX_ITERATIONS": 5000}) # MOSEK - commercial solver #tried: ECOS (plugin only); SCS

    return r.value, problem.value

#########################################################################################################
"""
Compute the set of extreme points of the hyper-rectangle
"""

def compute_Y_Xi_sum(Xi,Y,alpha_hat,epsilon=1e-4):
    # Step 1: Normalize Y
    # Calculate the denominator for the normalization
    alpha_terms = np.maximum(np.sqrt(alpha_hat * (1 - alpha_hat)),epsilon)
    normalized_Y_vector = (Y - alpha_hat) / alpha_terms
    
    # Step 2: Multiply each element of normalized_Y_vector by Xi (broadcasted across columns)
    # Xi needs to be reshaped to (N, 1) for broadcasting to work correctly
    Xi = Xi.reshape(-1, 1)  # Make sure Xi is a column vector for broadcasting
    product = Xi * normalized_Y_vector
    
    # Step 3: Sum across rows (i.e., sum for each column) and then find the maximum of the absolute values
    sum_per_column = np.sum(product, axis=0)
    max_abs_sum = np.max(np.abs(sum_per_column))

    return max_abs_sum

def bootstrap_quantile(Y,quantile=0.95,B=500):
    """compute the bootstrapped qunatile of the corresponding term"""
    N_val = Y.shape[0]
    alpha_hat = np.mean(Y,axis=0)
    max_list = [] #store the B realized maximum values in bootstrap samples
    for _ in range(B):
        #generate N i.i.d. N(0,1)
        Xi = np.random.normal(0,1,N_val)
        max_list.append(compute_Y_Xi_sum(Xi,Y,alpha_hat))
    #find the quantile corresponding to the probability given (by default, quantile = 0.95)
    cv = np.quantile(max_list,quantile)
    
    return cv

def generate_uncertainty_marginal(vector_a, vector_b):
    # Ensure vectors are numpy arrays for easier manipulation
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)
    
    # Generate all combinations of indices (0 from vector_a, 1 from vector_b)
    index_combinations = itertools.product([0, 1], repeat=len(vector_a))
    
    # Create the resulting uncertainty marginal list
    uncertainty_marginal = []
    for indices in index_combinations:
        # Generate a new vector based on the current combination of indices
        combination_vector = [vector_a[i] if index == 0 else vector_b[i] for i, index in enumerate(indices)]
        uncertainty_marginal.append(combination_vector)
    
    return uncertainty_marginal


#########################################################################################################
"""
Utility functions for First-order Estimators
"""
def indices_combination(M):
    all_combinations = [comb for k in range(2, M+1) for comb in combinations(range(M), k)]
    return all_combinations

def compute_gradient(Y, alpha, W, all_combinations):
    _, M = Y.shape
    p = W.shape[1]  
    
    # Create an indicator matrix using NumPy
    comb_matrix = np.zeros((M, p), dtype=bool)
    for l, comb in enumerate(all_combinations):
        comb_matrix[list(comb), l] = True

    # Ensure the arrays are standard NumPy arrays and not matrices
    comb_matrix = np.asarray(comb_matrix)
    Y = np.asarray(Y)
    alpha = np.asarray(alpha)
    W = np.asarray(W)

    # Expand dimensions for broadcasting
    comb_matrix_expanded = comb_matrix[np.newaxis, :, :]  # shape (1, M, p)
    Y_expanded = Y[:, :, np.newaxis]  # shape (N, M, 1)
    alpha_expanded = alpha[np.newaxis, :, np.newaxis]  # shape (1, M, 1)
    term1 = 1 - 2 * Y_expanded  # shape (N, M, 1)
    term2 = W[:, np.newaxis, :] / (2 * alpha_expanded * (1 - alpha_expanded))  # shape (N, 1, p)

    # Compute the gradient using broadcasting
    gradient = comb_matrix_expanded * term1 * term2

    return gradient

def grad_W_r_product(grad_W,r):
    # Initialize W_r_prod as an empty list
    W_r_prod_list = []

    M = grad_W.shape[1]
    N = grad_W.shape[0]

    
    # Compute W_r_prod for each k (since grad_W is 3D)
    for k in range(grad_W.shape[1]):
        W_r_prod_k = cp.matmul(grad_W[:, k, :],r)  # Shape (N,)
        W_r_prod_list.append(W_r_prod_k)
    
    # Stack the list to form a matrix W_r_prod
    W_r_prod = cp.vstack(W_r_prod_list)  # Shape (M, N)
    W_r_prod = cp.reshape(W_r_prod,(M,N)) # Shape (M, N)

    return W_r_prod
    
def W_r_product(W,r):
    p = W.shape[1]
    W_r_term = cp.matmul(W,r) #W:Nxp; r:px1
    return W_r_term

def W_FO_score(Y, alpha, alpha_hat, W, all_combinations):
    grad_W = compute_gradient(Y, alpha, W, all_combinations)  # shape (N, M, p)
    delta_alpha = alpha - alpha_hat  # shape (M,)
    # Reshape delta_alpha for broadcasting
    delta_alpha = delta_alpha[np.newaxis, :, np.newaxis]  # shape (1, M, 1)
    # Compute the inner product/multiplication
    delta_alpha_grad_W = np.sum(grad_W * delta_alpha, axis=1)  # shape (N, p)
    # Update W
    W_updated = W + delta_alpha_grad_W  # shape (N, p)
    return W_updated

#define the regularized loss functions with different weights 
"""weighted regularization"""
def FO_loss_unconditional(W,Y,all_combinations,alpha_hat,uncertainty_marginal,regularization,r_init):
    p = W.shape[1]
    N = W.shape[0]
    r = cp.Variable(p)
    r = cp.reshape(r,(p,1))
    weight = compute_wk_weights(W, r_init)
    weight = cp.reshape(weight,(p,1))
    penalty = cp.matmul(cp.abs(r.T),weight)
    alpha_hat = np.array(alpha_hat).reshape(1,-1) #shape = 1xM
    uncertainty_marginal = np.array(uncertainty_marginal) #(2^M,M)
    alpha_diff = uncertainty_marginal - alpha_hat #(2^M,M)
    
    alpha_hat = alpha_hat.reshape(-1,)
    grad_W = compute_gradient(Y, alpha_hat, W, all_combinations)

    grad_W_r_term = grad_W_r_product(grad_W,r) #NxM
    W_r_term = W_r_product(W,r) 
    #select the first-order term with maximum value regarding alpha
    first_order_term = cp.matmul(grad_W_r_term.T,alpha_diff.T) #(N,M) time (M,2^M), shape = (N,2^M)
    W_r_term = cp.reshape(W_r_term,(N,1))
    W_term = 1 + W_r_term + first_order_term
    ones = np.ones((1,N))
    loss = cp.min(cp.matmul(ones,cp.log(W_term)))/N

    objective = loss - regularization*penalty
    return objective, r

def optimize_parameters_FO_cvxpy(Y, lambda_, W, r_init):
    """
    Optimize parameters a and b using cvxpy for the plugin function.
    """
    all_combinations = indices_combination(Y.shape[1])
    alpha_hat = np.mean(Y,axis=0)

    ###generate extreme points
    s_hat = np.sqrt(alpha_hat*(1-alpha_hat)) 
    cv = bootstrap_quantile(Y)
    interval_low = alpha_hat - cv*s_hat
    interval_high = alpha_hat + cv*s_hat
    uncertainty_marginal = generate_uncertainty_marginal(interval_low, interval_high)                                                      
    #####

    objective, r = FO_loss_unconditional(W,Y,all_combinations,alpha_hat,uncertainty_marginal,lambda_,r_init)

    # Define the problem
    problem = cp.Problem(cp.Maximize(objective))
    
    # Solve the problem
    problem.solve(verbose=True,solver='MOSEK',
                  mosek_params={"MSK_IPAR_INTPNT_MAX_ITERATIONS": 5000}) # MOSEK - commercial solver #tried: ECOS (plugin only); SCS

    return r.value, problem.value
#########################################################################################################
