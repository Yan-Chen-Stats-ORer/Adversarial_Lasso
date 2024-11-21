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
import coptpy

# Set MOSEK solver parameters

# Suppress the specific CVXPY warning
warnings.filterwarnings("ignore", message="Constraint #.* contains too many subexpressions.*")


"""
In the following, the covariates are all scalars. 
(set d=1 for the notation in the paper)
"""

"""
# function: 
# uniform kernel with bandwidth h, sample X=(X_i)_{i\in[N]}, given parameter x, 
# compute the uniform kernel, 
# return: 
# a vector of N dimension K_h(X_i-x)=1/h*K(|X_i-x|/h)
# a N-by-N diagnal matrix W_x with W_x[i,i]=K_h(X_i-x), i.e. i-th entry of the vector computed above
"""
def uniform_kernel_bandwidth_h(X, x, h):
    """
    Generate uniform kernel with bandwidth h.
    
    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    x : float
        Given parameter.
    h : float
        Bandwidth.
    
    Returns:
    K_h_x : array-like, shape (N,)
        Vector of uniform kernel values.
    W_x : array-like, shape (N, N)
        N-by-N diagonal matrix with uniform kernel values on the diagonal.
    """
    N = len(X)
    
    # Compute the uniform kernel values
    K_h_x = np.where(np.abs(X - x) <= h, 1.0 / h, 0)
    
    # Create the diagonal matrix W_x
    W_x = np.diag(K_h_x)
    
    return K_h_x, W_x

def local_linear_regression(X, Y, x, h,epsilon=1e-3):
    """
    Perform local linear regression using uniform kernel.

    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    x : float
        Given parameter.
    h : float
        Bandwidth.

    Returns:
    alpha_tilde : array-like, shape (M,)
        M-dimensional vector of \tilde{\alpha}(x).
    beta_tilde : array-like, shape (M,)
        M-dimensional vector of \tilde{\beta}(x).
    """
    N = len(X)
    M = Y.shape[1]
    X = X.reshape(-1,)
    # Compute the uniform kernel values and diagonal matrix
    K_h_x, _ = uniform_kernel_bandwidth_h(X, x, h)
    # Construct the design matrix Z
    Z = np.vstack((np.ones(N), X - x)).T  # Shape (N, 2)
    
    alpha_tilde = np.zeros(M)
    beta_tilde = np.zeros(M)

    # Weight matrix for dimension j
    W_x_j = np.diag(K_h_x)
    
    for j in range(M):
        # Response vector for dimension j
        Y_j = Y[:, j]
        
        # Perform weighted least squares
        Z_W = Z.T @ W_x_j @ Z
        Z_W_inv = np.linalg.inv(Z_W)
        Z_W_Y = Z.T @ W_x_j @ Y_j
        
        theta = Z_W_inv @ Z_W_Y  # Shape (2,)
        
        alpha_tilde[j] = theta[0]
        beta_tilde[j] = theta[1]
    
    #trucate alpha_tilde to make it strictly between (0,1)
    alpha_tilde = np.maximum(alpha_tilde,epsilon)
    alpha_tilde = np.minimum(alpha_tilde,1-epsilon)
    
    return alpha_tilde, beta_tilde


def compute_s_hat(X, x, h):
    """
    Compute the s_hat matrix for local linear regression using uniform kernel.

    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    x : float
        Given parameter.
    h : float
        Bandwidth.

    Returns:
    s_hat : array-like, shape (N, 2)
        N x 2 matrix such that s_hat[i,:] = Z_i K_h(X_i-x) (Z.T W_x Z)^-1.
    """
    N = len(X)
    
    # Compute the uniform kernel values and diagonal matrix
    K_h_x, _ = uniform_kernel_bandwidth_h(X, x, h)
    K_h_x = K_h_x.reshape(-1,)
    # print('K_h_x',K_h_x.shape)
    X = X.reshape(-1,)
    # Construct the design matrix Z
    Z = np.vstack((np.ones(N), X - x)).T  # Shape (N, 2)
    W_x = np.diag(K_h_x)
    # Compute the weighted design matrix
    W_x_Z = W_x @ Z  
    # Compute (Z.T W_x Z)^-1
    Z_W_x_Z_inv = np.linalg.inv(Z.T @ W_x_Z)

    # Initialize s_hat matrix
    s_hat = np.zeros((N, 2))
    
    for i in range(N):
        Z_i = Z[i, :]  # 1x2 vector
        s_hat[i, :] = Z_i * K_h_x[i] @ Z_W_x_Z_inv
    
    return s_hat

def compute_epsilon(X, Y, x, h):
    """
    Compute the epsilon matrix.

    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    x : float
        Given parameter.
    h : float
        Bandwidth.

    Returns:
    epsilon : array-like, shape (N, M)
        N x M matrix such that epsilon_{ij} = Y_{ij} - \tilde{\alpha}_j(x) - \tilde{\beta}_j (X_i - x).
    """
    # Perform local linear regression to get alpha_tilde and beta_tilde
    alpha_tilde, beta_tilde = local_linear_regression(X, Y, x, h)
    
    # Compute epsilon
    N, M = Y.shape
    epsilon = np.zeros((N, M))
    X = X.reshape(-1,)

    for j in range(M):
        epsilon[:, j] = Y[:, j] - alpha_tilde[j] - beta_tilde[j] * (X - x)
    
    return epsilon

def compute_s_hat_total(epsilon, s_hat):
    """
    Compute the s_hat_total matrix.

    Parameters:
    epsilon : array-like, shape (N, M)
        Epsilon matrix.
    s_hat : array-like, shape (N, 2)
        S_hat matrix.

    Returns:
    s_hat_total : array-like, shape (M, 2)
        Matrix such that s_hat_total[j, l] = sqrt(1/N sum_{i=1}^N epsilon_{ij}^2 s_hat[i, l]^2).
    """
    N, M = epsilon.shape
    s_hat_total = np.zeros((M, 2))
    for j in range(M):
        for l in range(2):
            s_hat_total[j, l] = np.sqrt(np.mean((epsilon[:, j]**2) * (s_hat[:, l]**2)))
    return s_hat_total

def compute_cv(X, Y, x, h, delta, num_bootstrap=1000):
    """
    Compute the cv value using bootstrap.

    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    x : float
        Given parameter.
    h : float
        Bandwidth.
    delta : float
        Confidence level.
    num_bootstrap : int
        Number of bootstrap samples.

    Returns:
    cv : float
        (1 - delta)-quantile of the bootstrap samples.
    """
    N = len(X)
    M = Y.shape[1]

    # Compute epsilon and s_hat
    epsilon = compute_epsilon(X, Y, x, h)
    s_hat = compute_s_hat(X, x, h)
    
    # Compute s_hat_total
    s_hat_total = compute_s_hat_total(epsilon, s_hat)

    # Bootstrap samples
    bootstrap_samples = np.zeros(num_bootstrap)
    for b in range(num_bootstrap):
        xi = np.random.normal(0, 1, N)
        max_stat = np.max([
            np.abs(np.mean(xi * epsilon[:, j] * s_hat[:, l] / s_hat_total[j, l]))
            for j in range(M) for l in range(2)
        ])
        bootstrap_samples[b] = max_stat

    # Compute the (1 - delta)-quantile
    cv = np.quantile(bootstrap_samples, 1 - delta)
    return cv

def compute_combinations(M):
    """
    Generate all combinations of indices {0, 1, ..., M-1} with cardinality >= 2.
    
    Parameters:
    M : int
        Number of elements.
    
    Returns:
    combinations : list of tuples
        List containing all valid combinations.
    """
    combinations = []
    for r in range(2, M + 1):
        combinations.extend(itertools.combinations(range(M), r))
    return combinations

def compute_W_matrix(Y, alpha_tilde):
    """
    Compute the W matrix.
    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    alpha_tilde : array-like, shape (M,)
        M-dimensional vector of \tilde{\alpha}(x).
    Returns:
    W : array-like, shape (N, p)
        The W matrix.
    """
    N, M = Y.shape
    combinations = compute_combinations(M)
    p = len(combinations)

    # Initialize W matrix
    W = np.zeros((N, p))

    # Compute the necessary terms using broadcasting
    alpha_tilde = np.array(alpha_tilde)
    sqrt_terms = np.sqrt(alpha_tilde * (1 - alpha_tilde))  # Shape (M,)
    normalized_Y_diff = (Y - alpha_tilde) / sqrt_terms  # Shape (N, M)

    # Compute the product for each combination using broadcasting
    for l, combo in enumerate(combinations):
        W[:, l] = np.prod(normalized_Y_diff[:, combo], axis=1)

    return W, combinations

def compute_grad_W(Y, alpha_tilde, W, combinations):
    """
    Compute the gradient of W.
    
    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    alpha_tilde : array-like, shape (M,)
        M-dimensional vector of \tilde{\alpha}(x).
    W : array-like, shape (N, p)
        The W matrix.
    combinations : list of tuples
        List containing all valid combinations.
    
    Returns:
    grad_W : array-like, shape (N, M, p)
        The gradient of W matrix.
    """
    N, M = Y.shape
    p = len(combinations)

    # Initialize grad_W matrix
    grad_W = np.zeros((N, M, p))

    # Compute indicator matrix using broadcasting
    indicator_matrix = np.zeros((M, p), dtype=int)
    for l, combo in enumerate(combinations):
        indicator_matrix[list(combo), l] = 1

    # Compute (1 - 2Y) using broadcasting
    one_minus_2Y = 1 - 2 * Y[:, :, None]  # Shape (N, M, 1)

    # Compute (2 * alpha_j * (1 - alpha_j)) using broadcasting
    alpha_tilde = np.array(alpha_tilde)
    denom = 2 * alpha_tilde * (1 - alpha_tilde)  # Shape (M,)

    # Compute the gradient using broadcasting
    grad_W = indicator_matrix[None, :, :] * one_minus_2Y * W[:, None, :] / denom[:, None]

    return grad_W

def compute_grad_W_all_alpha(Y, alpha_tilde, W, combinations):
    """
    Compute the gradient of W.
    
    Parameters:
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    alpha_tilde : array-like, shape (N, M)
        N x M-dimensional matrix of \tilde{\alpha}(x).
    W : array-like, shape (N, p)
        The W matrix.
    combinations : list of tuples
        List containing all valid combinations.
    
    Returns:
    grad_W : array-like, shape (N, M, p)
        The gradient of W matrix.
    """
    N, M = Y.shape
    p = len(combinations)

    # Initialize grad_W matrix
    grad_W = np.zeros((N, M, p))

    # Compute indicator matrix using broadcasting
    indicator_matrix = np.zeros((M, p), dtype=int)
    for l, combo in enumerate(combinations):
        indicator_matrix[list(combo), l] = 1

    # Compute (1 - 2Y) using broadcasting
    one_minus_2Y = 1 - 2 * Y[:, :, None]  # Shape (N, M, 1)

    # Iterate over each sample
    for i in range(N):
        # Compute (2 * alpha_tilde_i * (1 - alpha_tilde_i)) for each sample i
        denom = 2 * alpha_tilde[i, :] * (1 - alpha_tilde[i, :])  # Shape (M,)

        # Compute the gradient for sample i
        grad_W[i, :, :] = indicator_matrix * one_minus_2Y[i, :, :] * W[i, None, :] / denom[:, None]

    return grad_W


def compute_uncertainty_set(X, Y, x, h, delta, num_bootstrap=1000):
    """
    Compute the uncertainty set for alpha_hat.

    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    x : float
        Given parameter.
    h : float
        Bandwidth.
    delta : float
        Confidence level.
    num_bootstrap : int
        Number of bootstrap samples.

    Returns:
    uncertainty_set : array-like, shape (2^M, N, M)
        Array containing all possible alpha_hat vectors.
    """
    N = len(X)
    M = Y.shape[1]

    # Perform local linear regression to get alpha_tilde and beta_tilde
    alpha_tilde, beta_tilde = local_linear_regression(X, Y, x, h)

    # Compute epsilon and s_hat
    epsilon = compute_epsilon(X, Y, x, h)
    s_hat = compute_s_hat(X, x, h)
   
    # Compute s_hat_total
    s_hat_total = compute_s_hat_total(epsilon, s_hat)

    # Compute cv using bootstrap
    cv = compute_cv(X, Y, x, h, delta, num_bootstrap)
    

    # Generate all combinations of alpha_j and beta_j within their bounds
    alpha_j_bounds = np.array([
        [alpha_tilde[j] - cv * s_hat_total[j, 0], alpha_tilde[j] + cv * s_hat_total[j, 0]]
        for j in range(M)
    ])

    beta_j_bounds = np.array([
        [beta_tilde[j] - cv * s_hat_total[j, 1], beta_tilde[j] + cv * s_hat_total[j, 1]]
        for j in range(M)
    ])

    # Generate the uncertainty set
    uncertainty_set = np.zeros((4**M, N, M))

    count = 0

    """maybe able to parallel (4**M, N, M)"""
    for alpha_combination in itertools.product(*alpha_j_bounds):
        for beta_combination in itertools.product(*beta_j_bounds):
            alpha_combination = np.array(alpha_combination)
            beta_combination = np.array(beta_combination)

            # print('alpha_combination',alpha_combination.shape)
            # print('beta_combination',beta_combination.shape)

            alpha_hat = alpha_combination + beta_combination * (X[:] - x)
            
            # print('alpha_hat',alpha_hat.shape)
            uncertainty_set[count,:,:] = alpha_hat 
            count += 1
            if count == 4**M:
                break
        if count == 4**M:
            break

    return uncertainty_set

def compute_max_alpha_W_r_prod(X, Y, x, h, r, delta=0.05):
    """
    Compute the minimum value of alpha_W_r_prod_i over the uncertainty set.
    
    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    x : float
        Given parameter.
    h : float
        Bandwidth.
    delta : float
        Confidence level.
    r : array-like, shape (N, p)
        N x p dimensional matrix r.
    
    Returns:
    min_alpha_W_r_prod : float
        Minimum value of alpha_W_r_prod_i.
    """
    N, M = Y.shape
    p = 2**M-M-1

    # Compute the uncertainty set
    uncertainty_set = compute_uncertainty_set(X, Y, x, h, delta)

    # Compute alpha_tilde for each X_i
    alpha_tilde_X = np.zeros((N, M))
    for i in range(N):
        alpha_tilde_X[i], _ = local_linear_regression(X, Y, X[i], h)
        # Compute W matrix
    W, combinations = compute_W_matrix(Y, alpha_tilde_X) 
    # Compute grad_W matrix
    grad_W = compute_grad_W_all_alpha(Y, alpha_tilde_X, W, combinations)  

    # Compute W_r_prod using broadcasting
    W_r_prod = np.einsum('ij,ikj->ik', r, grad_W)
    
    # Compute alpha_diff using broadcasting
    alpha_diff = uncertainty_set[:, :, :] - alpha_tilde_X[None, :, :]  # Shape (4^M, N, M)

    # Compute alpha_W_r_prod using broadcasting
    alpha_W_r_prod_sum = np.einsum('tij,ij->ti', alpha_diff, W_r_prod)  # Shape (4^M, N, p)
    
    return alpha_W_r_prod_sum  #shape = (4^M,N)


def compute_kernel_weights(X, x, h):
    """
    Compute the uniform kernel weights.
    """
    return np.where(np.abs(X - x) <= h, 1.0 / h, 0)

def compute_wk_weights(X, x, h, W, r, p):
    """
    Compute w_k and w_k^1 weights.
    """
    N = len(X)
    K_h = compute_kernel_weights(X, x, h)
    inner_product = np.einsum('ik,ik->i', W, r)  # Shape (N,)
    inner_product = inner_product.reshape(-1,1)
    # inner_product = np.sum(W * r, axis=1, keepdims=True)  # Shape will be (N, 1)
    
    w_k = np.sqrt(np.mean((K_h.T)@ ((W**2) / (1 + inner_product)**2), axis=0))  # Shape (p,1)
    
    w_k1 = np.sqrt(np.mean((K_h.T)@ ((W**2) * ((X - x)**2)/(h**2)/((1 + inner_product)**2)), axis=0))  # Shape (p,1)
    # w_k1 = np.sqrt(np.mean((K_h.T)@ ((W**2) * ((X - x)**2)/((1 + inner_product)**2)), axis=0))  # Shape (p,1)

    return w_k, w_k1


def plugin_function_cvxpy(X, Y, x, h, lambda_, W, p, r):
    """
    Objective function to minimize using cvxpy.
    """
    a = cp.Variable(p)
    b = cp.Variable(p)

    N = X.shape[0]

    # Reshape a and b for broadcasting
    a_broadcast = cp.reshape(a, (1, p))  # Shape (1, p)
    # Reshape b for broadcasting
    b_broadcast = cp.reshape(b, (1, p))  # Shape (1, p)
        
    # Compute r
    r_pred = a_broadcast + cp.matmul((X - x),b_broadcast) 
    
    # Compute inner product and kernel weights
    inner_product = cp.diag(cp.matmul(W, r_pred.T))  # Shape (N,)
    inner_product = cp.reshape(inner_product,(N,1))
    K_h = cp.reshape(compute_kernel_weights(X, x, h), (N,1))  # Flatten K_h to shape (N,)

    # Compute the first term
    log_term = cp.log(1 + inner_product)
    term1 = cp.sum(cp.multiply(K_h.T,log_term))
    
    # Compute w_k and w_k1 (assumes this function is already implemented)
    w_k, w_k1 = compute_wk_weights(X, x, h, W, r, p)
    
    # Compute the penalty term
    penalty = lambda_ * (cp.sum(cp.multiply(w_k,cp.abs(a.T))) + h * cp.sum(cp.multiply(w_k1,cp.abs(b.T))))
    
    # Define the objective
    objective = term1 - penalty
    
    return objective, a, b

def optimize_parameters_plugin_cvxpy(X, Y, x, h, lambda_, W, p, r):
    """
    Optimize parameters a and b using cvxpy for the plugin function.
    """
    objective, a, b = plugin_function_cvxpy(X, Y, x, h, lambda_, W, p, r)
    
    # Define the problem
    problem = cp.Problem(cp.Maximize(objective))
    
    # Solve the problem
    problem.solve(verbose=False,solver='MOSEK') # MOSEK - commercial solver #tried: ECOS (plugin only); SCS

    return a.value, b.value, problem.value

def compute_max_alpha_W_r_prod_cvxpy(X, Y, x, h, r, delta=0.1):
    """
    Compute the maximum value of alpha_W_r_prod_i over the uncertainty set as a CVXPY expression.
    
    Parameters:
    X : array-like, shape (N,)
        Sample data points.
    Y : array-like, shape (N, M)
        M-dimensional binary vectors corresponding to each X_i.
    x : float
        Given parameter.
    h : float
        Bandwidth.
    delta : float
        Confidence level.
    r : CVXPY expression, shape (N, p)
        N x p dimensional CVXPY expression.
    
    Returns:
    alpha*grad_W*r at all extreme points of the adversarial set 
    """
    N, M = Y.shape
    p = 2**M - M - 1

    # Compute the uncertainty set
    uncertainty_set = compute_uncertainty_set(X, Y, x, h, delta)

    # Compute alpha_tilde for each X_i
    alpha_tilde_X = np.zeros((N, M))
    for i in range(N):
        alpha_tilde_X[i], _ = local_linear_regression(X, Y, X[i], h)

    # Compute W matrix
    W, combinations = compute_W_matrix(Y, alpha_tilde_X)

    # Compute grad_W matrix
    grad_W = compute_grad_W_all_alpha(Y, alpha_tilde_X, W, combinations)  # Shape (N, M, p)

    # Initialize W_r_prod as an empty list
    W_r_prod_list = []
    
    # Compute W_r_prod for each k (since grad_W is 3D)
    for k in range(grad_W.shape[1]):
        W_r_prod_k = cp.diag(cp.matmul(grad_W[:, k, :],r.T))  # Shape (N,)
        W_r_prod_list.append(W_r_prod_k)
    
    # Stack the list to form a matrix W_r_prod
    W_r_prod = cp.vstack(W_r_prod_list)  # Shape (M, N)
    W_r_prod = cp.reshape(W_r_prod,(M,N)) # Shape (M, N)

    # Function to process a single scenario
    def process_scenario(t):
        alpha_diff_t = uncertainty_set[t, :, :] - alpha_tilde_X  # Shape (N, M)
        product = cp.diag(cp.matmul(alpha_diff_t, W_r_prod))  # Shape (N,)
        return product

    # Use ThreadPoolExecutor to parallelize the loop
    with ThreadPoolExecutor() as executor:
        alpha_W_r_prod_list = list(executor.map(process_scenario, range(uncertainty_set.shape[0])))

    # Stack all the results
    alpha_W_r_prod_sum = cp.vstack(alpha_W_r_prod_list)  # Shape (4^M, N)
    alpha_W_r_prod_sum = cp.reshape(alpha_W_r_prod_sum,(uncertainty_set.shape[0],N)) # Shape (4^M, N)

    return alpha_W_r_prod_sum # Shape (4^M, N)


def FO_function_cvxpy(X, Y, x, h, lambda_, W, p, r):
    """
    Objective function to minimize using cvxpy.
    """
    a = cp.Variable(p)
    b = cp.Variable(p)

    N = X.shape[0]

    # Reshape a and b for broadcasting
    a_broadcast = cp.reshape(a, (1, p))  # Shape (1, p)
    # Reshape b for broadcasting
    b_broadcast = cp.reshape(b, (1, p))  # Shape (1, p)
        
    # Compute r
    r_pred = a_broadcast + cp.matmul((X - x),b_broadcast) 
    
    # Compute max_alpha_W_r_prod (assumes this function is already implemented)
    alpha_W_r_prod_sum = compute_max_alpha_W_r_prod_cvxpy(X, Y, x, h, r_pred)
    # print('max_alpha_W_r_prod',max_alpha_W_r_prod.shape)
    
    # Compute inner product and kernel weights
    inner_product = cp.diag(cp.matmul(W, r_pred.T))  # Shape (N,)
    inner_product = cp.reshape(inner_product,(N,1))
    K_h = cp.reshape(compute_kernel_weights(X, x, h), (N,1))  # Flatten K_h to shape (N,)

    
    # Compute the first term 
    # Sum along the correct axis if intended
    sum_result = inner_product + alpha_W_r_prod_sum.T  # Shape (N, 4^M)
    log_term = cp.log(1 + sum_result)
    term1 = cp.min(cp.sum(cp.matmul(K_h.T,log_term))) 
    
    # Compute w_k and w_k1 (assumes this function is already implemented)
    w_k, w_k1 = compute_wk_weights(X, x, h, W, r, p)
    
    # Compute the penalty term
    penalty = lambda_ * (cp.sum(cp.multiply(w_k, cp.abs(a.T))) + h * cp.sum(cp.multiply(w_k1, cp.abs(b.T)))) 
    
    # Define the objective
    objective = term1 - penalty
    
    return objective, a, b

def FO_objective_cvxpy(X, Y, x, h, lambda_, W, p, r):
    """
    Objective function to minimize using cvxpy.
    """
    a = cp.Variable(p)
    b = cp.Variable(p)

    N = X.shape[0]

    # Reshape a and b for broadcasting
    a_broadcast = cp.reshape(a, (1, p))  # Shape (1, p)
    # Reshape b for broadcasting
    b_broadcast = cp.reshape(b, (1, p))  # Shape (1, p)
        
    # Compute r
    r_pred = a_broadcast + cp.matmul((X - x),b_broadcast) 
    
    # Compute max_alpha_W_r_prod (assumes this function is already implemented)
    alpha_W_r_prod_sum = compute_max_alpha_W_r_prod_cvxpy(X, Y, x, h, r_pred)
    
    # Compute inner product and kernel weights
    inner_product = cp.diag(cp.matmul(W, r_pred.T))  # Shape (N,)
    inner_product = cp.reshape(inner_product,(N,1))
    K_h = cp.reshape(compute_kernel_weights(X, x, h), (N,1))  # Flatten K_h to shape (N,)

    # Compute the first term 
    # Sum along the correct axis if intended
    sum_result = inner_product + alpha_W_r_prod_sum.T  # Shape (N, 4^M)
    log_term = cp.log(1 + sum_result) # Shape (N, 4^M)
    term1 = cp.matmul(K_h.T,log_term)  # Shape (1, 4^M)
    
    # Compute w_k and w_k1 (assumes this function is already implemented)
    w_k, w_k1 = compute_wk_weights(X, x, h, W, r, p)
    
    # Compute the penalty term
    penalty = lambda_ * (cp.sum(cp.multiply(w_k, cp.abs(a.T))) + h * cp.sum(cp.multiply(w_k1, cp.abs(b.T)))) 
    
    # Define the objective
    objective = term1 - penalty
    
    return objective, a, b

def optimize_parameters_FO_cvxpy(X, Y, x, h, lambda_, W, p, r):
    """
    Optimize parameters a and b using cvxpy.
    """
    objective, a, b = FO_function_cvxpy(X, Y, x, h, lambda_, W, p, r)
    
    # Define the problem
    problem = cp.Problem(cp.Maximize(objective))
    
    # Solve the problem
    problem.solve(verbose=False,solver='MOSEK')

    return a.value, b.value, problem.value


"""Baseline Naive model: NW estimator"""
#NW estimator:
def nadaraya_watson(X,h,x,W):
    """
    X: The input data, a 2D numpy array where each row is a data point.
    y: The output data, a 1D numpy array.
    x_query: The point at which to estimate the output.
    bandwidth: The bandwidth of the Gaussian kernel.
    """
    kernel_weight , _ = uniform_kernel_bandwidth_h(X, x, h)
    numerator = kernel_weight*W
    return np.sum(numerator,axis=0) / np.sum(kernel_weight)


"""Performance Metric"""
"""define L2 estimation error for r_0 under unconditional case"""
def L2_estimation_error(r_estimate, r_true):
    assert len(r_true) == len(r_estimate)
    return np.sum((r_true - r_estimate)**2)
