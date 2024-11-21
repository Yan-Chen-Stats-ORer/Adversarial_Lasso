"""
Synthetic data generation for simulation 
"""

import numpy as np
from itertools import product, combinations

"""compute marginal probability at a given x"""
def compute_alpha(x,theta):
    return 1 / (1 + np.exp(-theta * x))

"""generate bundles of [M] containing at least two elements"""
def generate_combinations(M):
    comb_list = []
    for r in range(2, M+1):
        comb_list.extend(combinations(range(M), r))
    return comb_list

"""
at given x, randomly pick s support, and make the entries as support for all X; 
then assign values to the support
"""
def draw_rho_at_x(s,M,x):
    p = 2**M-M-1
    non_zero_idx = np.random.choice(range(p),s,replace=False)
    support_r0 = np.array(non_zero_idx)
    # print(support_r0)
    rho = np.zeros((1,len(support_r0)))
    for j in range(len(support_r0)):
        k = support_r0[j]
        rho[:,j] = ((-1)**(k+1))*x*(k+1)*0.003
    return rho[0], support_r0

"""assign value to r_0(x)"""
def assign_r0_general_at_x(M,support_r0,rho):
    comb_list = generate_combinations(M)
    r_0_all = np.zeros(2**M - M - 1)
    for j in range(len(support_r0)):
        r_0_all[support_r0[j]] = rho[j]      
    return r_0_all, comb_list

"""compute probabilities for all possible M-dimensional binary vectors."""
def precompute_probabilities_x(M, alpha_x, r_0_x, comb_list):
    all_vectors = list(product([0, 1], repeat=M))
    probabilities = []
    
    for y in all_vectors:
        W_oracle = compute_W_oracle(y, alpha_x, comb_list)
        p_y = compute_p_y(y, alpha_x, r_0_x, W_oracle)
        probabilities.append(p_y)
    
    probabilities = np.array(probabilities)
    probabilities /= probabilities.sum()  # Normalize to sum to 1
    
    return all_vectors, probabilities

"""
generate 1-dimensional covariates
"""
def sample_X(N):
    return np.random.uniform(0, 1, N)

"""
r_0 generation for the general s 
"""
def draw_rho_general(X,support_r0):
    support_r0 = np.array(support_r0)
    rho = np.zeros((len(X),len(support_r0)))
    for j in range(len(support_r0)):
        k = support_r0[j]
        rho[:,j] = ((-1)**(k+1))*X*(k+1)*0.003
    return rho

def index_combinations_dict(comb_list):
    return {comb: idx for idx, comb in enumerate(comb_list)}

def compute_W_oracle(y, alpha, comb_list):
    W_oracle = np.zeros(len(comb_list))
    for idx, comb in enumerate(comb_list):
        product = 1
        for j in comb:
            product *= (y[j] - alpha[j]) / np.sqrt(alpha[j] * (1 - alpha[j]))
        W_oracle[idx] = product
    return W_oracle


"""Assign values to entries of r_0_all and make sure 1+W*r>0 always"""
def assign_r0_general_vectorized(M,X,support_r0):
    rho = draw_rho_general(X,support_r0)
    comb_list = generate_combinations(M)
    r_0_all = np.zeros((rho.shape[0], 2**M - M - 1))
    for j in range(len(support_r0)):
        r_0_all[:, support_r0[j]] = rho[:, j]      
    return r_0_all, comb_list

def compute_p_y(y, alpha, r_0, W_oracle):
    term1 = 1 + np.dot(W_oracle, r_0)
    term2 = np.prod([alpha[j]**y[j] * (1 - alpha[j])**(1 - y[j]) for j in range(len(y))])
    return term1 * term2

def precompute_probabilities_vectorized(M, alpha, r_0_all, comb_list):
    all_vectors = np.array(list(product([0, 1], repeat=M)))
    all_probs = np.zeros((alpha.shape[0], all_vectors.shape[0]))
    
    for i in range(alpha.shape[0]):
        r_0 = r_0_all[i,:]
        _, probabilities = precompute_probabilities_x(M, alpha[i], r_0 , comb_list)
        all_probs[i, :] = probabilities

    return all_vectors, all_probs

def sample_vectors_vectorized(N, all_vectors, probabilities):
    samples = np.zeros((N, len(all_vectors[0])), dtype=int)
    for i in range(N):
        samples[i] = all_vectors[np.random.choice(len(all_vectors), p=probabilities[i])]
    return samples

def sample_vectors(N, all_vectors, probabilities):
    """Sample N binary vectors based on the precomputed probabilities."""
    indices = np.random.choice(len(all_vectors), size=N, p=probabilities)
    samples = [all_vectors[idx] for idx in indices]
    return samples

def sample_X_discrete(N,x_values):
    return np.random.choice(x_values, size=N, replace=True)

"""compute marginal probabilities"""
def compute_alpha_vectorized(X, theta):
    return 1 / (1 + np.exp(-np.outer(X, theta)))




