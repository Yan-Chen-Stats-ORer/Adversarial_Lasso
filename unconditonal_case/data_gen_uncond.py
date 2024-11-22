import numpy as np
from itertools import product
from itertools import combinations
import itertools

def generate_combinations(M):
    """Generate all possible combinations of indices [0, 1, ..., M-1] with at least 2 elements."""
    comb_list = []
    for r in range(2, M+1):
        comb_list.extend(combinations(range(M), r))
    return comb_list

def pick_pairs(M, K):
    """Generate pairs as specified in the previous answer."""
    pairs = []
    max_pairs = min(K, M // 2)
    for i in range(max_pairs):
        pairs.append((2*i, 2*i + 1))
    return pairs

def index_combinations_dict(comb_list):
    """Create a dictionary to map combinations to their index."""
    return {comb: idx for idx, comb in enumerate(comb_list)}

def assign_r0(M, K, rho):
    """Assign values to r_0 matrix based on the specified conditions."""
    comb_list = generate_combinations(M)
    comb_dict = index_combinations_dict(comb_list)
    r_0 = [0] * (2**M - M - 1)
    
    pairs = pick_pairs(M, K)
    for k, pair in enumerate(pairs):
        if pair in comb_dict:
            r_0[comb_dict[pair]] = rho[k]
    
    return r_0, comb_list

def Z_vectors(y, alpha, eps=1e-3):
    numerator = y - alpha
    denominator = np.sqrt(alpha * (1 - alpha))
    z = numerator / np.maximum(denominator, eps)
    return z

def W_func(M,z):
    col_idx = range(M)
    W_idx = []
    W_l = []
    #find all possible combination of columns of column set size from 2 to M
    for L in range(2,M+1):
        for l in itertools.combinations(col_idx, L):
            W_idx.append(l)
            #for the columns in the subset, multiply the z
            multiplication = 1
            for idx in l:
                multiplication = multiplication * z[:,idx]
            W_l.append(multiplication)
    W_l = np.matrix(W_l)
    W_l = W_l.T
    return W_l, W_idx


def compute_W_matrix(y, alpha, comb_list):
    """Compute the W_oracle vector based on y, alpha, and the combination list."""
    W_oracle = np.zeros(len(comb_list))
    for idx, comb in enumerate(comb_list):
        product = 1
        for j in comb:
            product *= (y[j] - alpha[j]) / np.sqrt(alpha[j] * (1 - alpha[j]))
        W_oracle[idx] = product
    return W_oracle

def compute_p_y(y, alpha, r_0, W_oracle):
    """Compute p(y) based on the given formula."""
    r_0 = np.array(r_0)
    r_0 = r_0.reshape(-1,)
    term1 = 1 + np.dot(W_oracle, r_0)
    term2 = np.prod([alpha[j]**y[j] * (1 - alpha[j])**(1 - y[j]) for j in range(len(y))])
    return term1 * term2

def precompute_probabilities(M, alpha, r_0, comb_list):
    """Precompute probabilities for all possible M-dimensional binary vectors."""
    all_vectors = list(product([0, 1], repeat=M))
    probabilities = []
    
    for y in all_vectors:
        W_oracle = compute_W_matrix(y, alpha,comb_list)
        p_y = compute_p_y(y, alpha, r_0, W_oracle)
        probabilities.append(p_y)
    
    probabilities = np.array(probabilities)
    
    return all_vectors, probabilities

def sample_vectors(N, all_vectors, probabilities):
    """Sample N binary vectors based on the precomputed probabilities."""
    indices = np.random.choice(len(all_vectors), size=N, p=probabilities)
    samples = [all_vectors[idx] for idx in indices]
    return samples

def draw_rho_general(s,M):
    p = 2**M-M-1
    non_zero_idx = np.random.choice(range(p),s,replace=False)
    support_r0 = np.array(non_zero_idx)
    rho = np.random.uniform(low=-0.1 , high=0.8, size=s)
    return rho,support_r0

"""Assign values to entries of r_0 and make sure 1+W*r>0 always"""
def assign_r0_general_vectorized(s,M):
    rho,support_r0 = draw_rho_general(s,M)
    #first set the correlation coefficient according to 
    comb_list = generate_combinations(M)
    # comb_dict = index_combinations_dict(comb_list)
    r_0 = np.zeros(2**M - M - 1)
    for j in range(len(support_r0)):
        r_0[support_r0[j]] = rho[j]      
    return r_0, comb_list

def r0_setup(s,M,alpha):
    probabilities = -np.ones(2**M)
    while np.any(probabilities<0):
        r_0, comb_list = assign_r0_general_vectorized(s,M)
        all_vectors, probabilities = precompute_probabilities(M, alpha, r_0, comb_list)

    _, comb_list = assign_r0_general_vectorized(s,M)
    all_vectors = list(product([0, 1], repeat=M))
    _, probabilities = precompute_probabilities(M, alpha, r_0, comb_list)
    print('true correlation coefficient r_0',r_0)
    r_0 = np.array(r_0)
    r_0 = r_0.reshape(1,-1)
    return r_0, probabilities

"""define L2 estimation error for r_0 under unconditional case"""
def L2_estimation_error(r_estimate, r_true):
    r_estimate = r_estimate.reshape(1,-1)

    r_true = np.array(r_true)
    r_estimate = np.array(r_estimate)

    assert len(r_true) == len(r_estimate)
    return np.sum((r_true - r_estimate)**2)


"""define average L2 error for the estimation of r_0"""
def mean_squared_error(r_pred, r_true):
    r_true = np.array(r_true)
    r_pred = np.array(r_pred)
    assert len(r_true) == len(r_pred)
    return np.mean(np.sqrt(np.sum((r_true - r_pred)**2)),axis=0)