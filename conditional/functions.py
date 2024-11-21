import numpy as np
from utils import *

def initialize_storage():
    error_dict = {
        'NW': [], 'PI_0': [], 'PI_r0': [], 'FO_0': [], 'FO_r0': [],
        'PI_alpha0_0': [], 'PI_alpha0_r0': [],'PI_0_theory': [], 'PI_r0_theory': [], 
        'FO_0_theory': [], 'FO_r0_theory': [], 'PI_alpha0_0_theory': [], 'PI_alpha0_r0_theory': []
    }
    
    prob_error_dict = {
        'expected': {'NW': [], 'PI_0': [], 'PI_r0': [], 'FO_0': [], 'FO_r0': [], 'PI_alpha0_0': [], 'PI_alpha0_r0': [],'PI_0_theory': [], 'PI_r0_theory': [], 
        'FO_0_theory': [], 'FO_r0_theory': [], 'PI_alpha0_0_theory': [], 'PI_alpha0_r0_theory': []},
        'max': {'NW': [], 'PI_0': [], 'PI_r0': [], 'FO_0': [], 'FO_r0': [], 'PI_alpha0_0': [], 'PI_alpha0_r0': [], 'PI_0_theory': [], 'PI_r0_theory': [], 
        'FO_0_theory': [], 'FO_r0_theory': [], 'PI_alpha0_0_theory': [], 'PI_alpha0_r0_theory': []}
    }

    return error_dict, prob_error_dict


def cross_validation_folds(X, Y, lambdas_PI, lambdas_FO, r_0, p, h, x,r_0_specific,W_oracle):
    kf = KFold(n_splits=5, shuffle=True)
    
    PI_k_fold_scores_0 = {lam: [] for lam in lambdas_PI}
    PI_k_fold_scores_r0 = {lam: [] for lam in lambdas_PI}
    FO_k_fold_scores_0 = {lam: [] for lam in lambdas_FO}
    FO_k_fold_scores_r0 = {lam: [] for lam in lambdas_FO}
    PI_alpha0_k_fold_scores_0 = {lam: [] for lam in lambdas_PI}
    PI_alpha0_k_fold_scores_r0 = {lam: [] for lam in lambdas_PI}
    
    for train_index, _ in kf.split(X):
        CV_X, CV_Y = X[train_index, :], Y[train_index, :]
        alpha_tilde_CV, _ = local_linear_regression(CV_X, CV_Y, x, h)
        CV_W, _ = compute_W_matrix(CV_Y, alpha_tilde_CV)
        CV_W_oracle = W_oracle[train_index, :]

        for lam in lambdas_PI:
            PI_r0_x_predict_0, _, _ = optimize_parameters_plugin_cvxpy(CV_X, CV_Y, x, h, lam, CV_W, p, np.zeros((len(train_index), p)))
            PI_k_fold_scores_0[lam].append(L2_estimation_error(r_0_specific, PI_r0_x_predict_0))
            PI_r0_x_predict_r0, _, _ = optimize_parameters_plugin_cvxpy(CV_X, CV_Y, x, h, lam, CV_W, p, r_0[train_index, :])
            PI_k_fold_scores_r0[lam].append(L2_estimation_error(r_0_specific, PI_r0_x_predict_r0))
            
            PI_alpha0_r0_x_predict_0, _, _ = optimize_parameters_plugin_cvxpy(CV_X, CV_Y, x, h, lam, CV_W_oracle, p, np.zeros((len(train_index), p)))
            PI_alpha0_k_fold_scores_0[lam].append(L2_estimation_error(r_0_specific, PI_alpha0_r0_x_predict_0))
            PI_alpha0_r0_x_predict_r0, _, _ = optimize_parameters_plugin_cvxpy(CV_X, CV_Y, x, h, lam, CV_W_oracle, p, r_0[train_index, :])
            PI_alpha0_k_fold_scores_r0[lam].append(L2_estimation_error(r_0_specific, PI_alpha0_r0_x_predict_r0))

        
        for lam in lambdas_FO:
            FO_r0_x_predict_0, _, _ = optimize_parameters_FO_cvxpy(CV_X, CV_Y, x, h, lam, CV_W, p, np.zeros((len(train_index), p)))
            FO_k_fold_scores_0[lam].append(L2_estimation_error(r_0_specific, FO_r0_x_predict_0))
            FO_r0_x_predict_r0, _, _ = optimize_parameters_FO_cvxpy(CV_X, CV_Y, x, h, lam, CV_W, p, r_0[train_index, :])
            FO_k_fold_scores_r0[lam].append(L2_estimation_error(r_0_specific, FO_r0_x_predict_r0))

    return PI_k_fold_scores_0, PI_k_fold_scores_r0, FO_k_fold_scores_0, FO_k_fold_scores_r0, PI_alpha0_k_fold_scores_0, PI_alpha0_k_fold_scores_r0


def optimize_and_store_results(X, Y, W, scores, lambdas, function, p, h, x, var_init, error_list):
    best_lam, best_score = None, float('inf')
    for lam in lambdas:
        score = np.mean(scores[lam])
        if score < best_score:
            best_score = score
            best_lam = lam
    prediction, _, _ = function(X, Y, x, h, best_lam, W, p, var_init)     
    error_list.append(best_score)
    
    return best_lam, prediction

def compute_theoretical_lambda_plugin(W_oracle, r_0, p, M, N, h,X,x):
    W_a_squareroot, W_b_squareroot = compute_wk_weights(X, x, h, W_oracle, r_0, p)
    W_a, W_b = W_a_squareroot**2, W_b_squareroot**2
    w_k, w_k1 = compute_wk_weights(X, x, h, W_oracle, [0.0]*np.zeros((N, p)), p)
    penalty_a, penalty_b = np.max(W_a/w_k), np.max(W_b/w_k1)
    return np.maximum(penalty_a, penalty_b) + M / np.sqrt(N * h) + M * (h**2)

def compute_theoretical_lambda_FO(W_oracle, r_0, p, M, N, h,X,x):
    W_a_squareroot, W_b_squareroot = compute_wk_weights(X, x, h, W_oracle, r_0, p)
    W_a, W_b = W_a_squareroot**2, W_b_squareroot**2
    w_k, w_k1 = compute_wk_weights(X, x, h, W_oracle, [0.0]*np.zeros((N, p)), p)
    penalty_a, penalty_b = np.max(W_a/w_k), np.max(W_b/w_k1)
    return np.maximum(penalty_a, penalty_b) + np.sqrt(M) / np.sqrt(N * h) + np.sqrt(M) * (h**2)

def compute_theoretical_lambda_plugin_oracle(W_oracle, r_0, p, M, N, h,X,x):
    W_a_squareroot, W_b_squareroot = compute_wk_weights(X, x, h, W_oracle, r_0, p)
    W_a, W_b = W_a_squareroot**2, W_b_squareroot**2
    w_k, w_k1 = compute_wk_weights(X, x, h, W_oracle, [0.0]*np.zeros((N, p)), p)
    penalty_a, penalty_b = np.max(W_a/w_k), np.max(W_b/w_k1)
    return np.maximum(penalty_a, penalty_b) 


def print_final_results(error_dict, prob_error_dict):
    print("Final results:")
    
    print("NW Estimator Errors:")
    print("RMSE NW: ", np.sqrt(np.mean(error_dict['NW'])))
    
    for key in ['PI_0', 'PI_r0', 'FO_0', 'FO_r0', 'PI_alpha0_0', 'PI_alpha0_r0',
                'PI_0_theory', 'PI_r0_theory', 'FO_0_theory', 'FO_r0_theory', 
                'PI_alpha0_0_theory', 'PI_alpha0_r0_theory']:
        print(f"RMSE {key}: ", np.sqrt(np.mean(error_dict[key])))
    
    print("Probability Estimation Errors (Expected, Max):")
    for metric in ['expected', 'max']:
        for method in prob_error_dict[metric]:
            print(f"{metric.capitalize()} {method}: ", np.mean(prob_error_dict[metric][method]))
