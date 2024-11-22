import numpy as np
from utils_uncond import *
from data_gen_uncond import *

def initialize_storage():
    error_dict = {
        'SAA': [], 'PI_0': [], 'PI_r0': [], 'FO_0': [], 'FO_r0': [],
        'PI_alpha0_0': [], 'PI_alpha0_r0': [],'PI_0_theory': [], 'PI_r0_theory': [], 
        'FO_0_theory': [], 'FO_r0_theory': [], 'PI_alpha0_0_theory': [], 'PI_alpha0_r0_theory': []
    }
    
    prob_error_dict = {
        'expected': {'SAA': [], 'PI_0': [], 'PI_r0': [], 'FO_0': [], 'FO_r0': [], 'PI_alpha0_0': [], 'PI_alpha0_r0': [],'PI_0_theory': [], 'PI_r0_theory': [], 
        'FO_0_theory': [], 'FO_r0_theory': [], 'PI_alpha0_0_theory': [], 'PI_alpha0_r0_theory': []},
        'max': {'SAA': [], 'PI_0': [], 'PI_r0': [], 'FO_0': [], 'FO_r0': [], 'PI_alpha0_0': [], 'PI_alpha0_r0': [], 'PI_0_theory': [], 'PI_r0_theory': [], 
        'FO_0_theory': [], 'FO_r0_theory': [], 'PI_alpha0_0_theory': [], 'PI_alpha0_r0_theory': []}
    }

    return error_dict, prob_error_dict

def cross_validation_folds(Y, lambdas_PI, lambdas_FO, r_0, p, W_oracle):
    kf = KFold(n_splits=5, shuffle=True)
    
    PI_k_fold_scores_0 = {lam: [] for lam in lambdas_PI}
    PI_k_fold_scores_r0 = {lam: [] for lam in lambdas_PI}
    FO_k_fold_scores_0 = {lam: [] for lam in lambdas_FO}
    FO_k_fold_scores_r0 = {lam: [] for lam in lambdas_FO}
    PI_alpha0_k_fold_scores_0 = {lam: [] for lam in lambdas_PI}
    PI_alpha0_k_fold_scores_r0 = {lam: [] for lam in lambdas_PI}
    
    M = Y.shape[1]
    
    for train_index, _ in kf.split(Y):
        CV_Y = Y[train_index, :]
        alpha_hat = np.mean(CV_Y,axis=0)
        z = Z_vectors(CV_Y,alpha_hat)
        CV_W, _ = W_func(M,z)
        CV_W_oracle = W_oracle[train_index, :]

        for lam in lambdas_PI:
            PI_r0_x_predict_0, _ = optimize_parameters_plugin_cvxpy(CV_Y, lam, CV_W, np.zeros((p,1)))
            PI_k_fold_scores_0[lam].append(L2_estimation_error(r_0, PI_r0_x_predict_0))
            PI_r0_x_predict_r0, _= optimize_parameters_plugin_cvxpy(CV_Y, lam, CV_W, r_0)
            PI_k_fold_scores_r0[lam].append(L2_estimation_error(r_0, PI_r0_x_predict_r0))
            
            PI_alpha0_r0_x_predict_0, _ = optimize_parameters_plugin_cvxpy(CV_Y, lam, CV_W_oracle, np.zeros((p,1)))
            PI_alpha0_k_fold_scores_0[lam].append(L2_estimation_error(r_0, PI_alpha0_r0_x_predict_0))
            PI_alpha0_r0_x_predict_r0, _ = optimize_parameters_plugin_cvxpy(CV_Y, lam, CV_W_oracle, r_0)
            PI_alpha0_k_fold_scores_r0[lam].append(L2_estimation_error(r_0, PI_alpha0_r0_x_predict_r0))


        for lam in lambdas_FO: #Y, lambda_, W, r_init, alpha_hat,uncertainty_marginal
            FO_r0_x_predict_0, _ = optimize_parameters_FO_cvxpy(CV_Y, lam, CV_W, np.zeros((p,1)))
            FO_r0_x_predict_0 = FO_r0_x_predict_0.reshape(1,-1)
            FO_k_fold_scores_0[lam].append(L2_estimation_error(r_0, FO_r0_x_predict_0))
            
            FO_r0_x_predict_r0, _= optimize_parameters_FO_cvxpy(CV_Y, lam, CV_W, r_0)
            FO_r0_x_predict_r0 = FO_r0_x_predict_r0.reshape(1,-1)
            FO_k_fold_scores_r0[lam].append(L2_estimation_error(r_0, FO_r0_x_predict_r0))

    return PI_k_fold_scores_0, PI_k_fold_scores_r0, FO_k_fold_scores_0, FO_k_fold_scores_r0, PI_alpha0_k_fold_scores_0, PI_alpha0_k_fold_scores_r0

def optimize_and_store_results(Y, W, scores, lambdas, function, var_init, error_list):
    best_lam, best_score = None, float('inf')
    for lam in lambdas:
        score = np.mean(scores[lam])
        if score < best_score:
            best_score = score
            best_lam = lam
    prediction, _ = function(Y, best_lam, W, var_init)     
    error_list.append(best_score)
    
    return best_lam, prediction

def print_final_results(error_dict, prob_error_dict):
    print("Final results:")
    print("SAA Estimator Errors:")
    print("RMSE SAA: ", np.sqrt(np.mean(error_dict['SAA'])))
    
    for key in ['PI_0', 'PI_r0', 'FO_0', 'FO_r0', 'PI_alpha0_0', 'PI_alpha0_r0',
                'PI_0_theory', 'PI_r0_theory', 'FO_0_theory', 'FO_r0_theory', 
                'PI_alpha0_0_theory', 'PI_alpha0_r0_theory']:
        print(f"RMSE {key}: ", np.sqrt(np.mean(error_dict[key])))
    
    print("Probability Estimation Errors (Expected, Max):")
    for metric in ['expected', 'max']:
        for method in prob_error_dict[metric]:
            print(f"{metric.capitalize()} {method}: ", np.mean(prob_error_dict[metric][method]))
