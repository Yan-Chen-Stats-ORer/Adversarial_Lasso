import numpy as np
import matplotlib.pyplot as plt
import random 
import argparse
from scipy.stats import norm
import time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from data_gen import * 
from functions import *
from utils import *
from causal_util import *

"""
Estimate r_0(x) at a given x.
"""

#set parameters 
parser = argparse.ArgumentParser(description="Localized Lasso Conditional Case Illustration")
parser.add_argument("--N",default=100,help="sample size")
parser.add_argument("--M",default=4,help="binary vector dimension")
parser.add_argument("--s",default=2,help="cardinality of the support of r_0")
args = parser.parse_args()

N = int(args.N)
M = int(args.M) 
s = int(args.s)
p = 2**M - M - 1 #dimension of generalized correlation coefficient 
h = (np.log(p)/N)**(0.2)# Bandwidth
print('bandwidth',h)

# Estimate r_0 for specific values of x
X_0 = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
theta = np.random.uniform(low=0.0,high=1.0,size=M)
all_vectors = list(product([0, 1], repeat=M))
r_0_specific = {} #r_0(x) at each x
true_prob = {} #joint probabilities at each x
marginal = {} #marginal probabilities at each x

#compute propensity scores at each x 
for x in X_0:
    rho, support_r0 = draw_rho_at_x(s,M,x)
    r_0_specific_x, comb_list_specific = assign_r0_general_at_x(M, support_r0, rho)
    r_0_specific[x] = r_0_specific_x
    alpha_x = compute_alpha(x, theta)
    _, probabilities_x = precompute_probabilities_x(M, alpha_x, r_0_specific[x], comb_list_specific)
    true_prob[x] = probabilities_x
    marginal[x] = alpha_x

print('propensity scores', true_prob)

x_values_all = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
metrics = ["error", "prediction", "max_prob_est_error"]
prob_predict = ["propensity_treat_est", "propensity_control_est"]
estimators = ["NW", "PI_0", "PI_r0", "FO_0", "FO_r0","multinomial"]
comb_list = generate_combinations(M)

var_init = [0.0] * (2 * p)

# error_dict, prob_error_dict = initialize_storage()
# Define ATE treatment levels
# Define beta_vals
beta_vals = np.linspace(0.0, 1.5, 2**M)  # total number of levels of treatment
mu_vals = np.linspace(0.5, 2.0, 2**M)
# Initialize dictionary to store ATE_true results
ATE_real = {}

# Loop over beta_vals from index 1 to 15
for i in range(1, 2**M):
    ATE_real[i] = ATE_true(beta_vals[i],beta_vals[0],mu_vals[i],mu_vals[0],X_0)

# Initialize AIPW results for each estimator
AIPW_results = {estimator: {i: [] for i in range(1, 2**M)} for estimator in estimators}

# Set the target number of results
target_results = 100

# Initialize iteration counter
j = 0

while True:
    random.seed(j)
    print(f"Starting iteration {j}")
    #############################################################
    """generate samples"""
    N = int(args.N)
    X_original = sample_X_discrete(N,x_values_all)
    rho = draw_rho_general(X_original,support_r0)
    #generate marginal at all X 
    alpha_original = compute_alpha_vectorized(X_original, theta) 
    alpha_original  = np.array(alpha_original).reshape(N, -1)
    probabilities = np.array([true_prob[x] for x in X_original])
    r_0_original = np.array([r_0_specific[x] for x in X_original])
    r_0_original = np.array(r_0_original).reshape(N,-1)
    # Sample binary vectors for all X
    samples = sample_vectors_vectorized(N, all_vectors, probabilities)
    Y_original = np.array(samples)
    X_original = X_original.reshape(N, 1)

    AIPW_current_iter = {estimator: {i: [] for i in range(1, 2**M)} for estimator in estimators}

    # Split X_original and corresponding Y_original into two halves
    split_index = len(X_original) // 2
    X_halves = [X_original[:split_index], X_original[split_index:]]
    Y_halves = [Y_original[:split_index], Y_original[split_index:]]

    # Loop through each half for testing and training, checking whether some treatment levels have no data
    skip_iteration = False
    for test_half_index, (X_test_half, Y_test_half) in enumerate(zip(X_halves, Y_halves)):
        train_half_index = 1 - test_half_index  # The other half is the training set
        X_train_half = X_halves[train_half_index]
        Y_train_half = Y_halves[train_half_index]
        N_train = len(X_train_half)
        N_test = len(X_test_half)

        for i, T in enumerate(all_vectors):
            mask_level = np.where(np.all(Y_train_half == T, axis=1))[0]
            if len(mask_level) <= 2:
                print(f"Skipping the current iteration {j} due to insuffiencient mask_level for treatment vector {T}")
                skip_iteration = True
                break
        if skip_iteration:
            break
    
    if skip_iteration:
        print(f"Skipping iteration {j} due to empty mask_level for a treatment vector.")
        j += 1
        continue 
    
    # Proceed with computation if no empty mask_level
    # Loop through each half for testing and training
    for test_half_index, (X_test_half, Y_test_half) in enumerate(zip(X_halves, Y_halves)):
        train_half_index = 1 - test_half_index  # The other half is the training set
        
        X_train_half = X_halves[train_half_index]
        Y_train_half = Y_halves[train_half_index]
        N_train = len(X_train_half)
        N_test = len(X_test_half)
    
        # data_dicts = {f"{estimator}_{metric}": {x: [] for x in X_unique} for estimator in estimators for metric in metrics}
        prob_dicts = {f"{estimator}_propensity_treat_est": {i: {x: [] for x in X_unique} for i in range(1, 2**M)} for estimator in estimators}
        control_dicts = {f"{estimator}_propensity_control_est": {x: [] for x in X_unique} for estimator in estimators}
        
        # Unique values in the testing half
        X_unique = np.unique(X_test_half)

        """Fit propensity score"""
        ######### multinomial discrete choice modeling #########

        # Encode binary sequences as categorical labels
        y_labels = np.apply_along_axis(binary_to_int, axis=1, arr=Y_train_half)
        model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        model.fit(X_train_half, y_labels)
        predicted_probabilities = model.predict_proba(X_unique.reshape(-1, 1))

        # Build the multinomial_pred dictionary
        multinomial_pred = {}
        for x_val, probs in zip(X_unique, predicted_probabilities):
            multinomial_pred[x_val] = {i: prob for i, prob in enumerate(probs)}

        for x in X_unique:
            print(f"x = {x}")
            for level in range(1,2**M):
                prob_dicts["multinomial_propensity_treat_est"][level][x].append(multinomial_pred[x][level])
            control_dicts["multinomial_propensity_control_est"][x].append(multinomial_pred[x][0])
        

        for x in X_unique:
            print(f"x = {x}")
            # Only keep X in training set within h distance of x
            condition = np.abs(X_train_half - x) <= h
            X_train_filtered = X_train_half[condition]
            row_indices = np.where(condition)[0]
            #filter Y, alpha, r_0 by the row indices satisfying condition (s.t. kernel>0)
            Y_train_filtered = Y_train_half[row_indices]
            alpha_train_filtered = alpha_original[row_indices]
            r_0 = r_0_original[row_indices]
            N = len(Y_train_filtered)
            X_train_filtered = X_train_filtered.reshape(N,1)

            # Compute W_oracle and estimated alpha
            W_oracle, _ = compute_W_matrix(Y_train_filtered, alpha_train_filtered)
            alpha_tilde, _ = local_linear_regression(X_train_filtered, Y_train_filtered, x, h)
            W, _ = compute_W_matrix(Y_train_filtered, alpha_tilde)

            # NW Estimator
            NW_estimator = nadaraya_watson(X_train_filtered, h, x, W)
            # data_dicts["NW_prediction"][x].append(NW_estimator)
            NW_error = L2_estimation_error(NW_estimator, r_0_specific[x])

            # Probabilities
            r0_hat_NW = NW_estimator.reshape(-1, p)
            alpha_tilde_array = alpha_tilde.reshape(-1, M)
            _, probabilities_x_NW = precompute_probabilities_vectorized(M, alpha_tilde_array, r0_hat_NW, comb_list)

            for level in range(1,2**M):
                prob_dicts["NW_propensity_treat_est"][level][x].append(probabilities_x_NW[0][level])
            control_dicts["NW_propensity_control_est"][x].append(probabilities_x_NW[0][0])
            

            # Plug-in and First-Order Estimators (PI and FO) Processing
            lambda_values = {
                "PI_0": compute_theoretical_lambda_plugin(W_oracle, [0.0] * np.zeros((N, p)), p, M, N, h, X_train_filtered, x),
                "PI_r0": compute_theoretical_lambda_plugin(W_oracle, r_0, p, M, N, h, X_train_filtered, x),
                "FO_0": compute_theoretical_lambda_FO(W_oracle, [0.0] * np.zeros((N, p)), p, M, N, h, X_train_filtered, x),
                "FO_r0": compute_theoretical_lambda_FO(W_oracle, r_0, p, M, N, h, X_train_filtered, x)
                # "PI_alpha0_0": compute_theoretical_lambda_plugin_oracle(W_oracle, [0.0]*np.zeros((N, p)), p, M, N, h,X_train_filtered,x),
                # "PI_alpha0_r0": compute_theoretical_lambda_plugin_oracle(W_oracle, r_0, p, M, N, h,X_train_filtered,x)
            }

            for estimator, lambda_star in lambda_values.items():
                if "FO" in estimator:
                    optimizer = optimize_parameters_FO_cvxpy
                # if "alpha0" in estimator:
                #     optimizer = optimize_parameters_plugin_cvxpy
                elif "PI" in estimator:
                    optimizer = optimize_parameters_plugin_cvxpy
                
                prediction, _, _ = optimizer(X_train_filtered, Y_train_filtered, x, h, lambda_star, W, p, [0.0]*np.zeros((N, p)))
                # data_dicts[f"{estimator}_prediction"][x].append(prediction)
                r0_hat = prediction[:p].reshape(-1, p)
                _, probabilities_x_hat = precompute_probabilities_vectorized(M, alpha_tilde.reshape(-1, M), r0_hat, comb_list)
                for level in range(1,2**M):
                    prob_dicts[f"{estimator}_propensity_treat_est"][level][x].append(probabilities_x_hat[0][level])
                control_dicts[f"{estimator}_propensity_control_est"][x].append(probabilities_x_hat[0][0])
        
        """fit the outcome estimation model from the other half of dataset"""
        outcomes_train = np.zeros(N_train)
        outcomes_test = np.zeros(N_test)
        X_train_half = X_train_half.reshape(-1,)
        X_test_half = X_test_half.reshape(-1,)
        
        #generate outcome data
        for i, T in enumerate(all_vectors):
            mask_level = np.where(np.all(Y_train_half == T, axis=1))
            outcomes_train[mask_level] = outcome(beta_vals[i], mu_vals[i], X_train_half[mask_level])
            
            mask_level_test = np.where(np.all(Y_test_half == T, axis=1))
            outcomes_test[mask_level_test] = outcome(beta_vals[i], mu_vals[i], X_test_half[mask_level_test])

        mask_control = np.where(np.all(Y_train_half == all_vectors[0], axis=1))
        beta_0_hat, mu_0_hat = OLS_model(outcomes_train[mask_control], X_train_half[mask_control])

        # Loop through all levels from 1 to 2**M-1
        for level in range(1, 2**M):
            mask_treat_train = np.where(np.all(Y_train_half == all_vectors[level], axis=1))
            # Estimate beta_1_hat and mu_1_hat for this level
            beta_1_hat, mu_1_hat = OLS_model(outcomes_train[mask_treat_train], X_train_half[mask_treat_train])
            
            mask_treat_test = np.where(np.all(Y_test_half == all_vectors[level], axis=1))
            
            # Loop through estimators
            for estimator in estimators:
                # Get treatment and control probabilities from data_dicts
                treat_prob = prob_dicts[f"{estimator}_propensity_treat_est"][level]
                control_prob = control_dicts[f"{estimator}_propensity_control_est"]

                # Append AIPW results
                AIPW_current_iter[estimator][level].append(aipw_estimator(X_test_half, 
                                                        Y_test_half, outcomes_test, treat_prob,
                                                        control_prob, beta_0_hat, mu_0_hat,
                                                        beta_1_hat, mu_1_hat, mask_treat_test))
    
    #take average over two folds 
    for estimator in estimators:
        for level in range(1, 2**M):
            # Take the average of current iteration's values
            if AIPW_current_iter[estimator][level]:
                average_value = np.mean(AIPW_current_iter[estimator][level])
                # Append the average to AIPW_results
                AIPW_results[estimator][level].append(average_value)
            else:
                # Handle case where there are no values (e.g., append NaN or 0 if needed)
                AIPW_results[estimator][level].append(np.nan)

    print(f"Iteration {j} complete.")
    
    print(AIPW_results)

    N = int(args.N)
    M = int(args.M) 
    s = int(args.s)

    #compute coverage 
    T = len(AIPW_results['multinomial'][1]) #current iteration count
    coverage = coverage_compute(AIPW_results,estimators,ATE_real,true_prob,M,N,T)
    print('s=%i, M=%i, N=%i:'%(s,M,N))          
    print(coverage)

    # Check if all keys in AIPW_results have the required number of results
    all_completed = all(
        len(AIPW_results[estimator][level]) >= target_results
        for estimator in estimators
        for level in range(1, 2**M)
    )
    if all_completed:
        break  # Stop the loop once all keys have 50 results

    j += 1  # Increment iteration counter









