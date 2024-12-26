import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


"""
Generate outcomes on treatment and control groups
"""

def treat_outcome(beta_1,mu_1,X_1):
    """X_1: the covariate corresponding to treatment group"""
    N_1 = len(X_1)
    return beta_1+mu_1*X_1+np.random.normal(0,1,N_1)

def control_outcome(beta_0,mu,X_0):
    """X_0: the covariate corresponding to the control group"""
    N_0 = len(X_0)
    return beta_0+mu*X_0+np.random.normal(0,1,N_0)

##### outcome model for each combination of multiple treatment
def outcome(beta,mu,X):
    N_x = len(X)
    return beta+mu*X+np.random.normal(0,1,N_x)

"""
Run OLS to estimate treatment and control outcome models 
"""
def OLS_model(Y_group,X_group):
    X_group = np.array(X_group).reshape(-1, 1) 
    model = LinearRegression()
    model.fit(X_group, Y_group)
    # Get the coefficients and intercept
    intercept = model.intercept_
    coefficient = model.coef_
    return intercept, coefficient

def ATE_true(beta_t,beta_0,mu_t,mu_0,X):
    real_ate = (beta_t - beta_0)+(mu_t-mu_0)*np.mean(X)
    return real_ate

# Convert each binary sequence into a unique integer label
def binary_to_int(sequence):
    return int("".join(map(str, sequence)), 2)


def aipw_estimator(X,Y,outcome,treat_prob,control_prob,beta_0_hat,mu_0_hat,beta_1_hat,mu_1_hat,mask_treat):
    """
    Compute the AIPW estimator.

    Returns:
    float
        AIPW estimate of the average treatment effect.
    """
    mu_t = beta_1_hat+mu_1_hat*X        #treatment group model 
    mu_t_prime = beta_0_hat+mu_0_hat*X  #control group model
    term1 = np.mean(mu_t - mu_t_prime) 
    N = len(Y)

    # Create the treatment and control probabilities based on the covariates
    e_t = np.array([treat_prob[x] for x in X]).reshape(-1,)
    e_t_prime = np.array([control_prob[x] for x in X]).reshape(-1,)

    T_prime = np.where(np.all(Y==0,axis=1))  # Indicator for T' (Y_i = (0,0,0,0))

    term2 = (np.sum((outcome[mask_treat] - mu_t[mask_treat]) / e_t[mask_treat]))/N
    term3 = np.sum((outcome[T_prime] - mu_t_prime[T_prime]) / e_t_prime[T_prime])/N
    tau_aipw = term1 + term2 - term3
    return tau_aipw

# Compute the standard deviation of ATEs and generate confidence intervals
def compute_confidence_intervals(results,n):
    means = {level: np.mean(results[level]) for level in results}
    stds = {level: np.std(results[level], ddof=1) for level in results}  # ddof=1 for sample std
    cis = {level: (means[level] - 1.96 * stds[level]/np.sqrt(n), means[level] + 1.96 * stds[level]/np.sqrt(n)) for level in results}
    return means, stds, cis


def semiparametric_efficient_CI(probability,N,M,
                     X_0=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])):
    inv_control_avg = np.mean([1/probability[x][0] for x in X_0]) #E[1/e_0(X_i)]
    std = {}
    for t in range(1,2**M):
        std[t] = np.sqrt(1.0/15*0.01*(t**2) + inv_control_avg + np.mean([1/probability[x][t] for x in X_0]))
    # print('std',std)

    CI_width = {}
    for t in range(1,2**M):
        CI_width[t] = 1.96*std[t]/np.sqrt(N)

    # print('CI_width',CI_width)
    return CI_width

#compute coverage 
def coverage_compute(AIPW_results,estimators,ATE_real,true_prob,M=4,N=200,T=100,
                     X_0=np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])):
    CI_width = semiparametric_efficient_CI(true_prob,N*0.5,M,X_0)
    coverage = {estimator: {t: [] for t in range(1, 2**M)} for estimator in estimators}
    
    for estimator in estimators:
        for t in range(1, 2**M):
            coverage[estimator][t] = 0.0
    
    for estimator in estimators:
        for i in range(T):
            upper = {}
            lower = {}
            for t in range(1,2**M):
                aipw_predict = AIPW_results[estimator][t][i]
                upper[t] = aipw_predict + CI_width[t]
                lower[t] = aipw_predict - CI_width[t]
                if (ATE_real[t]>=lower[t]) & (ATE_real[t]<=upper[t]):
                    coverage[estimator][t]+=1
            
    for estimator in estimators:
        for t in range(1,2**M):
            coverage[estimator][t]=coverage[estimator][t]/T

    return coverage 
    

def plot_ate_confidence_intervals(AIPW_results, estimators, real_ATE,s,M,N,n):
    treatment_levels = range(1, 16)
    # estimator_colors = plt.cm.viridis(np.linspace(0, 1, len(estimators)))  # Color map for the estimators
    estimator_colors = ['purple','blue','#FFC107','green','#32CD32']
    plt.figure(figsize=(14, 8))

    for t in treatment_levels:
        positions = np.arange(len(estimators)) + (t - 1) * (len(estimators) + 1)  # Position estimators side by side
        means_list = []
        lower_cis = []
        upper_cis = []

        # Compute the means and confidence intervals for each estimator at treatment level t
        for i, estimator in enumerate(estimators):
            means, stds, cis = compute_confidence_intervals(AIPW_results[estimator],n)
            mean_val = means[t]
            lower_ci, upper_ci = cis[t]

            means_list.append(mean_val)
            lower_cis.append(lower_ci)
            upper_cis.append(upper_ci)

            # Plot the confidence interval as error bars
            plt.errorbar(t + i * 0.1, mean_val, 
                         yerr=[[mean_val - lower_ci], [upper_ci - mean_val]],
                         fmt='o', color=estimator_colors[i], label=estimator if t == 1 else "", capsize=5)

        # Plotting the dashed horizontal line for the real ATE at treatment level t
        plt.hlines(real_ATE[t], t - 0.3, t + 0.5, colors='red', linestyles='dashed', linewidth=2)

    # Adding title, labels, and legend
    plt.title('ATE Confidence Intervals for all estimators (s=%i,M=%i,N=%i)'%(s,M,N), fontsize=14)
    plt.xlabel('Treatment Level', fontsize=12)
    plt.ylabel('ATE Estimate', fontsize=12)
    plt.xticks(ticks=np.arange(1, 16), labels=np.arange(1, 16), fontsize=12)
    
    # Show the legend for the estimators (only for the first treatment level)
    plt.legend(loc='upper left')
    
    # Show plot
    plt.show()

def coverage_heatmap(estimator, coverage, colors, M=4):
    # t = 1, ..., 2^M
    t_values = range(1, 2**M)  
    s_values = [0, 2, 5, 10]  # s values
    N_values = [200, 300, 400]  # N values
    
    morandi_green_cmap = LinearSegmentedColormap.from_list("morandi_green", colors, N=256)

    # Mock data: Replace with real data
    color_dict = {
        (s, N): {t: coverage[(s, N)][estimator][t] for t in t_values}
        for s in s_values
        for N in N_values
    }

    # Prepare a grid of 3x5 for the heatmaps
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()

    # Plot heatmaps for each t
    for i, t in enumerate(t_values):
        # Extract data for heatmap
        heatmap_data = np.array([[color_dict[(s, N)][t] for s in s_values] for N in N_values])
        
        # Convert t to binary and format as (0, 0, 0, 1)
        binary_t = format(t, f'0{M}b')
        binary_t_formatted = f"({', '.join(binary_t)})"
        
        # Plot on the respective subplot
        ax = axes[i]
        cax = ax.imshow(heatmap_data, cmap=morandi_green_cmap, aspect='auto', origin='lower')
        ax.set_xticks(range(len(s_values)))
        ax.set_xticklabels(s_values, fontsize=8)
        ax.set_yticks(range(len(N_values)))
        ax.set_yticklabels(N_values, fontsize=8)
        ax.set_xlabel('$s$', fontsize=10)
        ax.set_ylabel('$N$', fontsize=10)
        ax.set_title(f'$t={binary_t_formatted}$', fontsize=12)

        # Add text labels for each block
        for y in range(heatmap_data.shape[0]):
            for x in range(heatmap_data.shape[1]):
                value = heatmap_data[y, x]
                ax.text(x, y, f'{value:.2f}', ha='center', va='center', fontsize=8, color='black')

    # Remove unused subplots if t_values is less than 15
    for i in range(len(t_values), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout to avoid overlaps
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)  # Add space for the main title

    # Add a single colorbar
    if estimator == 'NW':
        est = 'Nadaraya-Watson'
    elif estimator == 'PI_0':
        est = 'Plug-in'
    elif estimator == 'FO_0':
        est = 'First-order'
    elif estimator == 'multinomial':
        est = 'Multinomial Discrete Choice Modeling'
    
    cbar = fig.colorbar(cax, ax=axes, orientation='horizontal', fraction=0.05, pad=0.04, label='Coverage Value')
    plt.suptitle(f'Coverage Heatmaps for {est} Estimator', fontsize=16, y=0.97)
    plt.show()


def plot_ate_CI_new_estimators(AIPW_results, estimators, real_ATE,s,M,N,n):
    treatment_levels = range(1, 16)
    # estimator_colors = plt.cm.viridis(np.linspace(0, 1, len(estimators)))  # Color map for the estimators
    estimator_colors = ['blue','#FFC107','green','#32CD32']
    plt.figure(figsize=(14, 8))

    for t in treatment_levels:
        positions = np.arange(len(estimators)) + (t - 1) * (len(estimators) + 1)  # Position estimators side by side
        means_list = []
        lower_cis = []
        upper_cis = []
        print('treatment_level: ',t)

        # Compute the means and confidence intervals for each estimator at treatment level t
        for i, estimator in enumerate(estimators):
            means, stds, cis = compute_confidence_intervals(AIPW_results[estimator],n)
            mean_val = means[t]
            lower_ci, upper_ci = cis[t]
            print('estimator: ',estimator)
            print('lower_ci: ', lower_ci)
            print('upper_ci: ', upper_ci)

            means_list.append(mean_val)
            lower_cis.append(lower_ci)
            upper_cis.append(upper_ci)


            # Plot the confidence interval as error bars
            plt.errorbar(t + i * 0.1, mean_val, 
                         yerr=[[mean_val - lower_ci], [upper_ci - mean_val]],
                         fmt='o', color=estimator_colors[i], label=estimator if t == 1 else "", capsize=5)

        # Plotting the dashed horizontal line for the real ATE at treatment level t
        plt.hlines(real_ATE[t], t - 0.3, t + 0.5, colors='red', linestyles='dashed', linewidth=2)

    # Adding title, labels, and legend
    plt.title('ATE Confidence Intervals for all estimators (s=%i,M=%i,N=%i)'%(s,M,N), fontsize=14)
    plt.xlabel('Treatment Level', fontsize=12)
    plt.ylabel('ATE Estimate', fontsize=12)
    plt.xticks(ticks=np.arange(1, 16), labels=np.arange(1, 16), fontsize=12)
    
    # Show the legend for the estimators (only for the first treatment level)
    plt.legend(loc='upper left')
    
    # Show plot
    plt.show()

