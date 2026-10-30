# Methods Code

import random
import numpy as np

def prepare_data(data):

    """ Filters, pairs, and randomizes data"""
    
    # Take only subjects with >$50k of income (true positives)
    positive_data = data[data['income'] == 1]
    cond_A = positive_data[positive_data['sex'] == 'Male'].sample(frac=1, random_state=42).reset_index(drop=True)
    cond_B = positive_data[positive_data['sex'] == 'Female'].sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Test on matched pairs to compare the True Positive Rate fairly between both sexes
    n_pairs = min(len(cond_A), len(cond_B))

    return cond_A, cond_B, n_pairs
    

def seq_hyp_test(alpha, data, model):
    """
    Squential Hypothesis Testing Algorithm
    Returns true if the null hypothesis is rejected, returns false otherwise.

    Hypotheses:
        H0 (Null): TPR_Male = TPR_Female (No difference in True Positive Rates)
        H1 (Alt):  TPR_Male != TPR_Female (Significant difference in True Positive Rates)
    """
    cond_A, cond_B, n_pairs = prepare_data(data)
    
    K = 1
    tp_A = 0
    tp_B = 0
    lambda_curr = 0
    z = []

    # Beginning of actual sequential hypothesis testing algorithm
    for i in range(n_pairs):

        obs_A = cond_A[i]
        obs_B = cond_B[i]

        if model.predict(obs_A) == 1:
            tp_A += 1
        if model.predict(obs_B) == 1:
            tp_B += 1

        tpr_A = tp_A / (i + 1)
        tpr_B = tp_B / (i + 1)

        # Update wealth
        S = 1 + lambda_curr * (tpr_A - tpr_B)
        K = K*S

        # Online Newton Step (ONS)
        g = tpr_A-tpr_B
        z_t = g / (1 - lambda_curr * g)
        z.append(z_t)

        # Calculate the next lambda, clipped between -0.5 and 0.5
        lambda_next = np.clip(
            (2 / (2 - np.log(3))) *
            (z_t / (1 + np.sum(np.array(z)**2))) - lambda_curr,
            -0.5,
            0.5
        )
        lambda_curr = lambda_next

        if K >= 1 / alpha:
            return True
    
    U = random.random()
    if K >= U / alpha:
        return True
    return False