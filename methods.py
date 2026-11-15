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

def perm_test_m1(alpha, data, model, k):

    """
    Permuation Testing Method with No Correction for Repeated Tests
    """

    cond_A, cond_B, n_pairs = prepare_data(data)

    n = min(len(cond_A), len(cond_B))

    for i in range(k, n+1, k):

        sample_A = cond_A.iloc[:i]
        sample_B = cond_B.iloc[:i]

        pred_A = model.predict(sample_A)
        pred_B = model.predict(sample_B)

        tpr_A = np.mean(pred_A)
        tpr_B = np.mean(pred_B)

        # Permutation test
        p_value = permutation_test(pred_A, pred_B)

        if p_value <= alpha:
            return True

    return False

def perm_test_m2(alpha, data, model, k):

    """
    Permuation Testing Method with Bonferroni Correction for Repeated Tests
    """

    cond_A, cond_B, n_pairs = prepare_data(data)

    n = min(len(cond_A), len(cond_B))

    for j, i in enumerate(range(k, n + 1, k), start=1):

        sample_A = cond_A.iloc[:i]
        sample_B = cond_B.iloc[:i]

        pred_A = model.predict(sample_A)
        pred_B = model.predict(sample_B)

        tpr_A = np.mean(pred_A)
        tpr_B = np.mean(pred_B)

        # Permutation test
        p_value = permutation_test(pred_A, pred_B)

        #Corrected Significance Level
        alpha_j = alpha / (2 ** j)

        if p_value <= alpha_j:
            return True

    return False

def permutation_test(pred_A, pred_B, n_permutations=1000):

    """
    Permutation Testing Method

    Combine the predictions from both groups, shuffle the predictions, and split them into two new groups
    Calculate the permutations test statistic, record it in an array, and compare to observed statistic to obtain p-value
    """

    observed_diff = abs(np.mean(pred_A) - np.mean(pred_B))

    # Combine predictions from both groups
    combined = np.concatenate([pred_A, pred_B])
    n = len(pred_A)

    perm_diffs = []

    for _ in range(n_permutations):

        shuffled = np.random.permutation(combined)

        perm_A = shuffled[:n]
        perm_B = shuffled[n:]

        perm_diff = abs(np.mean(perm_A) - np.mean(perm_B))

        perm_diffs.append(perm_diff)

    # Monte Carlo p-value
    p_value = (
        np.sum(np.array(perm_diffs) >= observed_diff) + 1
    ) / (n_permutations + 1)

    return p_value