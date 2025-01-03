###### Your ID ######
# ID1: 312119126
# ID2: 209876846
#####################

# imports 
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


### Question 2 ###

def q2a(X, Y, Z):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    The number of parameters that define the joint distribution of X, Y and Z.
    """

    n = X.shape[1]  # gives the number of possible values for X
    m = Y.shape[1]
    k = Z.shape[1]

    # The join distribution is described by the P(X=x, Y=y, Z=z) probabilities for every n, m and k values that X, Y and Z can take
    # There is n*k*m total combinaisons
    # But, the sum of all thoses probabilites needs to be equal to 1 to respect the definition of a probability function
    # Then we only need to define n*m*k -1 parameters, since the last one is determined by the contraint of the sum equal to 1
   
    return (n*m*k)-1

def q2b(X, Y, Z):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """

    n = X.shape[1]
    m = Y.shape[1]
    k = Z.shape[1]

    # The variables are independant, then P(X=x, Y=y, Z=z) = P(X=x)P(Y=y)P(Z=z)
    # So we can define the marginale distributions separetly
    # As the same idea as Q2a, the sum of the probabilities needs to be 1,
    # then for X that takes n values: we define n-1 probabilities,
    # the last is determined by the contraint of the sum equal to 1
    # Same for Y and Z

    return (n-1)+(m-1)+(k-1)


def q2c(X, Y, Z):
    """
 
    Input:          
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    
    Returns:
    The number of parameters that define the joint distribution of X, Y and Z if we know that they are independent.
    """

    n = X.shape[1]
    m = Y.shape[1]
    k = Z.shape[1]

    # X, Y are conditionally independant given Z, then P(X=x, Y=y, Z=z) = P(X=x|Z=z)P(Y=y|Z=z)P(Z=z)
    # As the same idea, as q2b: For Z, we denike k-1 probabilities, then parameters
    # For P(X=x|Z=z), we have k values for Z and we need to define the probability for the values of X
    # the sum of the probabilities needs to be 1,
    # then for X that takes n values: we define n-1 probabilities,
    # the last is determined by the contraint of the sum equal to 1, as seen
    # so we have (n-1) parameters for each z (and we have k possibilities for z): then k(n-1) parameters for P(X=x|Z=z)
    # Same for P(Y=y|Z=z): k*(m-1)


    return k - 1 + k*(n-1) + k*(m-1)


### Question 3 ###

def my_EM(mus, sigmas, ws, known_params=None, eps=0.000001):
    """
 
    Input:          
    - mus   : a numpy array: holds the initial guess for means of the Gaussians.
    - sigmas: a numpy array: holds the initial guess for std of the Gaussians.
    - ws    : a numpy array: holds the initial guess for weights of the Gaussians.
    
    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).
    
    Returns:
    The output of the EM algorithms (the GMM final parameters): mus, sigmas, ws.
    """

    def get_known_params(index):
        mu, sigma, w = None, None, None

        if known_params is not None and 0 <= index < len(known_params):
            params = known_params[index]
            mu = params.get('mu')
            sigma = params.get('sigma')
            w = params.get('w')

        return mu, sigma, w

    def update_params(params, type='mu'):
        for i in range(len(params)):
            mu_k, sigma_k, w_k = get_known_params(i)
            if type == 'mu':
                if mu_k is not None:
                    params[i] = mu_k
            elif type == 'sigma':
                if sigma_k is not None:
                    params[i] = sigma_k
            elif type == 'w':
                if w_k is not None:
                    params[i] = w_k
        return params

    data = pd.read_csv('GMD.csv', header=None).values[:, 1]
    likelihood = []

    while True:
        probs = []
        for k in range(len(mus)):
            mu_k, sigma_k, w_k = get_known_params(k)
            mu = mus[k] if mu_k is None else mu_k
            sigma = sigmas[k] if sigma_k is None else sigma_k
            w = ws[k] if w_k is None else w_k

            prob = stats.norm.pdf(data, loc=mu, scale=sigma) * w
            probs.append(prob)

        probs = np.transpose(probs)
        sums = np.sum(probs, axis=1)[:, np.newaxis]
        r = probs / sums

        ws = update_params(np.mean(r, axis=0), 'w')

        mus = update_params(np.sum(r * data[:, np.newaxis], axis=0) * (1 / np.sum(r, axis=0)), 'mu')
        sigmas = update_params(
            np.sqrt(np.sum(r * (data[:, np.newaxis] - mus) ** 2, axis=0) * (1 / np.sum(r, axis=0))), 'sigma')

        likelihood.append(np.sum(np.log(np.sum(probs, axis=1))))
        if len(likelihood) > 1 and likelihood[-1] - likelihood[-2] < eps:
            break

    return mus, sigmas, ws


def q3d(mus, sigmas, ws, data_points=1000):
    """
 
    Input:          
    - mus   : a numpy array: holds the means of the gaussians.
    - sigmas: a numpy array: holds the stds of the gaussians.
    - ws    : a numpy array: holds the weights of the gaussians.
    
    * The function should be generic and support any number of Gaussians.
      (you don't need to get this number as a parameter for the function. You can conclude it from the other parameters).
    
    Returns:
    The generated data.
    """

    gmm_choices = np.random.choice(len(mus), size=data_points, p=ws)

    data = np.zeros(data_points)
    for i, choice in enumerate(gmm_choices):
        data[i] = np.random.normal(loc=mus[choice], scale=sigmas[choice])

    return data


### Question 4 ###

def q4a(mu=75000, sigma=37500, salary=50000):
    """
 
    Input:          
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The percent of people earn less than 'salary'.
    """

    return stats.norm.cdf(salary, loc=mu, scale=sigma) * 100


def q4b(mu=75000, sigma=37500, min_salary=45000, max_salary=65000):
    """
 
    Input:          
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The percent of people earn between 'min_salary' and 'max_salary'.
    """

    return (stats.norm.cdf(max_salary, loc=mu, scale=sigma) - stats.norm.cdf(min_salary, loc=mu, scale=sigma)) * 100


def q4c(mu=75000, sigma=37500, salary=85000):
    """
 
    Input:          
    - mu   : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma: The std of the annual salaries of employees in a large Randomistan company.
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The percent of people earn more than 'salary'.
    """

    return (1 - stats.norm.cdf(salary, loc=mu, scale=sigma)) * 100


def q4d(mu=75000, sigma=37500, salary=140000, n_employees=1000):
    """
 
    Input:          
    - mu         : The mean of the annual salaries of employees in a large Randomistan company.
    - sigma      : The std of the annual salaries of employees in a large Randomistan company.
    - n_employees: The number of employees in the company
    The annual salary of employees in a large Randomistan company are approximately normally distributed.
    
    Returns:
    The number of employees in the company that you expect to earn more than 'salary'.
    """

    return (1 - stats.norm.cdf(salary, loc=mu, scale=sigma)) * n_employees


### Question 5 ###

def CC_Expected(N=10):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    E(T_N)
    """

    # By class3, we have the definition E(T_N) = N * ∑ 1/i  - for i=1,...N

    return N * np.sum(1 / np.arange(1, N + 1))


def CC_Variance(N=10):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    V(T_N)
    """

    # By class3, we have the definition V(T_N) = N^2 * ∑ (1/i^2) - N * H(N)   - for i=1,...N

    h_s = np.sum([1 / i**2 for i in range(1, N + 1)])
    return N**2 * h_s - CC_Expected(N)


def CC_T_Steps(N=10, n_steps=30):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    The probability that T_N > n_steps
    """

    #E(T_N):
    mean = CC_Expected(N=10)

    #VAR(T_N)
    var = CC_Variance(N=10)

    # If n_steps ≤ E(T_N), return 1 (impossible to exceed)
    if n_steps <= mean:
        return 1.0

    # Calculation of the limit with Chebyshev
    cN = n_steps - mean
    return var / cN**2


def CC_S_Steps(N=10, n_steps=30):
    """
 
    Input:          
    - N: Number of different coupons.
    
    Returns:
    The probability that S_N > n_steps
    """

    # Sn = waiting time until different coupons are collected twice.
    # We use an approach by convultion
    # We have: P(S_N = k) = ∑ P(G_N = i)* P(S_N = k-i)  -  sum for i=1,...k-1
    # where G_N is the time to get the N_th coupon: We init P(T_1 = k), ) for all other values
    # G_i ˜ Geo(p = (N-i+1) /N)

    P_S = np.zeros(n_steps + 1)
    P_S[0] = 1.0  # S_0 = 1

    # We calculate P(S_N > n_steps) recursively:

    for i in range(1, N + 1):

        P_new = np.zeros(n_steps + 1)
        p = i / N  # P of succes of the i-e coupon - more you go, more chnace you have to get a coupon

        for k in range(1, n_steps + 1):  # calculation for the n_steps of P(S_N = k)
            for j in range(1, k + 1):  # Sommation - waiting time for coupon i
                if k - j >= 0:
                    P_new[k] = P_new[k] + stats.geom.pmf(j, p) * P_S[k - j]   # P(S_N = k)  = ∑ P(G_N = i)* P(S_N = k-i)

        P_S = P_new  # dist of the new coupon - P(S_N = k)

    # P(S_N > n_steps) = 1 - P(S_N ≤ n_steps) (the sum of all P(S_N = k; k≤ n_steps) )
    return 1.0 - np.sum(P_S[:n_steps + 1])

