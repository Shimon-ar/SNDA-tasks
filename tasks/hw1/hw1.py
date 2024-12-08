###### Your ID ######
# ID1:
# ID2:
#####################

# imports
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


### Question 1 ###

def find_sample_size_binom(min_defective=1, defective_rate=0.05, p_n_defective_ts=0.85):
    """
    Using Binom to returns the minimal number of samples required to have requested probability of receiving
    at least x defective products from a production line with a defective rate.
    """
    n = 1
    while True:
        p_at_most_x_defective = stats.binom.cdf(min_defective - 1, n, defective_rate)
        p_at_least_x_defective = 1 - p_at_most_x_defective
        if p_at_least_x_defective > p_n_defective_ts:
            return n
        n += 1


def find_sample_size_nbinom(min_defective=1, defective_rate=0.05, p_n_defective_ts=0.85):
    """
    Using NBinom to returns the minimal number of samples required to have requested probability of receiving
    at least x defective products from a production line with a defective rate.
    """
    n = min_defective
    while True:
        p_n_trials = stats.nbinom.cdf(n - min_defective, min_defective, defective_rate)
        if p_n_trials > p_n_defective_ts:
            return n
        n += 1


def compare_q1():
    n_independent_samples_first_part = find_sample_size_binom(min_defective=5, defective_rate=0.1, p_n_defective_ts=0.9)
    n_independent_samples_second_part = find_sample_size_binom(min_defective=15, defective_rate=0.3,
                                                               p_n_defective_ts=0.9)
    return n_independent_samples_first_part, n_independent_samples_second_part


def same_prob():
    n = 15
    while True:
        p_first_part = stats.nbinom.cdf(n - 5, 5, 0.1)
        p_second_part = stats.nbinom.cdf(n - 15, 15, 0.3)
        if np.isclose(p_first_part, p_second_part, atol=1e-2):
            return n
        n += 1


### Question 2 ###

def empirical_centralized_third_moment(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=100, seed=None):
    """
    Create k experiments where X is sampled. Calculate the empirical centralized third moment of Y based
    on your k experiments.
    """
    if seed:
        np.random.seed(seed)
    X = np.random.multinomial(n, p, size=k)
    Y = X[:, 1] + X[:, 2] + X[:, 3]
    E_Y = np.mean(Y)
    empirical_moment = np.mean((Y - E_Y) ** 3)
    return empirical_moment

# print(empirical_centralized_third_moment(n=20, p=[0.2, 0.1, 0.1, 0.1, 0.2, 0.3], k=1000, seed=None))

def class_moment(n=20, p=0.3):
    moment = n * p * (1 - p) * (1 - 2 * p)
    return moment

# print(class_moment(n=20, p=0.3))


def plot_moments(n=20):

    data = [empirical_centralized_third_moment(n) for k in range(1000)]

    plt.hist(data, bins=20)
    plt.axvline(class_moment(n), color='red')

    plt.xlabel('Moments')
    plt.ylabel('Experiment')
    plt.show()

    dist_var = np.var(data)
    return dist_var


def plot_moments_smaller_variance():
    dist_var = plot_moments(5)
    return dist_var


### Question 3 ###

def NFoldConv(P, n):
    """
    Calculating the distribution, Q, of the sum of n independent repeats of random variables,
    each of which has the distribution P.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    - n: An integer.

    Returns:
    - Q: 2d numpy array: [[values], [probabilities]].
    """

    Q = P
    for _ in range(n - 1):
        sum_dist = []
        for j in range(len(Q[0])):
            for k in range(len(P[0])):
                sum_dist.append([Q[0][j] + P[0][k], Q[1][j] * P[1][k]])
        sum_dist = np.array(sum_dist).T

        unique_elements, inverse_indices = np.unique(sum_dist[0], return_inverse=True)
        new_dist_p_sum = np.zeros_like(unique_elements, dtype=float)

        for i, _ in enumerate(unique_elements):
            new_dist_p_sum[i] = np.sum(sum_dist[1][inverse_indices == i])
        Q = np.array([unique_elements, new_dist_p_sum])
    return Q



def plot_dist(P):
    """
    Ploting the distribution P using barplot.

    Input:
    - P: 2d numpy array: [[values], [probabilities]].
    """



    categories = ['A', 'B', 'C', 'D']
    values = [4, 7, 1, 8]

    # Create the bar plot
    plt.bar(P[0], P[1])

    # Add labels and title
    plt.xlabel('Distribution')
    plt.ylabel('Probability')
    plt.show()

# Q = NFoldConv(np.array([[1, 2,5], [0.1, 0.5, 0.4]]), 2)
# plot_dist(Q)

### Qeustion 4 ###

def evenBinom(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p).

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """

    return prob


def evenBinomFormula(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p) Using a closed-form formula.
    It should also print the proof for the formula.

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """

    return prob


### Question 5 ###

def three_RV(X, Y, Z, joint_probs):
    """

    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    - joint_probs: 3d numpy array: joint probability of X, Y and Z

    Returns:
    - v: The variance of X + Y + Z.
    """

    return v


def three_RV_pairwise_independent(X, Y, Z, joint_probs):
    """

    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    - joint_probs: 3d numpy array: joint probability of X, Y and Z

    Returns:
    - v: The variance of X + Y + Z.
    """

    return v


def is_pairwise_collectively(X, Y, Z, joint_probs):
    """

    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    - joint_probs: 3d numpy array: joint probability of X, Y and Z

    Returns:
    TRUE or FALSE
    """

    pass


### Question 6 ###

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.
    """

    pass
