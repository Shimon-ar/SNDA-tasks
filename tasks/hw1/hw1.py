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
        prob_even = 0

    for k in range(0, n + 1, 2):  # Only even k values
        prob_even += binom.pmf(k, n, p)  # P(X = k)
    
    return prob_even


def evenBinomFormula(n, p):
    """
    The program outputs the probability P(X\ is\ even) for the random variable X~Binom(n, p) Using a closed-form formula.
    It should also print the proof for the formula.

    Input:
    - n, p: The parameters for the binomial distribution.

    Returns:
    - prob: The output probability.
    """
    # Use the direct formula for the probability that X is even
    print("P(X is even) can be expressed in terms of n and p, givem by the binomial law: Binom(n,p)\n")
    # not finished
    return (1 + (1 - 2 * p) ** n) / 2



### Question 5 ###

def three_RV(X, Y, Z, joint_probs):
    """
    Input:
    - X: 2d numpy array: [[values], [probabilities]].
    - Y: 2d numpy array: [[values], [probabilities]].
    - Z: 2d numpy array: [[values], [probabilities]].
    - joint_probs: 3d numpy array: joint probability of X, Y and Z.

    Returns:
    - v: The variance of X + Y + Z.
    """

    # Compute the expected values of X, Y, and Z with the classic definition: E[X] = ∑ xi * P(X=xi) = ∑ [value i] * [proba i]
    EX = np.sum(X[0] * X[1])
    EY = np.sum(Y[0] * Y[1])
    EZ = np.sum(Z[0] * Z[1])

    
    # Compute E[X**2]= ∑ xi**2 * P(X=xi)
    EX2 = np.sum((X[0] ** 2) * X[1])
    EY2 = np.sum((Y[0] ** 2) * Y[1])
    EZ2 = np.sum((Z[0] ** 2) * Z[1])
    
    # Compute the variances of X, Y, and Z: Var[X] = E[X**2] - E[X]**2
    VarX = EX2 - EX**2
    VarY = EY2 - EY**2
    VarZ = EZ2 - EZ**2


    # Compute the covariances between pairs (X, Y), (X, Z), (Y, Z)
    # By def, COV[X,Y] = E[X*Y] − E[X]*E[Y]

    # Compute E[X*Y] = ∑ xi * yi * P(X=xi, Y=yi)

    # P(X=xi, Y=yi)
    # join_probs: P(X=xi, Y=yi, Z=zi)
    # we want P(X=xi, Y=yi), that can be write with the join_probs: P(X=xi, Y=yi) = ∑ P(X=xi, Y=yi, Z=zi), ∑ on all the zi
    P_XY = np.sum(joint_probs, axis=2) # axis = 2 , collapse Z (3rd elem of the join proba)

    # xi * yi
    # np.outer(X[0], Y[0]) produces a the matrix where each element is (xi * yi)  (2d)
    EXY = np.sum(P_XY * np.outer(X[0], Y[0]))
    
    #We now do the same for (X,Z) and (Y,Z):
    P_XZ = np.sum(joint_probs, axis=1)  # axis = 1 , collapse Y (2nd elem)
    EXZ = np.sum(P_XZ * np.outer(X[0], Z[0]))

    P_YZ = np.sum(joint_probs, axis=0)  # collapse X (first elem)
    EYZ = np.sum(P_YZ * np.outer(Y[0], Z[0]))


    # Covariance 
    CovXY = EXY - EX * EY
    CovXZ = EXZ - EX * EZ
    CovYZ = EYZ - EY * EZ


    # VAR(X+Y+Z) = Var(X) + Var(Y) + Var (Z) + 2*Cov(X,Y) + 2*Cov(XZ) + 2*Cov(YZ)
    return VarX + VarY + VarZ + 2 * CovXY + 2 * CovXZ + 2 * CovYZ

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

    # Calculate the variance of the sum X + Y + Z
    # Pairwise independent means that the covariance between any two of them is zero
    
    # Then VAR(X+Y+Z) = Var(X) + Var(Y) + Var (Z) + 2*Cov(X,Y) + 2*Cov(XZ) + 2*Cov(YZ) become VAR(X+Y+Z) = Var(X) + Var(Y) + Var (Z)
    
    # By the same idea of the last function, we need to:

    # Compute the expected values of X, Y, and Z with the classic definition: E[X] = ∑ xi * P(X=xi) = ∑ [value i] * [proba i]
    EX = np.sum(X[0] * X[1])
    EY = np.sum(Y[0] * Y[1])
    EZ = np.sum(Z[0] * Z[1])

    
    # Compute E[X**2]= ∑ xi**2 * P(X=xi)
    EX2 = np.sum((X[0] ** 2) * X[1])
    EY2 = np.sum((Y[0] ** 2) * Y[1])
    EZ2 = np.sum((Z[0] ** 2) * Z[1])
    
    # Compute the variances of X, Y, and Z: Var[X] = E[X**2] - E[X]**2
    VarX = EX2 - EX**2
    VarY = EY2 - EY**2
    VarZ = EZ2 - EZ**2

    return VarX + VarY + VarZ

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
    # pairwise: P(X=x,Y=y)=P(X=x)⋅P(Y=y)  /  P(Z=z,Y=y)=P(Z=z)⋅P(Y=y)  /   P(X=x,Z=z)=P(X=x)⋅P(Z=z)

    # collectivity: P(X=x,Y=y,Z=z)=P(X=x)⋅P(Y=y)⋅P(Z=z)
    
    # extract marginal probabilities of X, Y and Z from the join_proba
    P_X = np.sum(joint_probs, axis=(1, 2))  # axis=(1,2), bc we sum over Y and Z
    P_Y = np.sum(joint_probs, axis=(0, 2))  
    P_Z = np.sum(joint_probs, axis=(0, 1)) 

    for i, xi in enumerate(X[0]):  # enumerate give position, value  ->  use i to access to corresponding elements in other arrays like P_X[i] = P(X = xi) for that i
        for j, yj in enumerate(Y[0]):
            for k, zk in enumerate(Z[0]):
                # P(X = xi, Y = yj, Z = zk)
                P_join = joint_probs[i, j, k]

                # P(X = xi)*P(Y = yj)*P(Z = zk)
                P_sep = P_X[i] * P_Y[j] * P_Z[k]

                # P(X = xi, Y = yj, Z = zk) =? P(X = xi)*P(Y = yj)*P(Z = zk)
                if not np.isclose(P_join, P_sep):
                    return False  # we dont have the collectively independent

    # P(X = xi, Y = yj, Z = zk) = P(X = xi)*P(Y = yj)*P(Z = zk)  for every i, then pairwise => collectivity
    return True


### Question 6 ###

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.
    """
    def factorial(n):
    fact = 1
    for i in range(1, n + 1):
        fact *= i
    return fact

    exp = 0
    for k in range(n+1):
        c = ( factorial(n) ) / ( (factorial(n-k) * factorial(k) ) )
        proba = c * (p**k) * ((1-p)**(n-k))
        exp += (c * proba)

    return exp 
