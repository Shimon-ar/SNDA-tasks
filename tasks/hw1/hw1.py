###### Your ID ######
# ID1:
# ID2:
#####################

# imports
import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt


def print_in_plot(steps):
    fig, ax = plt.subplots(figsize=(10, 8))
    for text, ypos, fontsize in steps:
        ax.text(0.5, ypos, text, fontsize=fontsize, ha='center', va='center', fontweight='normal')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


### Question 1 ###

def find_sample_size_binom(min_defective=1, defective_rate=0.05, p_n_defective_ts=0.85, print_answer=True):
    """
    Using Binom to returns the minimal number of samples required to have requested probability of receiving
    at least x defective products from a production line with a defective rate.
    """

    steps = [
        (r"QUESTION 1.A:", 0.9, 18),
        (r"Using the binomial distribution: X ~ Binom(n, 0.03)", 0.8, 15),
        (r"Where X is the number of defective products in n independent samples", 0.75, 15),
        (r"With defective probability of 3%", 0.71, 15),
        (r"We want to find n which $P(X \geq 1) = 0.85$", 0.65, 15),
        (r"$P(X \geq 1) = 1 - P(X = 0) = 1 - 0.97^n$", 0.58, 14),
        (r"Then $0.85 = 1 - 0.97^n$ so $n = 62.2$", 0.53, 14),
        (r"Hence we will need 63 samples to have probability of 85%", 0.45, 16),
        (r"of at least 1 defective product", 0.42, 16)
    ]
    if print_answer:
        print_in_plot(steps)

    n = max(min_defective - 1, 1)
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
    n_independent_samples_first_part = find_sample_size_binom(min_defective=5, defective_rate=0.1, p_n_defective_ts=0.9,
                                                              print_answer=False)
    n_independent_samples_second_part = find_sample_size_binom(min_defective=15, defective_rate=0.3,
                                                               p_n_defective_ts=0.9, print_answer=False)
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


def class_moment(n=20, p=0.3):
    moment = n * p * (1 - p) * (1 - 2 * p)
    return moment


def plot_moments(n=20, k=1000):
    data = [empirical_centralized_third_moment(n) for _ in range(k)]

    plt.title('Empirical Centralized Third Moment')
    plt.hist(data, bins=20)
    plt.axvline(class_moment(n), color='red')
    plt.xlabel('Moments')
    plt.ylabel('Experiment')
    plt.show()

    dist_var = np.var(data)
    return dist_var


def plot_moments_smaller_variance():
    steps = [
        (r"Question 2 Smaller variance:", 0.9, 18),
        (r"As $n$ gets bigger, the count for each $X_i$ will increase", 0.85, 15),
        (r"thus the distance from the expected value will also increase", 0.8, 15),
        (r"which will affect the variance to be bigger.", 0.75, 15),
    ]
    print_in_plot(steps)
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

    """
    Each iteration we calculate the convolution of the current distribution with P.
    This true because we have n independent repeats of random variables, each of which has the distribution P.
    in each iteration the following steps are occurred:
    sum_dist is a 2d array which each row holds the sum and the distribution for it
    Next we transform the sum_dist so the first row will hold the unique sums and the second row will hold the probabilities for each sum
    then we get the unique elements and the inverse indices to know the position of each element in the unique elements array
    then we calculate the new distribution for the sum of the two distributions
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
    plt.title('Q3 Distribution of P')
    plt.bar(P[0], P[1])
    plt.xlabel('Distribution')
    plt.ylabel('Probability')
    plt.show()


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
        prob_even += stats.binom.pmf(k, n, p)  # P(X = k)

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
    prove = [
        (r"QUESTION 4 - Prove:", 0.9, 18),
        (r"P(X is even) can be expressed in terms of n and p, given by the binomial law: X ~ Binom(n,p)", 0.8, 15),
        (r"Using the generative function for the Binomiale distribution, we get the following:", 0.7, 15),
        (
            r"$\left((1-p) + p\right)^n = 1 = \sum_{k=0}^{n} \binom{n}{k} p^k (1-p)^{n-k} = P(X=0) + P(X=1) + \ldots + P(X=n) = $ P(X is Even) + P(X is Odd)",
            0.6, 14),
        (
            r"$\left((1-p) - p\right)^n = 1 = \sum_{k=0}^{n} \binom{n}{k} (-p)^k (1-p)^{n-k} = P(X=0) - P(X=1) + P(X=2) \ldots = $ P(X is Even) - P(X is Odd)",
            0.5, 14),
        (r"Then the sum of this two expression gives us:", 0.4, 15),
        (r"2*P(X is even) = $\left((1-p) - p\right)^n + \left((1-p) + p\right)^n = 1 + (1 - 2p)^n$", 0.3, 16),
        (r"And finally we get: P(X is even) = $\frac{1 + (1 - 2p)^n}{2}$", 0.2, 16)
    ]
    print_in_plot(prove)

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

    steps = [
        (r"QUESTION 5A:", 0.9, 18),
        (r"$V(X+ (Y + Z)) = V(X) + V(Y + Z) + 2COV(X, Y + Z)$", 0.8, 15),
        (r"$ = V(X) + V(Y) + V(Z) + 2COV(Y, Z) + 2COV(X, Y + Z)$", 0.7, 15),
        (r"Using the linearity of expectation we get:", 0.6, 15),
        (r"$COV(X, Y + Z) = E(XY + XZ) - E(X)*E(Y+Z)$", 0.5, 15),
        (r"$= E(XY) + E(XZ) - E(X)*E(Y) - E(X)*E(Z)) = COV(X,Y) + COV(X,Z)$", 0.4, 15),
        (r"Then:", 0.3, 15),
        (r"$V(X + Y + Z) = V(X) + V(Y) + V(Z) + 2COV(Y, Z) + 2COV(X,Y) + 2COV(X,Z)$", 0.2, 15)
    ]

    print_in_plot(steps)

    EX = np.sum(X[0] * X[1])
    EY = np.sum(Y[0] * Y[1])
    EZ = np.sum(Z[0] * Z[1])

    EX2 = np.sum((X[0] ** 2) * X[1])
    EY2 = np.sum((Y[0] ** 2) * Y[1])
    EZ2 = np.sum((Z[0] ** 2) * Z[1])

    VarX = EX2 - EX ** 2
    VarY = EY2 - EY ** 2
    VarZ = EZ2 - EZ ** 2


    P_XY = np.sum(joint_probs, axis=2)  # axis = 2 , collapse Z (3rd elem of the join proba)
    # np.outer(X[0], Y[0]) produces a the matrix where each element is (xi * yi)  (2d)
    EXY = np.sum(P_XY * np.outer(X[0], Y[0]))

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
    steps = [
        (r"QUESTION 5B:", 0.9, 18),
        (r"If X, Y, Z are pairwise independent the covariance", 0.8, 15),
        (r"for each pair is 0 so:", 0.7, 15),
        (r"$V(X + Y + Z) = V(X) + V(Y) + V(Z)$", 0.6, 15),
    ]
    print_in_plot(steps)

    EX = np.sum(X[0] * X[1])
    EY = np.sum(Y[0] * Y[1])
    EZ = np.sum(Z[0] * Z[1])

    EX2 = np.sum((X[0] ** 2) * X[1])
    EY2 = np.sum((Y[0] ** 2) * Y[1])
    EZ2 = np.sum((Z[0] ** 2) * Z[1])

    VarX = EX2 - EX ** 2
    VarY = EY2 - EY ** 2
    VarZ = EZ2 - EZ ** 2

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
    steps = [
        (r"QUESTION 5C:", 0.9, 18),
        (r"The answer is NO we will use the following example:", 0.8, 15),
        (r"Let $X, Y \sim B(0.5)$ and $Z = X \oplus Y$", 0.7, 15),
        (r"$P(X=x, Z=z) = \frac{1}{4} = P(X=x) \cdot P(Z=z)$", 0.6, 15),
        (r"In the same way Y and Z are independent and X,Y independent", 0.5, 15),
        (r"So X,Y,Z are pairwise independent but:", 0.4, 15),
        (r"$P(X=0, Y=0, Z=1) = 0 \neq P(X=0) \cdot P(Y=0) \cdot P(Z=1)$", 0.3, 15),
    ]
    print_in_plot(steps)

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(Z.shape[0]):
                if not np.isclose(joint_probs[i, j, k], X[i] * Y[j] * Z[k]):
                    return False  # we dont have the collectively independent
    return True


### Question 6 ###

def expectedC(n, p):
    """
    The program outputs the expected value of the RV C as defined in the notebook.
    """
    prove = [
        ("QUESTION 6 - Prove:", 0.9, 18),
        (r"$E(C) = \sum_{w \in \Omega} C(w) \cdot P(w)$", 0.8, 16),
        (r"$C(w)$ is defined as the number of different strings with exactly $W(w)$ 1's.", 0.7, 12),
        (
            r"Let $k = W(w)$. Then, $C(w) = \binom{n}{k}$ and $P(w) = p^k \cdot (1-p)^{n-k}$ for independent tossing of a $p$-coin.",
            0.6, 12),
        (r"Thus, all sequences of exactly $k$ 1's are the same, so we sum over different $k$ 1's:", 0.5, 12),
        (
            r"$E(C) = \sum_{k=0}^{n} \binom{n}{k} \cdot \binom{n}{k} \cdot p^k \cdot (1-p)^{n-k} = \sum_{k=0}^{n} \binom{n}{k}^2 \cdot p^k \cdot (1-p)^{n-k}$",
            0.4, 16)
    ]

    print_in_plot(prove)
    E = 0
    for k in range(n + 1):
        E += special.comb(n, k) * stats.binom.pmf(k, n, p)

    return E
