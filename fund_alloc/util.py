import numpy as np
from scipy.stats import dlaplace, norm


def dlaplace_fn(loc, scale):
    return loc + dlaplace.rvs(1 / scale)


def dgaussian_fn(loc, scale):
    N = (2 * scale ** 2)
    x = np.arange(- N, + N + 1, 1)
    xU, xL = x + 0.5, x - 0.5
    prob = norm.cdf(xU, scale=scale) - norm.cdf(xL, scale=scale)
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    return loc + np.random.choice(x, size=1, p=prob).astype(int)
