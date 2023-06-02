import timeit
from functools import partial

import numpy as np
import torch
from scipy.stats import norm


def gpu_dgaussian_samples(scale, device, shape):
    N = (2 * scale ** 2)
    x = np.arange(- N, + N + 1, 1)
    xU, xL = x + 0.5, x - 0.5
    prob = norm.cdf(xU, scale=scale) - norm.cdf(xL, scale=scale)
    prob = torch.tensor(prob / prob.sum(), device=device)  # normalize the probabilities so their sum is 1

    # create a tensor of samples
    num_samples = int(np.prod(shape))
    samples = torch.multinomial(prob, num_samples=num_samples, replacement=True).to(torch.int)
    samples = samples.view(shape)
    return samples


def main():
    length = 13000
    samples = 100000
    num_times = 1

    print(f'Running on cuda...')
    squeeze = 2
    p = partial(gpu_dgaussian_samples, 1 / 0.01, 'cuda', (int(samples / squeeze), length))
    cuda_time = timeit.timeit(p, number=num_times * squeeze) / (num_times * squeeze)
    print(f'GPU time: {cuda_time:.3f} seconds')

    print(f'Running on cpu...')
    p = partial(gpu_dgaussian_samples, 1 / 0.01, 'cpu', (samples, length))
    cpu_time = timeit.timeit(p, number=num_times) / num_times
    print(f'CPU time: {cpu_time:.3f} seconds')


if __name__ == '__main__':
    main()
