import torch
from math import exp
from random import random
from abc import ABC, abstractmethod
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from numpy.polynomial.polynomial import Polynomial
from torch.distributions.normal import Normal
from torch.distributions.exponential import Exponential


class Proposal(ABC):
    symmetric = False
    def sample(self, prev=None):
        pass

    def log_prob(self, x, prev=None):
        pass


class MetropolisHastings:
    """
    Metropolis Hastings algorithm for sampling from a distribution p

    Attributes
    ----------
    p : distribution to sample from
    q : proposal distribution
    prev : previous sample
    """
    def __init__(self, p, q : Proposal):
        self.p = p
        self.q = q
        self.prev = None
        self.samples = []
    
    def _acceptance_prob(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not isinstance(self.prev, torch.Tensor):
            self.prev = torch.tensor(self.prev)

        try:
            if not self.q.symmetric:
                a = self.p.log_prob(x) + self.q.log_prob(self.prev, x) - self.p.log_prob(self.prev) - self.q.log_prob(x, self.prev)
            else:
                a = self.p.log_prob(x) - self.p.log_prob(self.prev)
        except ValueError:
            return 0
        a = min(1, exp(a))
        return a        

    def _step(self):
        # Sample from q
        x = self.q.sample(self.prev)
        
        # Compute acceptance probability
        a = self._acceptance_prob(x)
        if a >= 1 or random() < a:
            self.prev = x

    def sample(self, n, x0, burnin=0, **kwargs):
        self.prev = x0
        for i in range(burnin):
            self._step()

        for i in range(n):
            self._step()
            self.samples.append(self.prev)
        return self.samples

    def _trace_plot(self, ax, **kwargs):
        ax.plot(self.samples)
        ax.set_title("Trace plot")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Sample value")
        return ax
    
    def _autocorrelation_plot(self, ax, **kwargs):
        ax.acorr(self.samples)
        ax.set_title("Autocorrelation plot")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        return ax
    
    def _wassterstein_distances(self, window_width = 100, **kwargs):
        w_dists = []
        target_samples = self.p.sample([len(self.samples)])

        for i in range(0 , len(self.samples) - window_width + 1, window_width):
            w_dists.append(wasserstein_distance(self.samples[i:i+window_width], target_samples[i:i+window_width]))
        return w_dists

    def _wasserstein_distance_plot(self, ax, window_width = 100, **kwargs):
        w_dists = self._wassterstein_distances(window_width, **kwargs)
        
        x_arr = np.arange(window_width - 1, len(self.samples), window_width)
        ax.plot(x_arr, w_dists, label="Wasserstein distance")

        # quadratic fit
        
        with np.printoptions(precision=2, suppress=True, formatter={'float_kind':'{:0.2f}'.format}):
            poly = Polynomial.fit(x_arr, w_dists, 1)
            ax.plot(x_arr, poly(x_arr), label=f"Linear fit: {poly.__str__()}")
            ax.set_title("Wasserstein distance plot")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Wasserstein distance")
            ax.legend()
        return ax
    
    def plot(self, **kwargs):
        fig, axs = plt.subplots(2, 2, figsize=(15, 5))
        plot_funcs = [self._trace_plot, self._autocorrelation_plot, self._wasserstein_distance_plot]

        for ax, plot_func in zip(axs.flatten(), plot_funcs):
            plot_func(ax, **kwargs)
        
        plt.tight_layout()
        return axs


class Mixture:
    def __init__(self, p1, p2, p1_weight=0.5):
        self.p1 = p1
        self.p2 = p2
        self.p1_weight = p1_weight

    def sample(self, arg):
        if random() < self.p1_weight:
            return self.p1.sample(arg)
        else:
            return self.p2.sample(arg)

    def log_prob(self, x, prev=None):
        return np.log(self.p1_weight * np.exp(self.p1.log_prob(x)) + (1 - self.p1_weight) * np.exp(self.p2.log_prob(x)))
    

class IndependentGaussianProposal(Proposal):
    def __init__(self, mu, sigma):
        self.normal = Normal(mu, sigma)

    def log_prob(self, x, param=None):
        return self.normal.log_prob(x)
    
    def sample(self, prev=None):
        return self.normal.sample()
    

class RandomWalkGaussianProposal(Proposal):
    def __init__(self, sigma):
        self.normal = Normal(0, sigma)
        self.symmetric = True

    def log_prob(self, x, prev):
        return self.normal.log_prob(x - prev)
    
    def sample(self, prev):
        return prev + self.normal.sample()


class AggregatePlotter:
    def sample(self, samples, repetitions, obj, **kwargs):
        self.store = []
        for i in range(repetitions):
            temp_obj = deepcopy(obj)
            temp_obj.sample(samples, **kwargs)
            self.store.append(temp_obj._wassterstein_distances(**kwargs))
        self.store = np.array(self.store)
    
    def plot(self):
        mean_w_dists = np.mean(self.store, axis=0)
        plt.plot(mean_w_dists)