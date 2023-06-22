import torch
from math import exp
from random import random
from abc import ABC, abstractmethod
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import wasserstein_distance
from numpy.polynomial.polynomial import Polynomial
from torch.distributions.normal import Normal
from torch.distributions.exponential import Exponential

from sympy import init_printing
import sympy.stats as ss
from sympy import Symbol, simplify, diff, cse
import sympy

class Proposal(ABC):
    symmetric = False
    def sample(self, prev=None):
        pass

    def log_prob(self, x, prev=None):
        pass

    def __str__(self) -> str:
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
    
    def _hist_plot(self, ax, **kwargs):
        x_arr = np.linspace(-15, 10, 100)
        pdf = np.exp([self.p.log_prob(torch.tensor(x)) for x in x_arr])
        ax.plot(x_arr, pdf, label="Target distribution")

        ax.hist(self.samples, density=True, bins=50)
        ax.set_title("Histogram")
        ax.set_xlabel("Sample value")
        ax.set_ylabel("Density")       
        return ax
    
    def plot(self, **kwargs):
        fig, axs = plt.subplots(1, 2, figsize=(15, 2.5))
        plot_funcs = [self._trace_plot, self._hist_plot]

        for ax, plot_func in zip(axs.flatten(), plot_funcs):
            plot_func(ax, **kwargs)
        
        plt.tight_layout()
        return axs
        

class IndependentGaussianProposal(Proposal):
    def __init__(self, mu, sigma):
        self.normal = Normal(mu, sigma)

    def log_prob(self, x, param=None):
        return self.normal.log_prob(x)
    
    def sample(self, prev=None):
        return self.normal.sample()
    
    def __str__(self) -> str:
        return "Independent Gaussian proposal"
    

class RandomWalkGaussianProposal(Proposal):
    def __init__(self, sigma):
        self.normal = Normal(0, sigma)
        self.symmetric = True

    def log_prob(self, x, prev):
        return self.normal.log_prob(x - prev)
    
    def sample(self, prev):
        return prev + self.normal.sample()
    
    def __str__(self) -> str:
        return "Random walk Gaussian proposal"


class AggregatePlotter:
    def sample(self, samples, repetitions, obj, window_width, **kwargs):
        self.window_width = window_width
        self.samples = samples
        self.store = []
        for i in range(repetitions):
            temp_obj = deepcopy(obj)
            temp_obj.sample(samples, **kwargs)
            self.store.append(temp_obj._wassterstein_distances(**kwargs))
        self.store = np.array(self.store)
    
    def plot(self):
        window_width = self.window_width
        print(window_width, self.samples, self.store.shape)
        x_arr = np.arange(window_width - 1, self.samples, window_width)
        mean_w_dists = np.mean(self.store, axis=0)
        print(x_arr.shape, mean_w_dists.shape)
        poly = Polynomial.fit(x_arr, mean_w_dists, 1)
        
        plt.plot(mean_w_dists)
        plt.plot(x_arr, poly(x_arr), label=f"Linear fit: {poly.__str__()}")


class Mixture:
    def __init__(self, distributions, weights=None):
        self.distributions = distributions
        self.weights = np.ones(len(distributions)) / len(distributions) if weights is None else np.array(weights) / sum(weights)
        self.sympy_weights = [sympy.Rational(w, sum(weights)) for w in weights]

        if all([isinstance(d, Normal) for d in self.distributions]):
            means = [sympy.nsimplify(d.mean) for d in self.distributions]
            stds = [sympy.nsimplify(d.stddev) for d in self.distributions]

            densities = [ss.density(ss.Normal("x", mean, std))(Symbol("x")) for mean, std in zip(means, stds)]
            expr = 0

            for d, w in zip(densities, self.sympy_weights):
                expr += d * w
            log_expr = sympy.log(expr).simplify().factor().cancel()
            grad = diff(log_expr, Symbol("x")).powsimp(force=True).simplify().factor().cancel()
            grad = grad.simplify().cancel().rewrite().factor().simplify().factor().cancel().factor()
            numerator, denominator = sympy.fraction(grad)
            grad = (numerator / sympy.exp(50)).expand() / (denominator / sympy.exp(50)).expand()
            grad.cancel().simplify()
            self.score = sympy.utilities.lambdify(Symbol("x"), grad, "numpy")
            print(grad)
        else:
            self.score = None
    
    def sample(self, n):
        counts = np.random.multinomial(n[0], self.weights)
        out = np.concatenate([d.sample([c]) for d, c in zip(self.distributions, counts)])
        np.random.shuffle(out)
        return out
    
    def log_prob(self, x):
        log_probs = np.array([d.log_prob(x) for d in self.distributions])
        probs = np.exp(log_probs)
        return np.log(np.sum(probs * self.weights))
    
    def plot_pdf(self, x_arr, starting=None, title=False):
        if starting is not None:
            plt.axvline(starting, color="red", label="Starting point", linestyle="--")
            plt.legend()
        plt.plot(x_arr, [np.exp(self.log_prob(x)) for x in x_arr])

        if title:
            plt.title("pdf of target distribution")
    
    # def score(self, x):
    #     if self.grad is None:
    #         raise NotImplementedError("Gradient is not implemented for this distribution")
        
    #     return self.grad(x)

        

class MALA(Proposal):
    def __init__(self, lr, grad_log_prob, str_rep="MALA"):
        self.lr = lr
        self.grad_log_prob = grad_log_prob
        self.str_rep = str_rep
    
    def log_prob(self, x, prev):
        dist = Normal(prev + self.lr * self.grad_log_prob(prev), np.sqrt(self.lr * 2))
        return dist.log_prob(x)

    def sample(self, prev):
        dist = Normal(prev + self.lr * self.grad_log_prob(prev), np.sqrt(self.lr * 2))
        return dist.sample()
    
    def __str__(self) -> str:
        return self.str_rep
    

class ULA(MetropolisHastings):
    def _step(self):
        # Sample from q
        x = self.q.sample(self.prev)
        self.prev = x
    
    def _acceptance_prob(self, x):
        pass

    def __str__(self) -> str:
        return "ULA"

class RepeatedSampler:
    def __init__(self, sampler_args, sampler=MetropolisHastings) -> None:
        self.sampler_args = sampler_args
        self.w_dists = None
        self.sampler = sampler
    
    def repeated_sample(self, repetitions, sample_length, x0=None, burnin=0, **kwargs):
        self.repetitions = repetitions
        self.samples = [self.sampler(**self.sampler_args) for _ in range(repetitions)]
        for i in range(repetitions):
            if callable(x0):
                self.samples[i].sample(sample_length, x0(), burnin, **kwargs)                
            else:
                self.samples[i].sample(sample_length, x0, burnin, **kwargs)

    def _average_wassterstein_distance(self, window_width):
        w_dists = []
        
        for sample in self.samples:
            w_dists.append(sample._wassterstein_distances(window_width))
        w_dists = np.array(w_dists)
        self.w_dists = w_dists
        mean_w_dists = np.mean(w_dists, axis=0)

        return mean_w_dists
    
    def _plot_average_wassterstein_distance(self, window_width, a=None, ax=None, fit=False, title=False):
        if ax is None:
            ax = plt.gca()
        
        mean_w_dists = self._average_wassterstein_distance(window_width)
        x_arr = np.arange(window_width - 1, len(self.samples[0].samples), window_width)
        

        std_w_dists = np.std(self.w_dists, axis=0) / np.sqrt(self.repetitions)

        ax.plot(x_arr, mean_w_dists, label=self.sampler_args["q"].__str__())
        ax.fill_between(x_arr, mean_w_dists - std_w_dists, mean_w_dists + std_w_dists, alpha=0.2)

        if fit:
            poly = Polynomial.fit(x_arr, mean_w_dists, 1)
            ax.plot(x_arr, poly(x_arr), label=f"Linear fit: {poly.__str__()}")
        if title:
            ax.set_title("Average Wasserstein distance plot")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Wasserstein distance")
        ax.legend(loc='upper right')


class RepeatedSamplerWrapper:
    def plot(self, repeated_samplers, sample_args, plot_args):
        fig, ax = plt.subplots()

        for sampler in repeated_samplers:
            sampler.repeated_sample(**sample_args)
            sampler._plot_average_wassterstein_distance(**plot_args, ax=ax)