from collections import defaultdict
import math
from os import stat
from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational, Semiring, expectation_semiring_builder
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.fsa.sampler import Sampler
from rayuela.fsa.pathsum import Pathsum, Strategy
from rayuela.fsa.random import random_machine

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import lift
import entropy

def make_acyclic_machine(states=3):
    """Make an acyclic FSA (homomorphic to a DAG) with outgoing weights from a node summing to 1"""
    fsa = FSA(Real)
    for i in range(states):
        # use Dirichlet to generate outgoing weights
        s = np.random.dirichlet([1.0] * (states - i - 1), 1).tolist()[0]
        for w, j in zip(s, range(i + 1, states)):
            fsa.add_arc(State(i), Sym(j), State(j), Real(w))

    fsa.set_I(State(0), Real(1.0))
    fsa.set_F(State(states - 1), Real(1.0))

    return fsa

def make_cyclic_machine(states=3):
    """Make an cyclic FSA (homomorphic to a complete directed graph) with outgoing weights from a node summing to 1"""
    fsa = FSA(Real)
    for i in range(states - 1):
        # use Dirichlet to generate outgoing weights
        s = np.random.dirichlet([1.0] * states, 1).tolist()[0]
        for j in range(states):
            fsa.add_arc(State(i), Sym(j), State(j), Real(s[j]))

    fsa.set_I(State(0), Real(1.0))
    fsa.set_F(State(states - 1), Real(1.0))

    return fsa

def get_samples(fsa: FSA, samples):
    s = [Sampler(fsa)._ancestral(fsa) for _ in range(samples)]
    return s

def sample_fsa(orig: FSA, s=None, samples=1000):
    """Generate MLE structure of FSA based on sampling x values"""
    if not s:
        s = get_samples(orig, samples)
    delta = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)
    any = defaultdict(int)

    # construct new transition function
    for samp in s:
        unique = set(samp)
        for i in unique:
            any[i] += 1
        for i in range(len(samp)):
            l = samp[i - 1] if i > 0 else 0
            delta[l][samp[i]] += 1
            tot[l] += 1
    
    # make FSA
    fsa = FSA(Real)
    for i in delta:
        for j in delta[i]:
            fsa.add_arc(State(i), Sym(str(j)), State(j), Real(delta[i][j] / tot[i]))
    fsa.set_I(State(0), Real(1.0))
    fsa.set_F(State(s[0][-1]), Real(1.0))

    return fsa, s, delta, tot, any

def estimate_covariance(samps, state: State, P_q: float, H_q: float, partition=100):
    """Estimate the covariance based on a given partition size."""
    covariance = 0.0

    ct = 0
    for i in range(0, len(samps), partition):
        ct += 1
        subsamp = samps[i:i + partition]

        # estimate prob of state
        p_q = len([x for x in subsamp if state in x]) / len(subsamp)

        # estimate entropy of outgoing edges
        delta = []
        for s in subsamp:
            for i in range(len(s) - 1):
                if s[i] == state: delta.append(s[i + 1])
        h_q = entropy.nsb(*entropy.prob(delta))

        # update covariance
        covariance += (p_q - P_q) * (h_q - H_q)

    covariance /= ct
    return covariance

def run_iter(orig: FSA, samples=None, num_samps=1000):
    """Generate a random FSA and get true + estimated entropies"""
    res = defaultdict(float)

    # lift the acyclic random FSA to expectation semiring
    fsa = lift(orig, lambda x: (x, Real(-float(x) * math.log(float(x)))))

    # get new fsa
    orig_samp, samps, delta, ct, prob = sample_fsa(orig, samples, samples=num_samps)
    fsa_samp = lift(orig_samp, lambda x: (x, Real(-float(x) * math.log(float(x)))))

    res['Unstructured MLE'] = entropy.mle(*entropy.prob(samps))
    res['Structured MLE'] = float(fsa_samp.pathsum().score[1])
    res['Unstructured NSB'] = entropy.nsb(*entropy.prob(samps))

    # structured estimator
    for state in ct:
        N = sum(delta[state].values())
        ct_q = ct[state] / num_samps
        P_q = prob[state] / num_samps
        H_q = entropy.nsb([x / N for x in delta[state].values()], N, delta[state])

        # calculate covariance
        covariance = estimate_covariance(samps, state, P_q, H_q, partition=100)
            
        res['Structured NSB (ct)'] += ct_q * H_q
        res['Structured NSB (p)'] += P_q * H_q
        res['Structured NSB (p, +Covariance)'] += P_q * H_q + covariance

    # entropy pathsum
    return res

def graph_convergence(states, cyclic=False, resample=True):
    """Graph convergence of MLE to true"""
    # make FSA and get true entropy
    fsa = make_cyclic_machine(states=states) if cyclic else make_acyclic_machine(states=states)
    lifted = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
    true = float(lifted.pathsum().score[1])

    # run sampling for various # of samples
    X = list(range(1, 1000, 100))
    Ys = defaultdict(list)
    s = None
    if not resample:
        s = get_samples(fsa, 200)
    for i in tqdm(X):
        res = run_iter(fsa, samples=s[:i] if s else None, num_samps=i)
        for i in res:
            Ys[i].append(res[i])
    
    # plot
    for Y in Ys:
        plt.plot(X, Ys[Y], label=Y)
    plt.axhline(y=true, color='r', label='True')
    plt.xlabel('# Paths Sampled')
    plt.ylabel('Entropy (nats)')
    plt.title(f'# Samples vs. Entropy for random {"" if cyclic else "a"}cyclic FSA with {states} states')
    plt.legend()
    plt.show()

def main():
    graph_convergence(states=10, cyclic=True)

if __name__ == "__main__":
    main()