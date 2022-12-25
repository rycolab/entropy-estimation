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

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

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
    """Make a cyclic FSA (homomorphic to a complete directed graph) with outgoing weights from a node summing to 1"""
    fsa = FSA(Real)
    for i in range(states - 1):
        # use Dirichlet to generate outgoing weights
        s = np.random.dirichlet([1.0] * states, 1).tolist()[0]
        for j in range(states):
            fsa.add_arc(State(i), Sym(j), State(j), Real(s[j]))

    fsa.set_I(State(0), Real(1.0))
    fsa.set_F(State(states - 1), Real(1.0))

    return fsa

def run_iter(fsa: FSA, samples=None, num_samps=1000, more=False):
    """Estimate structured and unstructured entropy given a true FSA and number of samples to get"""
    if not samples:
        samples = get_samples(fsa, num_samps)
    return estimate_entropy(*fsa_from_samples(samples), more=more)

def graph_convergence(states, cyclic=False, resample=True):
    """Graph convergence of entropy estimates to true value over sample size"""
    # make FSA and get true entropy
    fsa = make_cyclic_machine(states=states) if cyclic else make_acyclic_machine(states=states)
    lifted = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
    true = float(lifted.pathsum().score[1])

    # run sampling for various # of samples
    X = list(range(1, 100, 1))
    Ys = defaultdict(list)

    # if resample is True, then we generate a new sample every time, otherwise we keep one throughout
    s = None if resample else get_samples(fsa, 200)

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
    graph_convergence(states=10, cyclic=False, resample=True)

if __name__ == "__main__":
    main()