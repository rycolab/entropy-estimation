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

def sample_fsa(orig: FSA, samples=1000):
    """Generate MLE structure of FSA based on sampling x values"""
    s = Sampler(orig).ancestral(samples)
    delta = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)

    # construct new transition function
    for samp in s:
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

    return fsa, s, delta, tot

def run_iter(orig: FSA, samples=1000):
    """Generate a random FSA and get true + estimated entropies"""

    # lift the acyclic random FSA to expectation semiring
    fsa = lift(orig, lambda x: (x, Real(-float(x) * math.log(float(x)))))

    # get new fsa
    orig_samp, samps, delta, ct = sample_fsa(orig, samples=samples)
    fsa_samp = lift(orig_samp, lambda x: (x, Real(-float(x) * math.log(float(x)))))

    # dumb_mle
    mle_dumb = entropy.mle(*entropy.prob(samps))

    # smart structured estimator
    res = 0.0
    for state in ct:
        N = sum(delta[state].values())
        res += (ct[state] / samples) * entropy.nsb([x / N for x in delta[state].values()], N, delta[state])

    # entropy pathsum
    return {'mle on FSA': float(fsa_samp.pathsum().score[1]), 'mle on samples': mle_dumb, 'structured nsb': res}

def graph_convergence(states=20):
    """Graph convergence of MLE to true"""
    # make FSA and get true entropy
    fsa = make_acyclic_machine(states=states)
    lifted = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
    true = float(lifted.pathsum().score[1])

    # run sampling for various # of samples
    X = list(range(1, 200, 2))
    Ys = defaultdict(list)
    for i in tqdm(X):
        res = run_iter(fsa, samples=i)
        for i in res:
            Ys[i].append(res[i])
    
    # plot
    for Y in Ys:
        plt.plot(X, Ys[Y], label=Y)
    plt.axhline(y=true, color='r', label='True')
    plt.xlabel('# Paths Sampled')
    plt.ylabel('Entropy (nats)')
    plt.title(f'# Samples vs. Entropy for random acyclic FSA with {states} states')
    plt.legend()
    plt.show()

def main():
    graph_convergence()

if __name__ == "__main__":
    main()