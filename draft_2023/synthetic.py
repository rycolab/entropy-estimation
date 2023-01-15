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
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import lift, fsa_from_samples, estimate_entropy, get_samples

from plotnine import ggplot, geom_line, geom_point, aes, stat_smooth, facet_wrap
from plotnine.scales import scale_y_log10

from scipy.stats import tukey_hsd

import logging
logger = logging.getLogger()
logger.disabled = True

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

def measure_convergence(states, sampling, cyclic, resample, fsas, prog=True):
    """Measure convergence of entropy estimates to true value over sample size for a single FSA"""
    # run sampling for various # of samples
    X = sampling
    results = []

    for f in tqdm(range(fsas)):
        # make FSA and get true entropy
        fsa = make_cyclic_machine(states=states) if cyclic else make_acyclic_machine(states=states)
        lifted = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))
        true = float(lifted.pathsum().score[1])

        # if resample is True, then we generate a new sample every time, otherwise we keep one throughout
        s = None if resample else get_samples(fsa, X[-1])

        for samp in (tqdm(X) if prog else X):
            res = run_iter(fsa, samples=s[:samp] if s else None, num_samps=samp)
            for key in res:
                results.append({
                    'samples': samp,
                    'method': key,
                    'mse': ((res[key] - true)**2),
                    'mab': (abs(res[key] - true)),
                    'cyclic': cyclic,
                    'states': states
                })
    
    return results
    
def graph_convergence(states: list[int], sampling: list[int], cyclic: bool, resample: bool, fsas: int, tukey: bool = False, graph: bool = False):

    # get results
    results = []
    sampling = sampling
    for state in states:
        results.extend(measure_convergence(state, sampling, cyclic, resample, fsas, prog=False))

    # significance test
    if tukey:
        grouped = defaultdict(lambda: defaultdict(list))
        for res in results:
            grouped[(res['states'], res['cyclic'], res['samples'])][res['method']].append(res)
        for group in grouped:
            arrs = [np.array([y['mse'] for y in x]) for x in grouped[group].values()]
            test = tukey_hsd(*arrs)
            print('===========================')
            print(group)
            print(list(grouped[group].keys()))
            for arr in arrs:
                print(arr.mean(), arr.std())
            print(test)
    
    # make graph
    if graph:
        df = pd.DataFrame(results)
        print(df)
        plot = (ggplot(df, aes(x='samples', y='mse', color='factor(method)'))
            + geom_line(stat='summary')
            + scale_y_log10())
        plot.draw(show=True)
        plot.save(filename='plots/synthetic.pdf', height=3, width=3)


def main():
    # graph_convergence(states=[2, 5, 10], sampling=[10, 100], cyclic=True, resample=False, fsas=200, tukey=True)
    graph_convergence(states=[5, 10, 50], sampling=[10, 100], cyclic=False, resample=False, fsas=200, tukey=True)

if __name__ == "__main__":
    main()