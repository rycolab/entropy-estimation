from rayuela.base.semiring import Boolean, Real, Tropical, \
    String, Integer, Rational, Semiring, expectation_semiring_builder
from rayuela.base.symbol import Sym, ε
from rayuela.fsa.fsa import FSA
from rayuela.fsa.state import State
from rayuela.fsa.sampler import Sampler
from rayuela.fsa.pathsum import Pathsum, Strategy

from collections import defaultdict
import entropy
import math


def lift(old_fsa: FSA, func):
    # init semiring
    exp = expectation_semiring_builder(Real, Real)

    # lift our old fsa onto the expectation semiring
    fsa = FSA(exp)
    s, e = None, None
    for p, w in old_fsa.I:
        s = p
        fsa.set_I(p, w=exp.one)
    for p, w in old_fsa.F:
        e = p
        fsa.set_F(p, w=exp.one)
    for p in old_fsa.Q:
        for a, q, w in old_fsa.arcs(p):
            l = func(w)
            fsa.add_arc(p, a, q, w=exp(l[0], l[1]))
    return fsa

def fsa_from_samples(samples):
    """Generate MLE structure of FSA based on sampling x values"""
    delta = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)

    # construct new transition function
    for samp in samples:
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
    fsa.set_F(State(samples[0][-1]), Real(1.0))

    return fsa, samples, delta, tot

def estimate_entropy(fsa: FSA, samples, delta, ct, more=False):
    """Calculate the entropy estimates, given the samples and the correspondingly constructed MLE FSA"""
    res = defaultdict(float)

    # lift to expectation semiring
    entropy_fsa = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))

    # simple estimates to get
    res['Unstructured MLE'] = entropy.mle(*entropy.prob(samples))
    res['Structured MLE'] = float(entropy_fsa.pathsum().score[1])
    res['Unstructured NSB'] = entropy.nsb(*entropy.prob(samples))

    # structured NSB (or other) estimator
    for state in ct:
        N = sum(delta[state].values())
        ct_q = ct[state] / len(samples)
        dist_q = [x / N for x in delta[state].values()], N, delta[state]
        if more:
            for func in entropy.funcs:
                res[f'Structured {func.__name__}'] += ct_q * func(*dist_q)
        else:
            res['Structured NSB'] += ct_q * entropy.nsb(*dist_q)
    
    return res