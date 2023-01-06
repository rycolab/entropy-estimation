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

def sample(pfsa, sampler):
    """Sample a path from the PFSA, including symbols on the arcs (for trajectory entropy)."""
    cur = sampler._draw({p : w for p, w in pfsa.I})
    output = [(None, cur)]

    while cur:
        D = {(a, j) : w for a, j, w in pfsa.arcs(cur)}
        D[(0, 0)] = pfsa.ρ[cur]

        (a, cur) = sampler._draw(D)
        if a != 0: output.append((a, cur))

    return tuple(output)

def get_samples(fsa: FSA, samples):
    sampler = Sampler(fsa)
    s = [sample(fsa, sampler) for _ in range(samples)]
    return s

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

    # construct new transition function, add dummy nodes to start and end
    for samp in samples:
        last = None
        tot[None] += 1
        for (symbol, state) in samp:
            delta[last][(symbol, state)] += 1
            tot[state] += 1
            last = state
        delta[last][(None, None)] += 1
    
    # make FSA
    fsa = FSA(Real)
    for i in delta:
        for (symbol, j) in delta[i]:
            # initials and finals (dummy Nones)
            if j is None: fsa.set_F(State(i), Real(1.0))
            elif i is None: fsa.set_I(State(j), Real(delta[i][(symbol, j)] / tot[i]))
            # actual transitions
            else: fsa.add_arc(State(i), Sym(symbol), State(j), Real(delta[i][(symbol, j)] / tot[i]))

    return fsa, samples, delta, tot

def estimate_entropy(fsa: FSA, samples, delta, ct, more=False):
    """Calculate the entropy estimates, given the samples and the correspondingly constructed MLE FSA
    - fsa: the FSA whose entropy is being estimated
    - samples: the samples from which the estimation is done
    - delta: transition function
    - more: whether to use only NSB or other estimators too
    """
    res = defaultdict(float)

    # lift to expectation semiring
    entropy_fsa = lift(fsa, lambda x: (x, Real(-float(x) * math.log(float(x)))))

    # simple estimates to get
    res['Unstructured MLE'] = entropy.mle(*entropy.prob(samples))
    res['Structured MLE (pathsum)'] = float(entropy_fsa.pathsum().score[1])
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
            res['Structured MLE'] += ct_q * entropy.mle(*dist_q)
    
    return res